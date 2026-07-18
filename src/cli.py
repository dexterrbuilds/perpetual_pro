"""CLI orchestration for Data Mode and Screen Mode."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger
from rich.console import Console
from rich.panel import Panel

from src.analysis.confluence import ConfluenceEngine, FullAnalysis
from src.data.exchange import ExchangeClient, list_supported_exchanges
from src.data.multi_tf import fetch_multi_timeframe
from src.data.news import NewsAnalyzer
from src.report.generator import ReportGenerator
from src.utils.config import AppConfig, load_config, setup_logging
from src.utils.helpers import normalize_symbol


console = Console()


def parse_region(region: str) -> Tuple[int, int, int, int]:
    parts = [p.strip() for p in region.split(",")]
    if len(parts) != 4:
        raise ValueError("Region must be L,T,W,H")
    left, top, width, height = map(int, parts)
    return left, top, width, height


def parse_higher_tfs(raw: Optional[str], config: AppConfig) -> List[str]:
    if not raw:
        return list(config.timeframes.higher)
    return [x.strip() for x in raw.split(",") if x.strip()]


def run_data_mode(
    symbol: str,
    config: AppConfig,
    exchange: Optional[str] = None,
    timeframe: Optional[str] = None,
    higher: Optional[str] = None,
    account_balance: Optional[float] = None,
    risk_pct: Optional[float] = None,
    no_news: bool = False,
    no_save: bool = False,
) -> FullAnalysis:
    """Primary data mode: live market data + full analysis stack."""
    ex_id = (exchange or config.exchange.default).lower()
    primary_tf = timeframe or config.timeframes.primary
    higher_tfs = parse_higher_tfs(higher, config)
    sym = normalize_symbol(symbol)

    console.print(
        Panel(
            f"[bold]Data Mode[/]\n"
            f"Symbol: [cyan]{sym}[/]\n"
            f"Exchange: [cyan]{ex_id}[/]\n"
            f"Primary TF: [cyan]{primary_tf}[/]  Higher: [cyan]{', '.join(higher_tfs)}[/]",
            title="perpetual_pro",
            border_style="cyan",
        )
    )

    client = ExchangeClient(exchange_id=ex_id, config=config)
    try:
        mtf = fetch_multi_timeframe(
            client,
            symbol=sym,
            primary_tf=primary_tf,
            higher_tfs=higher_tfs,
            limit=config.timeframes.ohlcv_limit,
            include_snapshot=True,
            config=config,
        )
        if mtf.primary.empty:
            raise RuntimeError(
                f"No OHLCV data for {sym} on {ex_id}. "
                f"Check symbol format (e.g. BTC/USDT:USDT) and exchange."
            )

        news_bundle = None
        if not no_news and config.news.enabled:
            with console.status("[bold green]Fetching news & sentiment..."):
                news_bundle = NewsAnalyzer(config=config).analyze(sym)

        with console.status("[bold green]Running pro confluence engine..."):
            engine = ConfluenceEngine(config)
            analysis = engine.analyze(
                mtf,
                news=news_bundle,
                simulated_capital=account_balance,
                risk_pct=risk_pct,
            )

        reporter = ReportGenerator(config, console=console)
        reporter.render(analysis)

        if not no_save and (config.output.save_json or config.output.save_markdown):
            paths = reporter.save(analysis)
            for kind, path in paths.items():
                console.print(f"[dim]Saved {kind}: {path}[/]")

        return analysis
    finally:
        client.close()


def run_screen_mode(
    config: AppConfig,
    symbol: Optional[str] = None,
    exchange: Optional[str] = None,
    timeframe: Optional[str] = None,
    higher: Optional[str] = None,
    region: Optional[str] = None,
    image_path: Optional[str] = None,
    full_screen: bool = False,
    account_balance: Optional[float] = None,
    risk_pct: Optional[float] = None,
    no_news: bool = False,
    no_save: bool = False,
    no_annotate: bool = False,
) -> FullAnalysis:
    """Screen mode: capture chart → OCR/CV → data fallback analysis."""
    from src.vision.capture import ScreenCapture
    from src.vision.chart_detect import ChartVision
    from src.vision.ocr import OCREngine
    from src.vision.preprocess import annotate_levels

    console.print(
        Panel(
            "[bold]Screen Mode[/]\n"
            "Capture → preprocess → dual OCR → CV patterns → full data analysis fallback",
            title="perpetual_pro",
            border_style="magenta",
        )
    )

    cap = ScreenCapture(output_dir=config.resolve_path(config.screen.output_dir))
    if image_path:
        capture = cap.capture_from_file(image_path)
    elif region:
        left, top, width, height = parse_region(region)
        capture = cap.capture_region(left, top, width, height, save=config.screen.save_capture)
    elif full_screen:
        capture = cap.capture_full(monitor=1, save=config.screen.save_capture)
    else:
        console.print("[yellow]Select the chart region on your screen...[/]")
        capture = cap.capture_interactive(save=config.screen.save_capture)

    if capture.path:
        console.print(f"[dim]Capture saved: {capture.path}[/]")

    dark = config.screen.dark_theme
    ocr = OCREngine(config=config)
    with console.status("[bold green]Running dual OCR..."):
        ocr_result = ocr.extract(capture.image, dark_theme=dark)

    vision = ChartVision(config=config)
    with console.status("[bold green]Computer vision chart analysis..."):
        vis = vision.analyze(capture.image, dark_theme=dark)

    # Resolve symbol / TF
    resolved_symbol = symbol
    if not resolved_symbol and ocr_result.symbol:
        resolved_symbol = ocr_result.symbol
    if not resolved_symbol:
        console.print(
            "[red]Could not detect symbol from screen. "
            "Pass --symbol BTC/USDT:USDT explicitly.[/]"
        )
        raise SystemExit(2)

    try:
        resolved_symbol = normalize_symbol(resolved_symbol)
    except ValueError:
        pass

    primary_tf = timeframe or ocr_result.timeframe or config.timeframes.primary
    ex_id = (exchange or config.exchange.default).lower()
    higher_tfs = parse_higher_tfs(higher, config)

    vision_notes = (
        f"OCR: symbol={ocr_result.symbol} tf={ocr_result.timeframe} "
        f"conf={ocr_result.confidence:.2f} engines={', '.join(ocr_result.engine_notes)}\n"
        f"CV: candles={vis.candles_detected} trend≈{vis.trend_guess} "
        f"conf={vis.confidence:.2f}\n"
        f"CV notes: {'; '.join(vis.notes)}\n"
    )
    if vis.ollama_summary:
        vision_notes += f"\nOllama: {vis.ollama_summary}\n"
    if ocr_result.indicators_mentioned:
        vision_notes += f"Indicators visible: {', '.join(ocr_result.indicators_mentioned)}\n"

    console.print(Panel(vision_notes, title="Vision / OCR Result", border_style="white"))

    # Full data analysis fallback (always preferred for accuracy)
    client = ExchangeClient(exchange_id=ex_id, config=config)
    try:
        with console.status(f"[bold green]Fetching live data for {resolved_symbol}..."):
            mtf = fetch_multi_timeframe(
                client,
                symbol=resolved_symbol,
                primary_tf=primary_tf,
                higher_tfs=higher_tfs,
                limit=config.timeframes.ohlcv_limit,
                include_snapshot=True,
                config=config,
            )

        if mtf.primary.empty:
            console.print(
                "[yellow]Live data failed — report will rely more on vision/OCR only.[/]"
            )
            # Minimal synthetic analysis shell
            from src.analysis.confluence import FullAnalysis
            from src.analysis.risk import RiskManager

            analysis = FullAnalysis(
                symbol=resolved_symbol,
                exchange_id=ex_id,
                primary_tf=primary_tf,
            )
            analysis.bias = (
                "bullish"
                if vis.trend_guess == "up"
                else ("bearish" if vis.trend_guess == "down" else "neutral")
            )
            analysis.direction = (
                "long"
                if analysis.bias == "bullish"
                else ("short" if analysis.bias == "bearish" else "flat")
            )
            analysis.confidence = max(25.0, vis.confidence * 100 * 0.6)
            analysis.setup_name = "Vision-only (data unavailable)"
            analysis.trader_commentary = (
                vision_notes
                + " Live market data unavailable; treat this as low-confidence visual read only."
            )
            analysis.warnings.append("Data fallback failed — vision-only mode")
            price = ocr_result.prices[len(ocr_result.prices) // 2] if ocr_result.prices else 0.0
            analysis.meta = {"price": price, "atr": price * 0.01 if price else 0}
            if price:
                analysis.trade_plan = RiskManager(config=config).build_plan(
                    analysis.direction, price, price * 0.01, confidence=analysis.confidence
                )
        else:
            news_bundle = None
            if not no_news and config.news.enabled:
                news_bundle = NewsAnalyzer(config=config).analyze(resolved_symbol)

            # Soft-blend vision trend into warnings if conflicts
            engine = ConfluenceEngine(config)
            analysis = engine.analyze(
                mtf,
                news=news_bundle,
                simulated_capital=account_balance,
                risk_pct=risk_pct,
            )
            if vis.trend_guess in ("up", "down"):
                visual_bias = "bullish" if vis.trend_guess == "up" else "bearish"
                if visual_bias != analysis.bias and analysis.bias != "neutral":
                    analysis.warnings.append(
                        f"Screen trend guess ({vis.trend_guess}) conflicts with data bias "
                        f"({analysis.bias}) — trust data more; verify chart symbol/TF."
                    )
                elif analysis.bias == "neutral" and vis.confidence > 0.4:
                    analysis.warnings.append(
                        f"Data neutral; screen suggests {vis.trend_guess}. Still wait for confirmation."
                    )

        reporter = ReportGenerator(config, console=console)
        reporter.render(analysis, vision_notes=vision_notes)

        extra = {
            "vision_notes": vision_notes,
            "ocr": {
                "symbol": ocr_result.symbol,
                "timeframe": ocr_result.timeframe,
                "confidence": ocr_result.confidence,
                "prices": ocr_result.prices[:20],
            },
            "cv": {
                "candles": vis.candles_detected,
                "trend_guess": vis.trend_guess,
                "confidence": vis.confidence,
                "notes": vis.notes,
                "ollama": vis.ollama_summary,
            },
            "capture_path": str(capture.path) if capture.path else None,
        }

        # Annotate capture
        if (
            config.screen.annotate
            and not no_annotate
            and analysis.trade_plan
            and capture.image
        ):
            try:
                plan = analysis.trade_plan
                annotated = annotate_levels(
                    capture.image,
                    levels=analysis.key_levels,
                    entry=(plan.entry_low, plan.entry_high),
                    stop=plan.stop_loss,
                    tps=plan.take_profits,
                )
                out_dir = config.resolve_path(config.screen.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                ann_path = out_dir / f"annotated_{Path(capture.path).stem if capture.path else 'chart'}.png"
                annotated.save(ann_path)
                extra["annotated_path"] = str(ann_path)
                console.print(f"[dim]Annotated chart: {ann_path}[/]")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Annotation failed: {}", exc)

        if not no_save and (config.output.save_json or config.output.save_markdown):
            paths = reporter.save(analysis, extra=extra)
            for kind, path in paths.items():
                console.print(f"[dim]Saved {kind}: {path}[/]")

        return analysis
    finally:
        client.close()


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="perpetual_pro",
        description=(
            "Professional crypto perpetual futures analysis CLI — "
            "live data mode or screen capture mode."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py BTC/USDT:USDT
  python main.py ETH --exchange bybit --timeframe 5m --higher 15m,1h,4h
  python main.py --screen
  python main.py --screen --symbol SOL --region 100,100,1200,800
  python main.py --screen --image ./chart.png --exchange binanceusdm
        """,
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default=None,
        help="Symbol (e.g. BTC, BTCUSDT, BTC/USDT:USDT). Optional in --screen if OCR works.",
    )
    parser.add_argument(
        "--screen",
        action="store_true",
        help="Screen mode: capture chart, OCR + CV, then full analysis",
    )
    parser.add_argument(
        "--exchange",
        "-e",
        default=None,
        help=f"Exchange: {', '.join(list_supported_exchanges())}",
    )
    parser.add_argument("--timeframe", "-t", default=None, help="Primary timeframe (e.g. 15m)")
    parser.add_argument(
        "--higher",
        default=None,
        help="Higher timeframes comma-separated (e.g. 1h,4h,1d)",
    )
    parser.add_argument("--config", "-c", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--balance",
        type=float,
        default=None,
        help="Simulated capital for position sizing (default $1000)",
    )
    parser.add_argument(
        "--risk",
        type=float,
        default=None,
        help="Risk percent of simulated capital (default 1.0)",
    )
    parser.add_argument("--no-news", action="store_true", help="Skip news fetch")
    parser.add_argument("--no-save", action="store_true", help="Do not save MD/JSON reports")
    parser.add_argument(
        "--region",
        default=None,
        help="Screen region L,T,W,H (pixels). Implies --screen",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Analyze an existing chart image path. Implies --screen",
    )
    parser.add_argument(
        "--full-screen",
        action="store_true",
        help="Capture full primary monitor (no interactive crop). Implies --screen",
    )
    parser.add_argument("--no-annotate", action="store_true", help="Skip annotated chart image")
    parser.add_argument(
        "--list-exchanges",
        action="store_true",
        help="List supported exchanges and exit",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="DEBUG logging")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_exchanges:
        console.print("Supported exchanges:")
        for ex in list_supported_exchanges():
            console.print(f"  • {ex}")
        return 0

    config = load_config(args.config)
    if args.verbose:
        config.logging.level = "DEBUG"
    setup_logging(config)

    screen_mode = args.screen or args.region or args.image or args.full_screen

    try:
        if screen_mode:
            run_screen_mode(
                config=config,
                symbol=args.symbol,
                exchange=args.exchange,
                timeframe=args.timeframe,
                higher=args.higher,
                region=args.region,
                image_path=args.image,
                full_screen=args.full_screen,
                account_balance=args.balance,
                risk_pct=args.risk,
                no_news=args.no_news,
                no_save=args.no_save,
                no_annotate=args.no_annotate,
            )
        else:
            if not args.symbol:
                parser.error("symbol is required in data mode (or use --screen)")
            run_data_mode(
                symbol=args.symbol,
                config=config,
                exchange=args.exchange,
                timeframe=args.timeframe,
                higher=args.higher,
                account_balance=args.balance,
                risk_pct=args.risk,
                no_news=args.no_news,
                no_save=args.no_save,
            )
        return 0
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/]")
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal error: {}", exc)
        console.print(f"[bold red]Error:[/] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

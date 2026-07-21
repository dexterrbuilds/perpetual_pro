"""Beautiful Rich terminal reports + Markdown/JSON export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich.rule import Rule

from src.analysis.confluence import FullAnalysis
from src.utils.config import AppConfig
from src.utils.helpers import ensure_dir, format_price, slugify, utc_now_iso


DISCLAIMER = (
    "NOT FINANCIAL ADVICE. This tool is for educational and research purposes only. "
    "Crypto perpetual futures are highly leveraged and risky. You can lose more than "
    "your initial margin. Always do your own research and never risk capital you cannot "
    "afford to lose."
)


class ReportGenerator:
    def __init__(self, config: AppConfig, console: Optional[Console] = None) -> None:
        self.config = config
        self.console = console or Console(highlight=False)

    def render(self, analysis: FullAnalysis, vision_notes: Optional[str] = None) -> None:
        """Print full pro report to the terminal."""
        c = self.console
        c.print()
        c.print(
            Panel(
                Text.from_markup(
                    f"[bold cyan]PERPETUAL PRO[/]  ·  {analysis.symbol}  ·  "
                    f"{analysis.exchange_id}  ·  {analysis.primary_tf}\n"
                    f"[dim]{analysis.generated_at}[/]"
                ),
                border_style="cyan",
                box=box.DOUBLE,
            )
        )

        # Bias header
        bias_color = {
            "bullish": "green",
            "bearish": "red",
            "neutral": "yellow",
        }.get(analysis.bias, "white")
        header = Table.grid(padding=(0, 2))
        header.add_column(style="bold")
        header.add_column()
        header.add_row("Overall Bias", f"[{bias_color}]{analysis.bias.upper()}[/]")
        header.add_row("Confidence", f"[bold]{analysis.confidence:.1f}%[/]")
        header.add_row("Setup", analysis.setup_name or "—")
        header.add_row("Tags", ", ".join(analysis.strategy_tags) or "—")
        header.add_row("Confluence", f"{analysis.confluence_total:+.3f}")
        c.print(Panel(header, title="Bias & Setup", border_style=bias_color))

        # Pro trader setup card
        plan = analysis.trade_plan
        price = analysis.meta.get("price")
        if plan:
            pro_lines = plan.to_pro_lines(analysis.symbol, price)
            headline = pro_lines[0] if pro_lines else "SETUP"
            body = "\n".join(pro_lines[1:]) if len(pro_lines) > 1 else "—"
            c.print(
                Panel(
                    f"[bold white]{headline}[/]\n\n{body}",
                    title="🚨 Primary Setup",
                    border_style="magenta",
                    box=box.HEAVY,
                )
            )
            if plan.notes:
                for n in plan.notes[:6]:
                    c.print(f"  [dim]• {n}[/]")

        # Confluence breakdown
        ft = Table(title="Confluence Score Breakdown", box=box.MINIMAL_DOUBLE_HEAD)
        ft.add_column("Factor")
        ft.add_column("Score", justify="right")
        ft.add_column("Weight", justify="right")
        ft.add_column("Weighted", justify="right")
        ft.add_column("Detail")
        for f in analysis.factors:
            sc = f.score
            color = "green" if sc > 0.1 else ("red" if sc < -0.1 else "yellow")
            ft.add_row(
                f.name,
                f"[{color}]{sc:+.3f}[/]",
                f"{f.weight:.2f}",
                f"{f.weighted:+.3f}",
                (f.detail or "")[:80],
            )
        c.print(ft)

        # Patterns
        if analysis.patterns and analysis.patterns.hits:
            ptbl = Table(title="Key Patterns", box=box.SIMPLE)
            ptbl.add_column("Pattern")
            ptbl.add_column("Type")
            ptbl.add_column("Bias")
            ptbl.add_column("Conf %", justify="right")
            ptbl.add_column("Note")
            for h in analysis.patterns.top_hits:
                bc = "green" if h.bias == "bullish" else ("red" if h.bias == "bearish" else "yellow")
                ptbl.add_row(h.name, h.kind, f"[{bc}]{h.bias}[/]", f"{h.confidence:.0f}", h.note[:50])
            c.print(ptbl)

        # Market structure
        if analysis.structure:
            st = analysis.structure
            body = (
                f"[bold]{st.summary}[/]\n"
                f"Wyckoff: {st.wyckoff_phase}\n"
                f"{st.wyckoff_notes}\n"
                f"{st.elliott_notes}"
            )
            if st.volume_profile_poc:
                body += (
                    f"\nVolume profile POC≈{format_price(st.volume_profile_poc, price)}"
                    f"  VA=[{format_price(st.volume_profile_val or 0, price)} – "
                    f"{format_price(st.volume_profile_vah or 0, price)}]"
                )
            c.print(Panel(body, title="Market Structure", border_style="blue"))

        # Key levels
        if analysis.key_levels:
            lt = Table(title="Key Levels (nearest)", box=box.SIMPLE)
            lt.add_column("Kind")
            lt.add_column("Side")
            lt.add_column("Mid")
            lt.add_column("Dist %", justify="right")
            lt.add_column("Note")
            for lv in analysis.key_levels[:8]:
                lt.add_row(
                    str(lv.get("kind")),
                    str(lv.get("side")),
                    format_price(float(lv.get("mid", 0)), price),
                    f"{float(lv.get('distance_pct', 0)):+.2f}%",
                    str(lv.get("note", ""))[:40],
                )
            c.print(lt)

        # Derivatives
        if analysis.derivatives_notes:
            c.print(
                Panel(
                    "\n".join(f"• {n}" for n in analysis.derivatives_notes),
                    title="Derivatives",
                    border_style="bright_cyan",
                )
            )

        # Multi-TF
        if analysis.multi_tf_notes:
            c.print(
                Panel(
                    "\n".join(f"• {n}" for n in analysis.multi_tf_notes),
                    title="Multi-Timeframe",
                    border_style="bright_blue",
                )
            )

        # News
        if analysis.news:
            news_lines = [analysis.news.summary or ""]
            for item in analysis.news.top(5):
                news_lines.append(
                    f"  [{item.sentiment_score:+.2f}] {item.title[:100]}  ({item.source})"
                )
            c.print(Panel("\n".join(news_lines), title="News Impact", border_style="bright_yellow"))

        # Scenarios
        if analysis.scenarios:
            sc = analysis.scenarios
            stbl = Table(title="Scenarios", box=box.ROUNDED)
            stbl.add_column("Scenario")
            stbl.add_column("Prob %", justify="right")
            stbl.add_column("Trigger")
            stbl.add_column("Target")
            stbl.add_column("Invalidation")
            for row in (sc.bullish, sc.base, sc.bearish):
                stbl.add_row(
                    row["name"],
                    f"{row['probability']:.1f}",
                    str(row["trigger"])[:40],
                    str(row["target"]),
                    str(row["invalidation"])[:24],
                )
            c.print(stbl)

        # Divergences
        if analysis.indicators and analysis.indicators.divergences:
            c.print(Rule("Divergences"))
            for d in analysis.indicators.divergences:
                col = "green" if d.kind == "bullish" else "red"
                c.print(
                    f"  [{col}]{d.kind.upper()}[/] {d.oscillator} "
                    f"({d.confidence:.0f}%) — {d.note}"
                )

        # Vision
        if vision_notes:
            c.print(Panel(vision_notes, title="Screen / Vision", border_style="white"))

        # Commentary
        c.print(
            Panel(
                analysis.trader_commentary or "—",
                title="Trader Commentary",
                border_style="green",
            )
        )

        if analysis.warnings:
            c.print(Panel("\n".join(analysis.warnings), title="Warnings", border_style="red"))

        if self.config.output.show_disclaimer:
            c.print(Panel(DISCLAIMER, title="Disclaimer", border_style="dim", box=box.SQUARE))
        c.print()

    def to_dict(self, analysis: FullAnalysis, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        plan = analysis.trade_plan
        data: Dict[str, Any] = {
            "app": "perpetual_pro",
            "generated_at": analysis.generated_at,
            "symbol": analysis.symbol,
            "exchange": analysis.exchange_id,
            "primary_tf": analysis.primary_tf,
            # Clear trading signal
            "signal": {
                "bias": analysis.bias,
                "direction": analysis.direction,
                "confidence_pct": analysis.confidence,
                "technical_confidence": getattr(analysis, "technical_confidence", analysis.confidence),
                "llm_confidence": getattr(analysis, "llm_confidence", 0.0),
                "llm_confidence_reason": getattr(analysis, "llm_confidence_reason", ""),
                "llm_confidence_detail": getattr(analysis, "llm_confidence_detail", {}) or {},
                "rank_score": getattr(analysis, "rank_score", 0.0),
                "setup_name": analysis.setup_name,
                "strategy_tags": analysis.strategy_tags,
                "confluence_score": analysis.confluence_total,
            },
            "bias": analysis.bias,
            "direction": analysis.direction,
            "confidence": analysis.confidence,
            "technical_confidence": getattr(analysis, "technical_confidence", analysis.confidence),
            "llm_confidence": getattr(analysis, "llm_confidence", 0.0),
            "llm_confidence_reason": getattr(analysis, "llm_confidence_reason", ""),
            "llm_confidence_detail": getattr(analysis, "llm_confidence_detail", {}) or {},
            "rank_score": getattr(analysis, "rank_score", 0.0),
            "setup_name": analysis.setup_name,
            "strategy_tags": analysis.strategy_tags,
            "confluence_total": analysis.confluence_total,
            "confluence_breakdown": analysis.factor_breakdown(),
            "factors": analysis.factor_breakdown(),
            "primary_setup": None,
            "position_simulation": None,
            "trade_plan": None,
            "patterns": [],
            "structure": None,
            "key_levels": analysis.key_levels,
            "key_reasons": list(getattr(analysis, "key_reasons", None) or []),
            "key_risks": list(getattr(analysis, "key_risks", None) or []),
            "derivatives_notes": analysis.derivatives_notes,
            "multi_tf_notes": analysis.multi_tf_notes,
            "news": None,
            "scenarios": None,
            "trader_commentary": analysis.trader_commentary,
            "llm_narrative": None,
            "warnings": analysis.warnings,
            "meta": analysis.meta,
            "disclaimer": DISCLAIMER,
        }
        if plan:
            setup = plan.to_primary_setup()
            setup["headline"] = plan.setup_headline(analysis.symbol)
            setup["pro_lines"] = plan.to_pro_lines(
                analysis.symbol, analysis.meta.get("price") if analysis.meta else None
            )
            data["primary_setup"] = setup
            data["signal"]["headline"] = setup["headline"]
            data["signal"]["hold_label"] = getattr(plan, "hold_label", "")
            data["signal"]["hold_detail"] = getattr(plan, "hold_detail", "")
            data["signal"]["leverage_suggested"] = plan.leverage_suggested
            data["position_simulation"] = plan.to_position_simulation()
            data["trade_plan"] = {
                "direction": plan.direction,
                "entry_low": plan.entry_low,
                "entry_high": plan.entry_high,
                "stop_loss": plan.stop_loss,
                "take_profits": plan.take_profits,
                "risk_reward": plan.risk_reward,
                "alternative_entry_low": getattr(plan, "alternative_entry_low", None),
                "alternative_entry_high": getattr(plan, "alternative_entry_high", None),
                "alternative_entry_note": getattr(plan, "alternative_entry_note", ""),
                "position_size_units": plan.position_size_units,
                "position_size_notional": plan.position_size_notional,
                "margin_required": getattr(plan, "margin_required", None),
                "risk_amount": plan.risk_amount,
                "risk_pct": plan.risk_pct,
                "simulated_capital": getattr(plan, "simulated_capital", None),
                "leverage_suggested": plan.leverage_suggested,
                "leverage_reasoning": getattr(plan, "leverage_reasoning", ""),
                "potential_profits": getattr(plan, "potential_profits", []),
                "potential_profit_pcts": getattr(plan, "potential_profit_pcts", []),
                "quality": plan.quality,
                "notes": plan.notes,
                "invalidation": plan.invalidation,
                "hold_label": getattr(plan, "hold_label", ""),
                "hold_detail": getattr(plan, "hold_detail", ""),
                "hold_hours_max": getattr(plan, "hold_hours_max", 24.0),
                "headline": setup["headline"],
                "pro_lines": setup["pro_lines"],
                "is_simulation": True,
                "prop_mode": getattr(plan, "prop_mode", True),
                "prop_safe": getattr(plan, "prop_safe", True),
                "prop_flags": list(getattr(plan, "prop_flags", None) or []),
                "max_leverage_allowed": getattr(plan, "max_leverage_allowed", 5.0),
            }
            data["prop_safe"] = getattr(plan, "prop_safe", True)
            data["prop_flags"] = list(getattr(plan, "prop_flags", None) or [])
        if getattr(analysis, "llm", None):
            llm = analysis.llm
            data["llm_narrative"] = {
                "signal_narrative": llm.signal_narrative,
                "llm_confidence": getattr(llm, "llm_confidence", analysis.llm_confidence),
                "confidence_reason": getattr(llm, "confidence_reason", analysis.llm_confidence_reason),
                "confidence_detail": getattr(llm, "confidence_detail", None)
                or getattr(analysis, "llm_confidence_detail", {})
                or {},
                "leverage_reasoning": llm.leverage_reasoning,
                "key_reasons": llm.key_reasons,
                "key_risks": llm.key_risks,
                "scenarios": llm.scenarios,
                "provider": llm.provider,
                "model": llm.model,
            }
        else:
            # Still surface LLM confidence fields even when narrative object missing
            data["llm_narrative"] = {
                "llm_confidence": getattr(analysis, "llm_confidence", 0.0),
                "confidence_reason": getattr(analysis, "llm_confidence_reason", ""),
                "provider": "heuristic" if getattr(analysis, "llm_confidence", 0) else "none",
                "model": "",
            }
        if analysis.patterns:
            data["patterns"] = [
                {
                    "name": h.name,
                    "kind": h.kind,
                    "bias": h.bias,
                    "confidence": h.confidence,
                    "note": h.note,
                }
                for h in analysis.patterns.hits
            ]
        if analysis.structure:
            s = analysis.structure
            data["structure"] = {
                "trend": s.trend,
                "last_bos": s.last_bos,
                "last_choch": s.last_choch,
                "structure_score": s.structure_score,
                "wyckoff_phase": s.wyckoff_phase,
                "wyckoff_notes": s.wyckoff_notes,
                "elliott_notes": s.elliott_notes,
                "volume_profile_poc": s.volume_profile_poc,
                "volume_profile_vah": s.volume_profile_vah,
                "volume_profile_val": s.volume_profile_val,
                "summary": s.summary,
            }
        if analysis.news:
            data["news"] = {
                "bias": analysis.news.bias,
                "aggregate_sentiment": analysis.news.aggregate_sentiment,
                "summary": analysis.news.summary,
                "items": [
                    {
                        "title": i.title,
                        "source": i.source,
                        "url": i.url,
                        "sentiment_score": i.sentiment_score,
                    }
                    for i in analysis.news.items
                ],
            }
        if analysis.scenarios:
            data["scenarios"] = {
                "bullish": analysis.scenarios.bullish,
                "base": analysis.scenarios.base,
                "bearish": analysis.scenarios.bearish,
            }
        if analysis.indicators:
            data["indicator_summary"] = analysis.indicators.summary
            data["divergences"] = [
                {
                    "kind": d.kind,
                    "oscillator": d.oscillator,
                    "confidence": d.confidence,
                    "note": d.note,
                }
                for d in analysis.indicators.divergences
            ]
        if analysis.snapshot:
            snap = analysis.snapshot
            data["snapshot"] = {
                "last": snap.last,
                "funding_rate": snap.funding_rate,
                "open_interest": snap.open_interest,
                "long_short_ratio": snap.long_short_ratio,
                "percentage_24h": snap.percentage_24h,
            }
        if extra:
            data["extra"] = extra
        return data

    def to_markdown(self, analysis: FullAnalysis, extra: Optional[Dict[str, Any]] = None) -> str:
        price = analysis.meta.get("price")
        plan = analysis.trade_plan
        lines = [
            f"# perpetual_pro Report — {analysis.symbol}",
            "",
            f"- **Generated:** {analysis.generated_at}",
            f"- **Exchange:** {analysis.exchange_id}",
            f"- **Timeframe:** {analysis.primary_tf}",
            f"- **Bias:** {analysis.bias.upper()} ({analysis.confidence:.1f}% confidence)",
            f"- **LLM Confidence:** {getattr(analysis, 'llm_confidence', 0):.0f}%",
            f"- **LLM reason:** {getattr(analysis, 'llm_confidence_reason', '') or '—'}",
            f"- **Technical confidence:** {getattr(analysis, 'technical_confidence', analysis.confidence):.1f}%",
            f"- **Rank score:** {getattr(analysis, 'rank_score', 0):.1f}",
            f"- **Setup:** {analysis.setup_name}",
            f"- **Tags:** {', '.join(analysis.strategy_tags)}",
            f"- **Confluence:** {analysis.confluence_total:+.3f}",
            "",
            "## LLM Confidence",
            "",
        ]
        detail = getattr(analysis, "llm_confidence_detail", None) or {}
        lines.append(f"- **Score:** {getattr(analysis, 'llm_confidence', 0):.0f}%")
        lines.append(f"- **Headline:** {getattr(analysis, 'llm_confidence_reason', '') or '—'}")
        if detail.get("verdict"):
            lines.append(f"- **Verdict:** {detail.get('verdict')}")
        if detail.get("summary"):
            lines.append(f"- **Why:** {detail.get('summary')}")
        for s in detail.get("supporting") or []:
            lines.append(f"- ✅ {s}")
        for o in detail.get("opposing") or []:
            lines.append(f"- ⚠️ {o}")
        lines += ["", "## 🚨 Primary Setup", ""]
        if plan:
            for line in plan.to_pro_lines(analysis.symbol, price):
                lines.append(f"**{line}**" if line.startswith("🚨") or line.startswith("⏸") else f"- {line}")
            lines += [
                "",
                f"- **Sim leverage:** {plan.leverage_suggested:.0f}x (max {getattr(plan, 'max_leverage_allowed', 5):.0f}x)",
                f"- **Risk:** {plan.risk_pct:.2f}%",
                f"- **Prop-safe:** {getattr(plan, 'prop_safe', True)}",
                f"- **Quality:** {plan.quality}",
                "",
            ]
            flags = getattr(plan, "prop_flags", None) or []
            if flags:
                lines.append(f"- **Flags:** {', '.join(flags)}")
                lines.append("")
        else:
            lines.append("_No trade plan_\n")

        lines += ["## Confluence Breakdown", ""]
        lines.append("| Factor | Score | Weight | Weighted | Detail |")
        lines.append("|---|---:|---:|---:|---|")
        for f in analysis.factors:
            lines.append(
                f"| {f.name} | {f.score:+.3f} | {f.weight:.2f} | {f.weighted:+.3f} | {f.detail[:60]} |"
            )
        lines.append("")

        if analysis.patterns and analysis.patterns.hits:
            lines += ["## Patterns", ""]
            for h in analysis.patterns.top_hits:
                lines.append(f"- **{h.name}** ({h.kind}, {h.bias}, {h.confidence:.0f}%): {h.note}")
            lines.append("")

        if analysis.structure:
            s = analysis.structure
            lines += [
                "## Market Structure",
                "",
                s.summary,
                "",
                f"- Wyckoff: {s.wyckoff_phase} — {s.wyckoff_notes}",
                f"- Elliott: {s.elliott_notes}",
                "",
            ]

        if analysis.derivatives_notes:
            lines += ["## Derivatives", ""]
            for n in analysis.derivatives_notes:
                lines.append(f"- {n}")
            lines.append("")

        if analysis.multi_tf_notes:
            lines += ["## Multi-Timeframe", ""]
            for n in analysis.multi_tf_notes:
                lines.append(f"- {n}")
            lines.append("")

        if analysis.news:
            lines += ["## News", "", analysis.news.summary or "", ""]
            for i in analysis.news.top(8):
                lines.append(f"- ({i.sentiment_score:+.2f}) [{i.source}] {i.title}")
            lines.append("")

        if analysis.scenarios:
            lines += ["## Scenarios", ""]
            for row in (
                analysis.scenarios.bullish,
                analysis.scenarios.base,
                analysis.scenarios.bearish,
            ):
                lines.append(
                    f"### {row['name']} ({row['probability']}%)\n"
                    f"- Trigger: {row['trigger']}\n"
                    f"- Target: {row['target']}\n"
                    f"- Invalidation: {row['invalidation']}\n"
                    f"- {row['narrative']}\n"
                )

        lines += [
            "## Trader Commentary",
            "",
            analysis.trader_commentary or "",
            "",
            "## Disclaimer",
            "",
            DISCLAIMER,
            "",
        ]
        if extra and extra.get("vision_notes"):
            lines.insert(-4, "## Screen / Vision\n\n" + str(extra["vision_notes"]) + "\n")
        return "\n".join(lines)

    def save(
        self,
        analysis: FullAnalysis,
        extra: Optional[Dict[str, Any]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Path]:
        """Save Markdown and/or JSON report. Returns paths written."""
        out = ensure_dir(output_dir or self.config.resolve_path(self.config.output.output_dir))
        stamp = utc_now_iso().replace(":", "").replace("+00:00", "Z")
        base = f"{slugify(analysis.symbol)}_{analysis.primary_tf}_{stamp}"
        paths: Dict[str, Path] = {}

        if self.config.output.save_json:
            jp = out / f"{base}.json"
            with open(jp, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(analysis, extra=extra), f, indent=2, default=str)
            paths["json"] = jp
            logger.info("Saved JSON report {}", jp)

        if self.config.output.save_markdown:
            mp = out / f"{base}.md"
            with open(mp, "w", encoding="utf-8") as f:
                f.write(self.to_markdown(analysis, extra=extra))
            paths["markdown"] = mp
            logger.info("Saved Markdown report {}", mp)

        return paths

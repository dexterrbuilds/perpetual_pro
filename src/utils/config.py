"""Configuration loading and typed access for perpetual_pro."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv
from loguru import logger


# Project root: perpetual_pro/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# Canonical liquid crypto-perpetual universe used by scheduled scans, Telegram
# commands, and historical replay when no explicit symbol override is supplied.
DEFAULT_CRYPTO_WATCHLIST = [
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "XRP",
    "DOGE",
    "TRX",
    "LINK",
    "AVAX",
    "ADA",
    "SUI",
    "APT",
    "ARB",
    "OP",
    "INJ",
    "SEI",
    "HYPE",
    "WIF",
    "BONK",
    "PENGU",
    "FET",
    "TAO",
    "RENDER",
    "NEAR",
    "TON",
    "LTC",
    "BCH",
    "ICP",
    "ALGO",
    "FIL",
    "AAVE",
    "UNI",
    "CRV",
    "ENA",
    "JUP",
    "PYTH",
    "TIA",
    "KAS",
    "VIRTUAL",
    "HBAR",
    "XLM",
    "ZEC",
    "DOT",
    "ATOM",
    "ETC",
]


@dataclass
class ExchangeConfig:
    default: str = "okx"
    auto_fallback: bool = True
    fallback_exchanges: List[str] = field(
        default_factory=lambda: [
            "okx",
            "bybit",
            "binanceusdm",
            "bitget",
            "mexc",
            "bingx",
            "bitmart",
            "gate",
            "htx",
            "weex",
        ]
    )
    api_key: str = ""
    api_secret: str = ""
    password: str = ""
    enable_rate_limit: bool = True
    timeout_ms: int = 30000
    sandbox: bool = False


@dataclass
class RiskConfig:
    """Simulated capital sizing — not a live exchange balance."""

    simulated_capital: float = 1000.0
    risk_per_trade_pct: float = 1.0
    # Prop account mode: clamp risk 0.5–1% and max leverage 5x
    prop_mode: bool = True
    risk_per_trade_min_pct: float = 0.5
    risk_per_trade_max_pct: float = 1.0
    daily_drawdown_warn_pct: float = 3.0
    max_open_risk_pct: float = 2.0
    # Leverage bounds (prop default 1–5; day-trade mode uses 10–30)
    leverage_ceiling: float = 5.0
    leverage_floor: float = 1.0
    min_rr: float = 1.25
    default_stop_atr_mult: float = 1.0
    default_tp_atr_mults: List[float] = field(default_factory=lambda: [0.7, 1.3, 2.0, 3.0])
    # Legacy alias (read-only migration)
    account_balance: float = 1000.0
    max_leverage: int = 5


@dataclass
class TimeframesConfig:
    """Day-trade stack: 15m execution, 1h drive, 4h confirmation."""

    primary: str = "15m"
    higher: List[str] = field(default_factory=lambda: ["1h", "4h"])
    ohlcv_limit: int = 700
    cache_ttl_seconds: int = 300
    fetch_workers: int = 4


@dataclass
class LLMConfig:
    enabled: bool = True
    groq_api_key: str = ""
    gemini_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    gemini_model: str = "gemini-2.0-flash"
    timeout_s: int = 25


@dataclass
class AnalysisWeights:
    """Intraday perp weights. Directional inputs dominate contextual inputs."""

    trend: float = 0.17
    momentum: float = 0.18
    structure: float = 0.17
    multi_tf: float = 0.16
    volume: float = 0.11
    derivatives: float = 0.10
    volatility: float = 0.07
    patterns: float = 0.03
    news: float = 0.01


@dataclass
class AnalysisConfig:
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    ema_fast: int = 9
    ema_mid: int = 21
    ema_slow: int = 50
    ema_trend: int = 200
    weights: AnalysisWeights = field(default_factory=AnalysisWeights)
    min_confidence: float = 15.0
    max_confidence: float = 92.0
    directional_score_threshold: float = 0.20
    directional_confidence_threshold: float = 68.0
    legacy_v2_enabled: bool = True
    legacy_execution_min_score: float = 65.0
    execution_min_score: float = 72.0
    execution_confidence_buffer: float = 5.0
    max_immediate_sl_risk: float = 32.0
    max_chase_distance_atr: float = 1.0
    max_pre_entry_tp1_progress_pct: float = 70.0
    max_spread_bps: float = 12.0
    max_ticker_age_seconds: float = 45.0
    max_orderbook_age_seconds: float = 30.0
    min_tp2_rr: float = 1.25
    ready_entry_expiry_bars: int = 3
    retest_entry_expiry_bars: int = 6
    max_entry_valid_minutes: int = 180
    execution_policy_version: str = "execution_quality_v2a.1"
    execution_taker_fee_bps_per_side: float = 5.0
    execution_slippage_bps_per_side: float = 1.5
    execution_funding_bps_per_8h: float = 1.0
    execution_impact_notional_usd: float = 10000.0


@dataclass
class NewsConfig:
    enabled: bool = True
    cryptopanic_token: str = ""
    max_articles: int = 12
    lookback_hours: int = 4
    bullish_keywords: List[str] = field(default_factory=list)
    bearish_keywords: List[str] = field(default_factory=list)


@dataclass
class OCRConfig:
    engine: str = "dual"
    tesseract_cmd: str = ""
    languages: List[str] = field(default_factory=lambda: ["en"])
    easyocr_gpu: bool = False
    min_confidence: float = 0.35


@dataclass
class ScreenConfig:
    default_mode: str = "interactive"
    dark_theme: bool = True
    save_capture: bool = True
    annotate: bool = True
    output_dir: str = "./output"


@dataclass
class VisionConfig:
    use_ollama: bool = True
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llava"
    ollama_timeout_s: int = 45


@dataclass
class OutputConfig:
    save_markdown: bool = True
    save_json: bool = True
    output_dir: str = "./output"
    show_disclaimer: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "./logs/perpetual_pro.log"
    rotation: str = "10 MB"
    retention: str = "14 days"


@dataclass
class TelegramConfig:
    """Telegram alert policy.

    Secrets (bot token, chat id) are NEVER loaded from YAML — only from
    environment variables TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.
    """

    enabled: bool = False
    # Runtime-only; populated exclusively from env in _apply_env_overrides
    bot_token: str = ""
    chat_id: str = ""
    min_llm_confidence: float = 65.0  # deprecated compatibility; not a signal gate
    min_rank_score: float = 50.0
    parse_mode: str = "HTML"
    notify_on_empty: bool = True

    def credentials_configured(self) -> bool:
        return bool((self.bot_token or "").strip() and (self.chat_id or "").strip())


@dataclass
class SchedulerConfig:
    enabled: bool = False
    timezone: str = "Africa/Lagos"  # WAT
    # Empty when DST-aware sessions are configured; no hidden legacy fallback.
    times: List[str] = field(default_factory=list)
    sessions: List[Dict[str, str]] = field(
        default_factory=lambda: [
            {
                "name": "London confirmation",
                "time": "08:20",
                "timezone": "Europe/London",
            },
            {
                "name": "New York macro follow-through",
                "time": "08:50",
                "timezone": "America/New_York",
            },
            {
                "name": "New York open confirmation",
                "time": "09:50",
                "timezone": "America/New_York",
            },
            {
                "name": "New York liquidity window",
                "time": "15:20",
                "timezone": "America/New_York",
            },
        ]
    )
    watchlist: List[str] = field(
        default_factory=lambda: list(DEFAULT_CRYPTO_WATCHLIST)
    )
    exchange: str = "okx"
    timeframe: str = "15m"
    no_news: bool = False
    only_prop_safe: bool = True


@dataclass
class SignalTrackerConfig:
    """Low-overhead forward tracker for Telegram signal outcomes."""

    enabled: bool = True
    websocket_enabled: bool = True
    websocket_url: str = "wss://ws.okx.com:8443/ws/v5/public"
    database_path: str = "./data/signal_tracker.db"
    reconcile_interval_seconds: int = 1800
    lifecycle_check_seconds: int = 15
    notification_retry_seconds: int = 60
    durable_lifecycle_required: bool = False
    target_allocations: List[float] = field(
        default_factory=lambda: [0.25, 0.25, 0.25, 0.25]
    )


@dataclass
class OutcomeScoringConfig:
    """Outcome-calibrated scorer and durable candidate journal.

    ``database_url`` is populated only from DATABASE_URL.  Shadow mode records
    and evaluates candidates without changing production alerts.
    """

    enabled: bool = True
    mode: str = "shadow"  # off | shadow | production
    database_url: str = ""
    feature_schema_version: str = "3.0"
    model_refresh_seconds: int = 300
    minimum_training_samples: int = 500
    minimum_calibration_samples: int = 200
    confidence_floor: float = 80.0
    conservative_quantile: float = 0.10
    promotion_max_ece: float = 0.05
    promotion_minimum_unseen_samples: int = 200


@dataclass
class AppConfig:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    timeframes: TimeframesConfig = field(default_factory=TimeframesConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    signal_tracker: SignalTrackerConfig = field(default_factory=SignalTrackerConfig)
    outcome_scoring: OutcomeScoringConfig = field(default_factory=OutcomeScoringConfig)
    config_path: Optional[Path] = None

    def resolve_path(self, path: str) -> Path:
        """Resolve relative paths against project root or CWD."""
        p = Path(path)
        if p.is_absolute():
            return p
        # Prefer CWD for user-facing output; fall back to project root
        return Path.cwd() / p


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def _dict_to_config(data: Dict[str, Any], config_path: Optional[Path] = None) -> AppConfig:
    ex = data.get("exchange", {}) or {}
    risk = data.get("risk", {}) or {}
    tf = data.get("timeframes", {}) or {}
    an = data.get("analysis", {}) or {}
    weights = an.get("weights", {}) or {}
    news = data.get("news", {}) or {}
    ocr = data.get("ocr", {}) or {}
    screen = data.get("screen", {}) or {}
    vision = data.get("vision", {}) or {}
    llm = data.get("llm", {}) or {}
    output = data.get("output", {}) or {}
    logging_cfg = data.get("logging", {}) or {}
    tg = data.get("telegram", {}) or {}
    sched = data.get("scheduler", {}) or {}
    tracker = data.get("signal_tracker", {}) or {}
    outcome_scoring = data.get("outcome_scoring", {}) or {}

    sim_cap = risk.get("simulated_capital", risk.get("account_balance", 1000.0))
    prop_mode = bool(risk.get("prop_mode", True))
    default_lev_ceil = 5.0 if prop_mode else 30.0
    default_lev_floor = 1.0 if prop_mode else 10.0
    default_max_lev = 5 if prop_mode else 30
    return AppConfig(
        exchange=ExchangeConfig(
            default=str(ex.get("default", "okx")),
            auto_fallback=bool(ex.get("auto_fallback", True)),
            fallback_exchanges=list(
                ex.get(
                    "fallback_exchanges",
                    [
                        "bybit",
                        "binanceusdm",
                        "okx",
                        "bitget",
                        "mexc",
                        "bingx",
                        "bitmart",
                        "gate",
                        "htx",
                        "weex",
                    ],
                )
            ),
            api_key=str(ex.get("api_key", "") or ""),
            api_secret=str(ex.get("api_secret", "") or ""),
            password=str(ex.get("password", "") or ""),
            enable_rate_limit=bool(ex.get("enable_rate_limit", True)),
            timeout_ms=int(ex.get("timeout_ms", 30000)),
            sandbox=bool(ex.get("sandbox", False)),
        ),
        risk=RiskConfig(
            simulated_capital=float(sim_cap),
            account_balance=float(sim_cap),
            risk_per_trade_pct=float(risk.get("risk_per_trade_pct", 1.0)),
            prop_mode=prop_mode,
            risk_per_trade_min_pct=float(risk.get("risk_per_trade_min_pct", 0.5)),
            risk_per_trade_max_pct=float(risk.get("risk_per_trade_max_pct", 1.0)),
            daily_drawdown_warn_pct=float(risk.get("daily_drawdown_warn_pct", 3.0)),
            max_open_risk_pct=float(risk.get("max_open_risk_pct", 2.0)),
            leverage_ceiling=float(risk.get("leverage_ceiling", risk.get("max_leverage", default_lev_ceil))),
            leverage_floor=float(risk.get("leverage_floor", default_lev_floor)),
            max_leverage=int(risk.get("max_leverage", risk.get("leverage_ceiling", default_max_lev))),
            min_rr=float(risk.get("min_rr", 1.25)),
            default_stop_atr_mult=float(risk.get("default_stop_atr_mult", 1.0)),
            default_tp_atr_mults=list(risk.get("default_tp_atr_mults", [0.7, 1.3, 2.0, 3.0])),
        ),
        timeframes=TimeframesConfig(
            primary=str(tf.get("primary", "15m")),
            higher=list(tf.get("higher", ["1h", "4h"])),
            ohlcv_limit=int(tf.get("ohlcv_limit", 700)),
            cache_ttl_seconds=max(0, int(tf.get("cache_ttl_seconds", 300))),
            fetch_workers=max(1, min(8, int(tf.get("fetch_workers", 4)))),
        ),
        analysis=AnalysisConfig(
            rsi_period=int(an.get("rsi_period", 14)),
            macd_fast=int(an.get("macd_fast", 12)),
            macd_slow=int(an.get("macd_slow", 26)),
            macd_signal=int(an.get("macd_signal", 9)),
            atr_period=int(an.get("atr_period", 14)),
            ema_fast=int(an.get("ema_fast", 9)),
            ema_mid=int(an.get("ema_mid", 21)),
            ema_slow=int(an.get("ema_slow", 50)),
            ema_trend=int(an.get("ema_trend", 200)),
            weights=AnalysisWeights(
                trend=float(weights.get("trend", 0.17)),
                momentum=float(weights.get("momentum", 0.18)),
                structure=float(weights.get("structure", 0.17)),
                multi_tf=float(weights.get("multi_tf", 0.16)),
                volume=float(weights.get("volume", 0.11)),
                derivatives=float(weights.get("derivatives", 0.10)),
                volatility=float(weights.get("volatility", 0.07)),
                patterns=float(weights.get("patterns", 0.03)),
                news=float(weights.get("news", 0.01)),
            ),
            min_confidence=float(an.get("min_confidence", 15)),
            max_confidence=float(an.get("max_confidence", 92)),
            directional_score_threshold=float(an.get("directional_score_threshold", 0.20)),
            directional_confidence_threshold=float(
                an.get("directional_confidence_threshold", 68)
            ),
            legacy_v2_enabled=bool(an.get("legacy_v2_enabled", True)),
            legacy_execution_min_score=float(
                an.get("legacy_execution_min_score", 65)
            ),
            execution_min_score=float(an.get("execution_min_score", 72)),
            execution_confidence_buffer=max(
                0.0,
                float(an.get("execution_confidence_buffer", 5.0)),
            ),
            max_immediate_sl_risk=float(an.get("max_immediate_sl_risk", 32)),
            max_chase_distance_atr=float(
                an.get("max_chase_distance_atr", 1.0)
            ),
            max_pre_entry_tp1_progress_pct=float(
                an.get("max_pre_entry_tp1_progress_pct", 70.0)
            ),
            max_spread_bps=float(an.get("max_spread_bps", 12.0)),
            max_ticker_age_seconds=max(
                1.0, float(an.get("max_ticker_age_seconds", 45.0))
            ),
            max_orderbook_age_seconds=max(
                1.0, float(an.get("max_orderbook_age_seconds", 30.0))
            ),
            min_tp2_rr=float(an.get("min_tp2_rr", 1.25)),
            ready_entry_expiry_bars=max(
                1, int(an.get("ready_entry_expiry_bars", 3))
            ),
            retest_entry_expiry_bars=max(
                1, int(an.get("retest_entry_expiry_bars", 6))
            ),
            max_entry_valid_minutes=max(
                30, int(an.get("max_entry_valid_minutes", 180))
            ),
            execution_policy_version=str(
                an.get("execution_policy_version", "execution_quality_v2a.1")
            ),
            execution_taker_fee_bps_per_side=max(
                0.0, float(an.get("execution_taker_fee_bps_per_side", 5.0))
            ),
            execution_slippage_bps_per_side=max(
                0.0, float(an.get("execution_slippage_bps_per_side", 1.5))
            ),
            execution_funding_bps_per_8h=max(
                0.0, float(an.get("execution_funding_bps_per_8h", 1.0))
            ),
            execution_impact_notional_usd=max(
                100.0, float(an.get("execution_impact_notional_usd", 10000.0))
            ),
        ),
        news=NewsConfig(
            enabled=bool(news.get("enabled", True)),
            cryptopanic_token=str(news.get("cryptopanic_token", "") or ""),
            max_articles=int(news.get("max_articles", 12)),
            lookback_hours=int(news.get("lookback_hours", 4)),
            bullish_keywords=list(news.get("bullish_keywords", [])),
            bearish_keywords=list(news.get("bearish_keywords", [])),
        ),
        ocr=OCRConfig(
            engine=str(ocr.get("engine", "dual")),
            tesseract_cmd=str(ocr.get("tesseract_cmd", "") or ""),
            languages=list(ocr.get("languages", ["en"])),
            easyocr_gpu=bool(ocr.get("easyocr_gpu", False)),
            min_confidence=float(ocr.get("min_confidence", 0.35)),
        ),
        screen=ScreenConfig(
            default_mode=str(screen.get("default_mode", "interactive")),
            dark_theme=bool(screen.get("dark_theme", True)),
            save_capture=bool(screen.get("save_capture", True)),
            annotate=bool(screen.get("annotate", True)),
            output_dir=str(screen.get("output_dir", "./output")),
        ),
        vision=VisionConfig(
            use_ollama=bool(vision.get("use_ollama", True)),
            ollama_base_url=str(vision.get("ollama_base_url", "http://127.0.0.1:11434")),
            ollama_model=str(vision.get("ollama_model", "llava")),
            ollama_timeout_s=int(vision.get("ollama_timeout_s", 45)),
        ),
        llm=LLMConfig(
            enabled=bool(llm.get("enabled", True)),
            groq_api_key=str(llm.get("groq_api_key", "") or ""),
            gemini_api_key=str(llm.get("gemini_api_key", "") or ""),
            groq_model=str(llm.get("groq_model", "llama-3.1-8b-instant")),
            gemini_model=str(llm.get("gemini_model", "gemini-2.0-flash")),
            timeout_s=int(llm.get("timeout_s", 25)),
        ),
        output=OutputConfig(
            save_markdown=bool(output.get("save_markdown", True)),
            save_json=bool(output.get("save_json", True)),
            output_dir=str(output.get("output_dir", "./output")),
            show_disclaimer=bool(output.get("show_disclaimer", True)),
        ),
        logging=LoggingConfig(
            level=str(logging_cfg.get("level", "INFO")),
            file=str(logging_cfg.get("file", "./logs/perpetual_pro.log")),
            rotation=str(logging_cfg.get("rotation", "10 MB")),
            retention=str(logging_cfg.get("retention", "14 days")),
        ),
        telegram=TelegramConfig(
            # Never read bot_token / chat_id from YAML (secrets → env only)
            enabled=False,
            bot_token="",
            chat_id="",
            min_llm_confidence=float(tg.get("min_llm_confidence", 65)),
            min_rank_score=float(tg.get("min_rank_score", 50)),
            parse_mode=str(tg.get("parse_mode", "HTML") or "HTML"),
            notify_on_empty=bool(tg.get("notify_on_empty", True)),
        ),
        scheduler=SchedulerConfig(
            enabled=bool(sched.get("enabled", True)),
            timezone=str(sched.get("timezone", "Africa/Lagos") or "Africa/Lagos"),
            times=list(sched.get("times") or []),
            sessions=[
                {
                    "name": str(item.get("name") or "Trading session"),
                    "time": str(item.get("time") or "00:00"),
                    "timezone": str(item.get("timezone") or "UTC"),
                }
                for item in (
                    sched.get("sessions")
                    or [
                        {
                            "name": "London confirmation",
                            "time": "08:20",
                            "timezone": "Europe/London",
                        },
                        {
                            "name": "New York macro follow-through",
                            "time": "08:50",
                            "timezone": "America/New_York",
                        },
                        {
                            "name": "New York open confirmation",
                            "time": "09:50",
                            "timezone": "America/New_York",
                        },
                        {
                            "name": "New York liquidity window",
                            "time": "15:20",
                            "timezone": "America/New_York",
                        },
                    ]
                )
                if isinstance(item, dict)
            ],
            watchlist=[
                str(symbol).strip().upper()
                for symbol in (
                    sched.get("watchlist") or DEFAULT_CRYPTO_WATCHLIST
                )
                if str(symbol).strip()
            ],
            exchange=str(sched.get("exchange", "okx") or "okx"),
            timeframe=str(sched.get("timeframe", "15m") or "15m"),
            no_news=bool(sched.get("no_news", False)),
            only_prop_safe=bool(sched.get("only_prop_safe", True)),
        ),
        signal_tracker=SignalTrackerConfig(
            enabled=bool(tracker.get("enabled", True)),
            websocket_enabled=bool(tracker.get("websocket_enabled", True)),
            websocket_url=str(
                tracker.get(
                    "websocket_url",
                    "wss://ws.okx.com:8443/ws/v5/public",
                )
                or "wss://ws.okx.com:8443/ws/v5/public"
            ),
            database_path=str(
                tracker.get("database_path", "./data/signal_tracker.db")
                or "./data/signal_tracker.db"
            ),
            reconcile_interval_seconds=max(
                60,
                int(tracker.get("reconcile_interval_seconds", 1800)),
            ),
            lifecycle_check_seconds=max(
                5,
                int(tracker.get("lifecycle_check_seconds", 15)),
            ),
            notification_retry_seconds=max(
                15,
                int(tracker.get("notification_retry_seconds", 60)),
            ),
            durable_lifecycle_required=bool(
                tracker.get("durable_lifecycle_required", False)
            ),
            target_allocations=[
                max(0.0, float(value))
                for value in list(
                    tracker.get(
                        "target_allocations",
                        [0.25, 0.25, 0.25, 0.25],
                    )
                )[:4]
            ],
        ),
        outcome_scoring=OutcomeScoringConfig(
            enabled=bool(outcome_scoring.get("enabled", True)),
            mode=str(outcome_scoring.get("mode", "shadow") or "shadow").lower(),
            database_url="",
            feature_schema_version=str(
                outcome_scoring.get("feature_schema_version", "3.0") or "3.0"
            ),
            model_refresh_seconds=max(
                30,
                int(outcome_scoring.get("model_refresh_seconds", 300)),
            ),
            minimum_training_samples=max(
                100,
                int(outcome_scoring.get("minimum_training_samples", 500)),
            ),
            minimum_calibration_samples=max(
                50,
                int(outcome_scoring.get("minimum_calibration_samples", 200)),
            ),
            confidence_floor=float(
                outcome_scoring.get("confidence_floor", 80.0)
            ),
            conservative_quantile=float(
                outcome_scoring.get("conservative_quantile", 0.10)
            ),
            promotion_max_ece=min(
                0.20,
                max(
                    0.01,
                    float(outcome_scoring.get("promotion_max_ece", 0.05)),
                ),
            ),
            promotion_minimum_unseen_samples=max(
                100,
                int(
                    outcome_scoring.get(
                        "promotion_minimum_unseen_samples",
                        200,
                    )
                ),
            ),
        ),
        config_path=config_path,
    )


def _apply_env_overrides(cfg: AppConfig) -> AppConfig:
    """Apply environment variable overrides (secrets + common knobs)."""
    if os.getenv("EXCHANGE_API_KEY"):
        cfg.exchange.api_key = os.environ["EXCHANGE_API_KEY"]
    if os.getenv("EXCHANGE_API_SECRET"):
        cfg.exchange.api_secret = os.environ["EXCHANGE_API_SECRET"]
    if os.getenv("EXCHANGE_PASSWORD"):
        cfg.exchange.password = os.environ["EXCHANGE_PASSWORD"]
    if os.getenv("PERP_EXCHANGE"):
        cfg.exchange.default = os.environ["PERP_EXCHANGE"].strip().lower()
    if os.getenv("CRYPTOPANIC_TOKEN"):
        cfg.news.cryptopanic_token = os.environ["CRYPTOPANIC_TOKEN"]
    if os.getenv("SIMULATED_CAPITAL"):
        cfg.risk.simulated_capital = float(os.environ["SIMULATED_CAPITAL"])
        cfg.risk.account_balance = cfg.risk.simulated_capital
    elif os.getenv("ACCOUNT_BALANCE"):
        # legacy env
        cfg.risk.simulated_capital = float(os.environ["ACCOUNT_BALANCE"])
        cfg.risk.account_balance = cfg.risk.simulated_capital
    if os.getenv("RISK_PER_TRADE_PCT"):
        cfg.risk.risk_per_trade_pct = float(os.environ["RISK_PER_TRADE_PCT"])
    if os.getenv("PROP_MODE"):
        cfg.risk.prop_mode = os.environ["PROP_MODE"].strip().lower() in ("1", "true", "yes", "on")
    if os.getenv("LEGACY_V2_ENABLED"):
        cfg.analysis.legacy_v2_enabled = os.environ[
            "LEGACY_V2_ENABLED"
        ].strip().lower() in ("1", "true", "yes", "on")
    if os.getenv("LEGACY_V2_EXECUTION_MIN_SCORE"):
        cfg.analysis.execution_min_score = float(
            os.environ["LEGACY_V2_EXECUTION_MIN_SCORE"]
        )
    if os.getenv("LEGACY_V2_CONFIDENCE_BUFFER"):
        cfg.analysis.execution_confidence_buffer = max(
            0.0,
            float(os.environ["LEGACY_V2_CONFIDENCE_BUFFER"]),
        )
    # Telegram secrets: environment only — never from config.yaml
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    cfg.telegram.bot_token = token
    cfg.telegram.chat_id = chat_id
    # Auto-enable when both credentials are present (unless explicitly disabled)
    force_tg = (os.getenv("TELEGRAM_ENABLED") or "").strip().lower()
    if force_tg in ("0", "false", "no", "off"):
        cfg.telegram.enabled = False
    elif force_tg in ("1", "true", "yes", "on"):
        cfg.telegram.enabled = bool(token and chat_id)
    else:
        cfg.telegram.enabled = bool(token and chat_id)
    if os.getenv("SCHEDULER_ENABLED"):
        cfg.scheduler.enabled = os.environ["SCHEDULER_ENABLED"].strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    if os.getenv("SIGNAL_TRACKER_ENABLED"):
        cfg.signal_tracker.enabled = os.environ[
            "SIGNAL_TRACKER_ENABLED"
        ].strip().lower() in ("1", "true", "yes", "on")
    if os.getenv("SIGNAL_TRACKER_WEBSOCKET_ENABLED"):
        cfg.signal_tracker.websocket_enabled = os.environ[
            "SIGNAL_TRACKER_WEBSOCKET_ENABLED"
        ].strip().lower() in ("1", "true", "yes", "on")
    if os.getenv("SIGNAL_TRACKER_DB_PATH"):
        cfg.signal_tracker.database_path = os.environ[
            "SIGNAL_TRACKER_DB_PATH"
        ].strip()
    elif os.getenv("RAILWAY_VOLUME_MOUNT_PATH"):
        mount_path = Path(os.environ["RAILWAY_VOLUME_MOUNT_PATH"].strip())
        cfg.signal_tracker.database_path = str(
            mount_path / "perpetual_pro_signals.db"
        )
    if os.getenv("SIGNAL_TRACKER_RECONCILE_SECONDS"):
        cfg.signal_tracker.reconcile_interval_seconds = max(
            60,
            int(os.environ["SIGNAL_TRACKER_RECONCILE_SECONDS"]),
        )
    database_url = (os.getenv("DATABASE_URL") or "").strip()
    cfg.outcome_scoring.database_url = database_url
    if os.getenv("OUTCOME_SCORING_ENABLED"):
        cfg.outcome_scoring.enabled = os.environ[
            "OUTCOME_SCORING_ENABLED"
        ].strip().lower() in ("1", "true", "yes", "on")
    if os.getenv("OUTCOME_SCORING_MODE"):
        mode = os.environ["OUTCOME_SCORING_MODE"].strip().lower()
        cfg.outcome_scoring.mode = (
            mode if mode in ("off", "shadow", "production") else "shadow"
        )
    if os.getenv("OKX_PUBLIC_WEBSOCKET_URL"):
        cfg.signal_tracker.websocket_url = os.environ[
            "OKX_PUBLIC_WEBSOCKET_URL"
        ].strip()
    if os.getenv("TESSERACT_CMD"):
        cfg.ocr.tesseract_cmd = os.environ["TESSERACT_CMD"]
    if os.getenv("OLLAMA_BASE_URL"):
        cfg.vision.ollama_base_url = os.environ["OLLAMA_BASE_URL"]
    if os.getenv("OLLAMA_MODEL"):
        cfg.vision.ollama_model = os.environ["OLLAMA_MODEL"]
    if os.getenv("GROQ_API_KEY"):
        cfg.llm.groq_api_key = os.environ["GROQ_API_KEY"]
    if os.getenv("GEMINI_API_KEY"):
        cfg.llm.gemini_api_key = os.environ["GEMINI_API_KEY"]
    elif os.getenv("GOOGLE_API_KEY"):
        cfg.llm.gemini_api_key = os.environ["GOOGLE_API_KEY"]
    if os.getenv("GROQ_MODEL"):
        cfg.llm.groq_model = os.environ["GROQ_MODEL"]
    if os.getenv("GEMINI_MODEL"):
        cfg.llm.gemini_model = os.environ["GEMINI_MODEL"]
    return cfg


def load_config(path: Optional[Union[str, Path]] = None) -> AppConfig:
    """Load YAML config, merge defaults, apply .env overrides."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    load_dotenv(Path.cwd() / ".env", override=False)

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        # Try CWD then project root
        candidates = [Path.cwd() / config_path, PROJECT_ROOT / config_path, config_path]
        for c in candidates:
            if c.exists():
                config_path = c
                break

    data: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config must be a mapping: {config_path}")
            data = loaded
        logger.debug("Loaded config from {}", config_path)
    else:
        logger.warning("Config not found at {}; using built-in defaults", config_path)

    cfg = _dict_to_config(data, config_path if config_path.exists() else None)
    return _apply_env_overrides(cfg)


def setup_logging(cfg: AppConfig) -> None:
    """Configure loguru sinks from config."""
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level=cfg.logging.level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>\n",
        colorize=True,
    )
    log_path = cfg.resolve_path(cfg.logging.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        level=cfg.logging.level,
        rotation=cfg.logging.rotation,
        retention=cfg.logging.retention,
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )

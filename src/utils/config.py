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


@dataclass
class ExchangeConfig:
    default: str = "bybit"
    auto_fallback: bool = True
    fallback_exchanges: List[str] = field(
        default_factory=lambda: [
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
    # Leverage bounds (prop default 1–5; aggressive mode uses 20–100)
    leverage_ceiling: float = 5.0
    leverage_floor: float = 1.0
    min_rr: float = 1.2
    default_stop_atr_mult: float = 1.0
    default_tp_atr_mults: List[float] = field(default_factory=lambda: [0.7, 1.3, 2.0, 3.0])
    # Legacy alias (read-only migration)
    account_balance: float = 1000.0
    max_leverage: int = 5


@dataclass
class TimeframesConfig:
    """Day-trade stack: 5m / 15m / 1h with 4h confirmation."""

    primary: str = "15m"
    higher: List[str] = field(default_factory=lambda: ["5m", "1h", "4h"])
    ohlcv_limit: int = 500


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
    """Short-term perp weights — momentum, funding, volume, micro-structure first."""

    momentum: float = 0.20
    derivatives: float = 0.16
    structure: float = 0.14
    volume: float = 0.12
    multi_tf: float = 0.12
    patterns: float = 0.10
    trend: float = 0.10
    news: float = 0.06


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
    min_llm_confidence: float = 65.0
    min_rank_score: float = 50.0
    parse_mode: str = "HTML"
    notify_on_empty: bool = False

    def credentials_configured(self) -> bool:
        return bool((self.bot_token or "").strip() and (self.chat_id or "").strip())


@dataclass
class SchedulerConfig:
    enabled: bool = False
    timezone: str = "Africa/Lagos"  # WAT
    times: List[str] = field(default_factory=lambda: ["05:00", "16:00", "20:00"])
    watchlist: List[str] = field(default_factory=list)
    exchange: str = "bybit"
    timeframe: str = "15m"
    no_news: bool = False
    only_prop_safe: bool = True


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

    sim_cap = risk.get("simulated_capital", risk.get("account_balance", 1000.0))
    prop_mode = bool(risk.get("prop_mode", True))
    default_lev_ceil = 5.0 if prop_mode else 100.0
    default_lev_floor = 1.0 if prop_mode else 20.0
    default_max_lev = 5 if prop_mode else 100
    return AppConfig(
        exchange=ExchangeConfig(
            default=str(ex.get("default", "bybit")),
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
            min_rr=float(risk.get("min_rr", 1.2)),
            default_stop_atr_mult=float(risk.get("default_stop_atr_mult", 1.0)),
            default_tp_atr_mults=list(risk.get("default_tp_atr_mults", [0.7, 1.3, 2.0, 3.0])),
        ),
        timeframes=TimeframesConfig(
            primary=str(tf.get("primary", "15m")),
            higher=list(tf.get("higher", ["5m", "1h", "4h"])),
            ohlcv_limit=int(tf.get("ohlcv_limit", 500)),
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
                momentum=float(weights.get("momentum", 0.20)),
                derivatives=float(weights.get("derivatives", 0.16)),
                structure=float(weights.get("structure", 0.14)),
                volume=float(weights.get("volume", 0.12)),
                multi_tf=float(weights.get("multi_tf", 0.12)),
                patterns=float(weights.get("patterns", 0.10)),
                trend=float(weights.get("trend", 0.10)),
                news=float(weights.get("news", 0.06)),
            ),
            min_confidence=float(an.get("min_confidence", 15)),
            max_confidence=float(an.get("max_confidence", 92)),
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
            notify_on_empty=bool(tg.get("notify_on_empty", False)),
        ),
        scheduler=SchedulerConfig(
            enabled=bool(sched.get("enabled", True)),
            timezone=str(sched.get("timezone", "Africa/Lagos") or "Africa/Lagos"),
            times=list(sched.get("times", ["05:00", "16:00", "20:00"])),
            watchlist=list(sched.get("watchlist", [])),
            exchange=str(sched.get("exchange", "bybit") or "bybit"),
            timeframe=str(sched.get("timeframe", "15m") or "15m"),
            no_news=bool(sched.get("no_news", False)),
            only_prop_safe=bool(sched.get("only_prop_safe", True)),
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

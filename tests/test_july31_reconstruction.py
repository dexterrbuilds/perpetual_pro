from src.scheduler.scan_job import DEFAULT_WATCHLIST


def test_july31_watchlist_is_reconstructed_without_duplicates():
    assert DEFAULT_WATCHLIST == [
        "BTC", "ETH", "SOL", "BNB", "TRX", "UNI", "XRP", "DOGE", "LTC",
        "LINK", "BCH", "HBAR", "XLM", "HYPE", "ZEC", "XMR", "ICP", "ALGO",
        "AVAX", "PENGU", "WIF", "BONK",
    ]
    assert len(DEFAULT_WATCHLIST) == len(set(DEFAULT_WATCHLIST)) == 22


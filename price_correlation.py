"""
price_correlation.py
---------------------
Correlates Korean earnings transcript sentiment scores with subsequent
stock price movement on the Korea Stock Exchange (KOSPI/KOSDAQ).

Data source: yfinance (Yahoo Finance) — covers major KOSPI stocks via .KS suffix

Methodology:
    1. For each transcript, note the disclosure date (rcept_dt)
    2. Fetch stock price T-5 (5 days before) through T+20 (20 days after)
    3. Calculate:
       - Pre-announcement return: T-5 → T-1 (information leakage signal)
       - Post-announcement return: T+1 → T+5 (immediate reaction)
       - Extended return: T+1 → T+20 (sustained drift)
    4. Correlate sentiment_score with each return window

Usage:
    python price_correlation.py --input data/processed/sentiment_results.csv
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings("ignore")


# ── KOSPI ticker mapping ──────────────────────────────────────────────────────
# Yahoo Finance uses .KS suffix for KOSPI, .KQ for KOSDAQ
TICKER_MAP = {
    "Samsung Electronics": "005930.KS",
    "SK Hynix":            "000660.KS",
    "LG Electronics":      "066570.KS",
    "Kakao":               "035720.KS",
    "NAVER":               "035420.KS",
    "Hyundai Motor":       "005380.KS",
    "Kia":                 "000270.KS",
    "Samsung SDI":         "006400.KS",
    "SK Innovation":       "096770.KS",
    "Celltrion":           "068270.KS",
}


def fetch_price_window(
    ticker: str,
    event_date: str,
    pre_days: int = 5,
    post_days: int = 20,
) -> pd.DataFrame | None:
    """
    Fetch stock prices around an event date.

    Args:
        ticker:     Yahoo Finance ticker (e.g. '005930.KS')
        event_date: Disclosure date as 'YYYYMMDD'
        pre_days:   Trading days before event to fetch
        post_days:  Trading days after event to fetch

    Returns:
        DataFrame with Date, Close, Return columns or None if fetch fails
    """
    try:
        dt = datetime.strptime(event_date, "%Y%m%d")
        start = dt - timedelta(days=pre_days * 2)   # Buffer for weekends/holidays
        end   = dt + timedelta(days=post_days * 2)

        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return None

        df = df[["Close"]].copy()
        df.index = pd.to_datetime(df.index)
        df["Return"] = df["Close"].pct_change()

        return df

    except Exception as e:
        print(f"  ⚠ Price fetch failed for {ticker}: {e}")
        return None


def calculate_event_returns(
    price_df: pd.DataFrame,
    event_date: str,
) -> dict:
    """
    Calculate pre/post event returns relative to event date.

    Returns dict with:
        pre_5d_return:    cumulative return T-5 to T-1
        post_1d_return:   return on T+1
        post_5d_return:   cumulative return T+1 to T+5
        post_20d_return:  cumulative return T+1 to T+20
        event_day_return: return on T (day of announcement)
    """
    try:
        dt = datetime.strptime(event_date, "%Y%m%d")
        idx = price_df.index

        # Find nearest trading days
        pre_start  = idx[idx <= dt - timedelta(days=5)]
        pre_end    = idx[idx < dt]
        post_start = idx[idx > dt]

        if len(pre_start) == 0 or len(pre_end) == 0 or len(post_start) == 0:
            return {}

        t_minus_5 = pre_start[-1]
        t_minus_1 = pre_end[-1]
        t_plus_1  = post_start[0]

        post_5_dates  = post_start[:5]
        post_20_dates = post_start[:20]

        price_at = lambda d: float(price_df.loc[d, "Close"])

        pre_5d   = (price_at(t_minus_1) / price_at(t_minus_5) - 1) if t_minus_5 != t_minus_1 else 0
        post_1d  = (price_at(t_plus_1) / price_at(t_minus_1) - 1)
        post_5d  = (price_at(post_5_dates[-1]) / price_at(t_minus_1) - 1) if len(post_5_dates) >= 5 else None
        post_20d = (price_at(post_20_dates[-1]) / price_at(t_minus_1) - 1) if len(post_20_dates) >= 20 else None

        return {
            "pre_5d_return":   round(pre_5d * 100, 3),
            "post_1d_return":  round(post_1d * 100, 3),
            "post_5d_return":  round(post_5d * 100, 3) if post_5d is not None else None,
            "post_20d_return": round(post_20d * 100, 3) if post_20d is not None else None,
        }

    except Exception as e:
        print(f"  ⚠ Return calculation error: {e}")
        return {}


def run_correlation_analysis(
    sentiment_df: pd.DataFrame,
    output_dir: str = "data/processed",
) -> pd.DataFrame:
    """
    Full pipeline: fetch prices → calculate returns → merge with sentiment → correlate.

    Args:
        sentiment_df: Output from sentiment_pipeline.analyse_dataframe()

    Returns:
        DataFrame with sentiment scores and price returns merged
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n📈 Fetching stock prices and calculating event returns...")
    all_rows = []

    for _, row in sentiment_df.iterrows():
        company  = row["company"]
        ticker   = TICKER_MAP.get(company)
        evt_date = str(row["rcept_dt"])

        if not ticker:
            print(f"  ⚠ No ticker for {company}, skipping")
            continue

        price_df = fetch_price_window(ticker, evt_date)
        if price_df is None:
            continue

        returns = calculate_event_returns(price_df, evt_date)
        if not returns:
            continue

        merged_row = row.to_dict()
        merged_row.update(returns)
        merged_row["ticker"] = ticker
        all_rows.append(merged_row)

    if not all_rows:
        print("⚠ No data to correlate. Check ticker map and dates.")
        return pd.DataFrame()

    result_df = pd.DataFrame(all_rows)

    # ── Correlation analysis ──────────────────────────────────────────────
    print("\n── Sentiment-Price Correlation ──")
    return_cols = ["post_1d_return", "post_5d_return", "post_20d_return", "pre_5d_return"]

    for col in return_cols:
        valid = result_df[["sentiment_score", col]].dropna()
        if len(valid) >= 3:
            corr = valid["sentiment_score"].corr(valid[col])
            print(f"  Sentiment vs {col:20s}: r = {corr:.3f}  (n={len(valid)})")

    # Directional accuracy: positive sentiment → positive post-return?
    if "post_5d_return" in result_df.columns:
        valid = result_df[["sentiment_label", "post_5d_return"]].dropna()
        if len(valid) >= 3:
            correct = (
                ((valid["sentiment_label"] == "positive") & (valid["post_5d_return"] > 0)) |
                ((valid["sentiment_label"] == "negative") & (valid["post_5d_return"] < 0))
            ).sum()
            accuracy = correct / len(valid) * 100
            print(f"\n  Directional accuracy (5-day): {accuracy:.1f}%  (n={len(valid)})")

    out_path = os.path.join(output_dir, "sentiment_price_correlation.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Results saved to {out_path}")
    return result_df


# ── Demo mode with synthetic price data ───────────────────────────────────────
def demo_correlation(sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates price correlation using realistic synthetic returns.
    Used when yfinance is unavailable or for rapid development.
    """
    np.random.seed(42)
    demo_rows = []

    LABEL_BIAS = {"positive": 0.015, "negative": -0.012, "neutral": 0.001}

    for _, row in sentiment_df.iterrows():
        bias = LABEL_BIAS.get(row.get("sentiment_label", "neutral"), 0)
        noise = np.random.normal(0, 0.02)

        merged = row.to_dict()
        merged.update({
            "ticker":         TICKER_MAP.get(row["company"], "N/A"),
            "pre_5d_return":  round(np.random.normal(0, 0.015) * 100, 3),
            "post_1d_return": round((bias * 0.6 + noise * 0.4) * 100, 3),
            "post_5d_return": round((bias + noise) * 100, 3),
            "post_20d_return":round((bias * 1.5 + np.random.normal(0, 0.025)) * 100, 3),
        })
        demo_rows.append(merged)

    return pd.DataFrame(demo_rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/sentiment_results.csv")
    parser.add_argument("--demo",  action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.input):
        df = pd.read_csv(args.input)
    else:
        print("⚠ Input file not found. Run sentiment_pipeline.py first.")
        print("  Generating demo sentiment data...")
        from dart_scraper import generate_demo_transcripts
        from sentiment_pipeline import KoreanFinancialSentiment
        raw_df = generate_demo_transcripts()
        kfs    = KoreanFinancialSentiment()
        df     = kfs.analyse_dataframe(raw_df)

    if args.demo:
        result = demo_correlation(df)
    else:
        result = run_correlation_analysis(df)

    print(f"\nFinal dataset shape: {result.shape}")
    print(result[["company", "rcept_dt", "sentiment_label", "sentiment_score",
                  "post_5d_return"]].to_string())

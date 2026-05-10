"""
run_pipeline.py
---------------
End-to-end runner for the Korean Financial Sentiment Analyser.

Modes:
  --demo      Run with synthetic data (no API keys needed) — start here
  --full      Run with real DART data (requires DART API key)
  --dashboard Launch the Streamlit dashboard after processing

Steps:
  1. Fetch/generate earnings transcripts (dart_scraper.py)
  2. Run Korean sentiment analysis (sentiment_pipeline.py)
  3. Correlate with KOSPI price returns (price_correlation.py)
  4. Save results → data/processed/

Usage:
    # Quick demo run (no API keys):
    python run_pipeline.py --demo

    # Full run with real data:
    python run_pipeline.py --full --dart_key YOUR_DART_KEY

    # Launch dashboard after run:
    python run_pipeline.py --demo --dashboard
"""

import os
import sys
import argparse
import pandas as pd


def run_demo_pipeline():
    """Full pipeline with synthetic data — no API keys required."""
    print("=" * 60)
    print("  Korean Financial Sentiment Analyser — DEMO MODE")
    print("=" * 60)

    from dart_scraper import generate_demo_transcripts
    from sentiment_pipeline import KoreanFinancialSentiment
    from price_correlation import demo_correlation

    # Step 1: Load demo transcripts
    print("\n📡 Step 1: Loading demo DART transcripts...")
    raw_df = generate_demo_transcripts()
    print(f"  {len(raw_df)} transcripts from {raw_df['company'].nunique()} companies")

    # Step 2: Sentiment analysis
    print("\n🧠 Step 2: Running Korean sentiment analysis (keyword mode)...")
    kfs = KoreanFinancialSentiment()
    sent_df = kfs.analyse_dataframe(raw_df)

    # Step 3: Price correlation
    print("\n📈 Step 3: Simulating KOSPI price returns...")
    final_df = demo_correlation(sent_df)

    # Step 4: Save
    os.makedirs("data/processed", exist_ok=True)
    out = "data/processed/sentiment_price_correlation.csv"
    final_df.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n✅ Pipeline complete! Results saved to {out}")
    print(f"\n── Summary ──")
    print(f"  Total transcripts: {len(final_df)}")
    print(f"  Companies:         {final_df['company'].nunique()}")
    label_counts = final_df["sentiment_label"].value_counts()
    for label, count in label_counts.items():
        print(f"  {label.capitalize():12s}: {count}")

    if "post_5d_return" in final_df.columns:
        corr = final_df["sentiment_score"].corr(final_df["post_5d_return"])
        print(f"\n  Sentiment-Price Correlation (5D): r = {corr:.3f}")

    return final_df


def run_full_pipeline(dart_key: str):
    """Full pipeline with real DART data and live KOSPI prices."""
    print("=" * 60)
    print("  Korean Financial Sentiment Analyser — FULL MODE")
    print("=" * 60)

    from dart_scraper import fetch_earnings_transcripts
    from sentiment_pipeline import KoreanFinancialSentiment
    from price_correlation import run_correlation_analysis

    # Step 1
    print("\n📡 Step 1: Fetching DART transcripts...")
    raw_df = fetch_earnings_transcripts(api_key=dart_key, year=2024)

    if raw_df.empty:
        print("⚠ No transcripts fetched. Check your DART API key.")
        return

    # Step 2
    print("\n🧠 Step 2: Running KoBERT sentiment analysis...")
    kfs = KoreanFinancialSentiment()
    sent_df = kfs.analyse_dataframe(raw_df)
    sent_df.to_csv("data/processed/sentiment_results.csv", index=False, encoding="utf-8-sig")

    # Step 3
    print("\n📈 Step 3: Fetching KOSPI prices and correlating...")
    final_df = run_correlation_analysis(sent_df)

    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Korean Financial Sentiment Pipeline")
    parser.add_argument("--demo",      action="store_true", help="Run with demo data")
    parser.add_argument("--full",      action="store_true", help="Run with real DART data")
    parser.add_argument("--dart_key",  type=str, default="", help="DART Open API key")
    parser.add_argument("--dashboard", action="store_true", help="Launch dashboard after run")
    args = parser.parse_args()

    if args.full and args.dart_key:
        df = run_full_pipeline(args.dart_key)
    else:
        if args.full and not args.dart_key:
            print("⚠ --full requires --dart_key. Falling back to demo mode.")
        df = run_demo_pipeline()

    if args.dashboard:
        print("\n🚀 Launching Streamlit dashboard...")
        os.system("streamlit run dashboard.py")

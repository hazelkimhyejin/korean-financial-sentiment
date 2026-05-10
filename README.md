# 📊 Korean Financial Sentiment Analyser

> **Unique angle:** Korean DART earnings transcripts + KoBERT NLP + KOSPI price correlation  
> Built by a bilingual (English–Korean) ML practitioner — combining language fluency with financial data science.

## What This Does

Most sentiment-price research focuses on English-language sources. Korean market earnings calls are published on **DART** (Korea's SEC equivalent) — in Korean — and are almost entirely ignored by Western quant researchers.

This project:
1. **Scrapes** Korean earnings transcripts from the DART Open API
2. **Analyses** sentiment using KoBERT (Korean Financial BERT)
3. **Correlates** sentiment scores with subsequent KOSPI stock price movement
4. **Visualises** results in an interactive Streamlit dashboard

## Key Research Question
> *Does the tone of Korean earnings calls predict short-term stock price direction?*

## Live Dashboard
🔗 *(Streamlit deploy link — coming soon)*

## Preliminary Findings (Demo Data)
| Window | Sentiment-Return Correlation |
|---|---|
| Next day (T+1) | r ≈ +0.42 |
| 5-day window | r ≈ +0.58 |
| 20-day window | r ≈ +0.51 |

> Note: Demo data uses synthetic returns. Real correlations require live DART + yfinance data.

## Tech Stack
- **DART Open API** — Korea FSS official corporate disclosure system
- **KoBERT / KR-FinBERT-SC** — Korean Financial BERT (snunlp/KR-FinBert-SC)
- **yfinance** — KOSPI price data via Yahoo Finance (.KS tickers)
- **Plotly + Streamlit** — interactive dashboard
- **Python** · pandas · numpy · transformers · torch

## Project Structure
```
korean-financial-sentiment/
├── run_pipeline.py        # End-to-end runner
├── dart_scraper.py        # DART API transcript fetcher
├── sentiment_pipeline.py  # KoBERT sentiment classifier
├── price_correlation.py   # KOSPI return calculator + correlator
├── dashboard.py           # Streamlit dashboard
├── requirements.txt
├── data/
│   ├── raw/               # DART transcript CSVs
│   └── processed/         # Sentiment + price merged results
└── README.md
```

## Quickstart (Demo — No API Key Needed)
```bash
git clone https://github.com/hazelkimhyejin/korean-financial-sentiment
cd korean-financial-sentiment
pip install -r requirements.txt

# Run full pipeline with demo data
python run_pipeline.py --demo

# Launch dashboard
streamlit run dashboard.py
```

## Full Run (Real DART Data)
1. Get a free DART API key: https://opendart.fss.or.kr
2. Run:
```bash
python run_pipeline.py --full --dart_key YOUR_KEY --dashboard
```

## Target Companies (KOSPI)
Samsung Electronics · SK Hynix · LG Electronics · Kakao · NAVER  
Hyundai Motor · Kia · Samsung SDI · SK Innovation · Celltrion

## Roadmap
- [x] DART scraper with demo mode
- [x] KoBERT / keyword sentiment pipeline
- [x] KOSPI price correlation module
- [x] Streamlit dashboard (sentiment + scatter + live analyser)
- [ ] Fine-tune KR-FinBERT-SC on DART-specific vocabulary
- [ ] Expand to 50+ companies, 3-year history
- [ ] Add translation layer (Korean → English summaries)
- [ ] Deploy to Streamlit Cloud

## Why This Project?
Korean is one of the least-represented languages in financial NLP research. As a bilingual English–Korean professional with a background in financial data analysis and ML engineering, I'm building tooling that bridges this gap — with direct applicability to APAC investment research, Korean MNC investor relations, and cross-border financial analytics.

---

Built by **[Hazel I.](https://linkedin.com/in/hazel-ip-jl)**  
Applied AI & ML Portfolio · Singapore / Seoul · English–Korean Bilingual (TOPIK Level 4)

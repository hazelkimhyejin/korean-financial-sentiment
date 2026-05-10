"""
dart_scraper.py
---------------
Fetches Korean earnings call transcripts from DART (Data Analysis, Retrieval
and Transfer System) — Korea's official corporate disclosure platform.

DART Open API: https://opendart.fss.or.kr
Get a free API key at: https://opendart.fss.or.kr/uss/umt/EgovMberInsertView.do

Usage:
    python dart_scraper.py --api_key YOUR_KEY --corp_code 00126380 --year 2024
"""

import requests
import pandas as pd
import json
import time
import os
import argparse
from datetime import datetime

# ── Top 10 KOSPI companies and their DART corp codes ──────────────────────────
# Full list: https://opendart.fss.or.kr/api/corpCode.xml
TARGET_COMPANIES = {
    "Samsung Electronics": "00126380",
    "SK Hynix":            "00164779",
    "LG Electronics":      "00401731",
    "Kakao":               "00258801",
    "NAVER":               "00266961",
    "Hyundai Motor":       "00164742",
    "Kia":                 "00164788",
    "Samsung SDI":         "00126671",
    "SK Innovation":       "00631518",
    "Celltrion":           "00421045",
}

DART_BASE = "https://opendart.fss.or.kr/api"


def get_disclosure_list(api_key: str, corp_code: str, year: int) -> list[dict]:
    """Fetch list of disclosures for a company in a given year."""
    url = f"{DART_BASE}/list.json"
    params = {
        "crtfc_key": api_key,
        "corp_code":  corp_code,
        "bgn_de":     f"{year}0101",
        "end_de":     f"{year}1231",
        "pblntf_ty":  "A",      # A = regular disclosure (earnings calls live here)
        "page_count": 100,
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "000":
        print(f"  DART API warning: {data.get('message')}")
        return []

    return data.get("list", [])


def get_document_text(api_key: str, rcept_no: str) -> str:
    """Download and extract text from a DART filing."""
    url = f"{DART_BASE}/document.xml"
    params = {"crtfc_key": api_key, "rcept_no": rcept_no}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    # DART returns a zip-like XML bundle; extract raw text
    # For now return raw content — production version would unzip + parse HTML
    return resp.text[:5000]  # First 5000 chars as preview


def fetch_earnings_transcripts(
    api_key: str,
    companies: dict[str, str] | None = None,
    year: int = 2024,
    output_dir: str = "data/raw",
) -> pd.DataFrame:
    """
    Main fetch function. Downloads earnings-related disclosures for each company.

    Returns a DataFrame with columns:
        company, corp_code, rcept_no, report_nm, rcept_dt, text_preview
    """
    os.makedirs(output_dir, exist_ok=True)

    if companies is None:
        companies = TARGET_COMPANIES

    records = []

    for company_name, corp_code in companies.items():
        print(f"\n📡 Fetching disclosures for {company_name} ({corp_code})...")
        try:
            disclosures = get_disclosure_list(api_key, corp_code, year)

            # Filter for earnings-related keywords in Korean
            keywords = ["실적", "분기", "연간", "earnings", "결산"]
            earnings = [
                d for d in disclosures
                if any(k in d.get("report_nm", "") for k in keywords)
            ]

            print(f"  Found {len(disclosures)} total, {len(earnings)} earnings-related")

            for doc in earnings[:3]:  # Limit to 3 per company for rate limiting
                rcept_no   = doc["rcept_no"]
                report_nm  = doc["report_nm"]
                rcept_dt   = doc["rcept_dt"]

                print(f"  → {report_nm} ({rcept_dt})")
                text = get_document_text(api_key, rcept_no)

                records.append({
                    "company":      company_name,
                    "corp_code":    corp_code,
                    "rcept_no":     rcept_no,
                    "report_nm":    report_nm,
                    "rcept_dt":     rcept_dt,
                    "year":         year,
                    "text_preview": text,
                })

                time.sleep(0.5)  # Respect DART rate limit

        except Exception as e:
            print(f"  ⚠ Error fetching {company_name}: {e}")
            continue

    df = pd.DataFrame(records)
    out_path = os.path.join(output_dir, f"dart_transcripts_{year}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Saved {len(df)} records to {out_path}")
    return df


# ── Demo mode: generate synthetic transcripts for development ─────────────────
def generate_demo_transcripts(output_dir: str = "data/raw") -> pd.DataFrame:
    """
    Generates realistic synthetic Korean earnings transcript snippets
    for development/testing without a DART API key.

    In production, replace with fetch_earnings_transcripts().
    """
    os.makedirs(output_dir, exist_ok=True)

    DEMO_TRANSCRIPTS = [
        {
            "company": "Samsung Electronics",
            "corp_code": "00126380",
            "rcept_no": "20240115000001",
            "report_nm": "2023년 4분기 실적발표",
            "rcept_dt": "20240115",
            "year": 2023,
            "text_preview": (
                "2023년 4분기 실적을 발표합니다. 매출액은 전년 동기 대비 8% 증가하였으며, "
                "영업이익은 반도체 부문의 강한 회복세에 힘입어 크게 개선되었습니다. "
                "HBM 수요가 AI 서버 확대로 급증하고 있으며, 2024년에는 더욱 강한 성장이 "
                "기대됩니다. 파운드리 사업도 안정적인 수주를 유지하고 있습니다."
            ),
            "sentiment_label": "positive",
        },
        {
            "company": "Samsung Electronics",
            "corp_code": "00126380",
            "rcept_no": "20230715000002",
            "report_nm": "2023년 2분기 실적발표",
            "rcept_dt": "20230715",
            "year": 2023,
            "text_preview": (
                "2분기 실적은 업황 악화로 인해 어려운 환경이 지속되었습니다. "
                "메모리 반도체 가격 하락이 지속되고 있으며, 재고 조정 압력이 남아 있습니다. "
                "수요 회복 시점은 불확실하며, 하반기에도 보수적인 전망을 유지합니다. "
                "원가 절감과 효율화를 통해 수익성 방어에 집중할 계획입니다."
            ),
            "sentiment_label": "negative",
        },
        {
            "company": "SK Hynix",
            "corp_code": "00164779",
            "rcept_no": "20240125000003",
            "report_nm": "2023년 4분기 실적발표",
            "rcept_dt": "20240125",
            "year": 2023,
            "text_preview": (
                "HBM3E 양산을 성공적으로 시작하였으며, 엔비디아 등 주요 고객사로의 공급이 "
                "확대되고 있습니다. AI 메모리 수요는 예상을 상회하고 있으며, "
                "2024년 상반기 HBM 공급이 완판된 상태입니다. 실적 전망은 매우 긍정적입니다."
            ),
            "sentiment_label": "positive",
        },
        {
            "company": "NAVER",
            "corp_code": "00266961",
            "rcept_no": "20240208000004",
            "report_nm": "2023년 연간 실적발표",
            "rcept_dt": "20240208",
            "year": 2023,
            "text_preview": (
                "검색 광고 매출은 안정적이나 성장세가 둔화되고 있습니다. "
                "커머스와 핀테크 부문은 양호한 성장세를 유지하고 있습니다. "
                "클라우드 사업 확장과 AI 투자를 지속할 예정이며, "
                "일본 라인야후 관련 불확실성은 지속적으로 모니터링하고 있습니다."
            ),
            "sentiment_label": "neutral",
        },
        {
            "company": "Kakao",
            "corp_code": "00258801",
            "rcept_no": "20240214000005",
            "report_nm": "2023년 4분기 실적발표",
            "rcept_dt": "20240214",
            "year": 2023,
            "text_preview": (
                "규제 불확실성과 경기 둔화로 광고 매출이 부진하였습니다. "
                "플랫폼 사업 전반의 수익성 개선이 시급한 상황입니다. "
                "카카오페이와 카카오뱅크는 상대적으로 견조한 실적을 기록하였습니다. "
                "비용 효율화와 수익 구조 개선에 집중할 계획입니다."
            ),
            "sentiment_label": "negative",
        },
        {
            "company": "LG Electronics",
            "corp_code": "00401731",
            "rcept_no": "20240130000006",
            "report_nm": "2023년 4분기 실적발표",
            "rcept_dt": "20240130",
            "year": 2023,
            "text_preview": (
                "가전 사업은 글로벌 수요 부진에도 불구하고 프리미엄 제품 중심으로 "
                "안정적인 수익성을 유지하였습니다. 전장 부문은 전기차 확대에 힘입어 "
                "강한 성장세를 이어가고 있습니다. 2024년에는 B2B 사업 확대가 핵심 성장 동력이 될 것입니다."
            ),
            "sentiment_label": "positive",
        },
    ]

    df = pd.DataFrame(DEMO_TRANSCRIPTS)
    out_path = os.path.join(output_dir, "dart_transcripts_demo.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"✅ Generated {len(df)} demo transcripts → {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default="", help="DART Open API key")
    parser.add_argument("--year",    type=int, default=2024)
    parser.add_argument("--demo",    action="store_true", help="Use demo data (no API key needed)")
    args = parser.parse_args()

    if args.demo or not args.api_key:
        print("🔧 Running in DEMO mode (no DART API key required)")
        df = generate_demo_transcripts()
    else:
        df = fetch_earnings_transcripts(api_key=args.api_key, year=args.year)

    print(f"\nSample output:\n{df[['company','report_nm','rcept_dt']].to_string()}")

"""
sentiment_pipeline.py
----------------------
Korean financial sentiment analysis using pre-trained transformer models.

Model options (in order of preference):
1. snunlp/KR-FinBert-SC  — Korean Financial BERT, sentiment classification
2. monologg/koelectra-base-finetuned-sentiment  — KoELECTRA sentiment
3. cardiffnlp/twitter-xlm-roberta-base-sentiment — multilingual fallback

The pipeline:
  raw Korean text → chunking → model inference → sentiment score → label

Usage:
    from sentiment_pipeline import KoreanFinancialSentiment
    kfs = KoreanFinancialSentiment()
    result = kfs.analyse("매출액이 전년 대비 15% 증가하였습니다.")
    # → {'label': 'positive', 'score': 0.91, 'model': 'KR-FinBert-SC'}
"""

import pandas as pd
import numpy as np
import torch
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Literal


# ── Sentiment model config ────────────────────────────────────────────────────

# Primary: Korean Financial BERT (trained on Korean financial news)
# Falls back to multilingual model if not available
MODEL_PRIORITY = [
    "snunlp/KR-FinBert-SC",                          # Best for Korean finance
    "monologg/koelectra-base-finetuned-sentiment",    # General Korean sentiment
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",  # Multilingual fallback
]

# Korean financial sentiment keywords for rule-based augmentation
POSITIVE_KEYWORDS = [
    "증가", "성장", "개선", "확대", "상승", "호조", "긍정", "기대",
    "역대", "최고", "강세", "회복", "흑자", "수익", "양호", "견조",
    "완판", "급증", "돌파", "초과"
]
NEGATIVE_KEYWORDS = [
    "감소", "하락", "악화", "축소", "부진", "불확실", "우려", "리스크",
    "적자", "손실", "둔화", "압력", "약세", "위축", "지연", "저조",
    "어려움", "부담", "하향", "침체"
]


class KoreanFinancialSentiment:
    """
    Korean financial earnings transcript sentiment classifier.

    Combines transformer model predictions with keyword-based augmentation
    for robust financial text analysis.
    """

    def __init__(self, model_name: str | None = None, device: str = "auto"):
        self.model_name = None
        self.pipe = None
        self.device = 0 if (device == "auto" and torch.cuda.is_available()) else -1

        model_to_load = model_name or MODEL_PRIORITY[0]
        self._load_model(model_to_load)

    def _load_model(self, model_name: str):
        """Load transformer model with fallback chain."""
        models_to_try = [model_name] + [m for m in MODEL_PRIORITY if m != model_name]

        for model in models_to_try:
            try:
                print(f"⏳ Loading model: {model}")
                self.pipe = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=model,
                    device=self.device,
                    truncation=True,
                    max_length=512,
                )
                self.model_name = model
                print(f"✅ Model loaded: {model}")
                return
            except Exception as e:
                print(f"  ⚠ Could not load {model}: {e}")
                continue

        print("⚠ No transformer model loaded. Falling back to keyword-only mode.")

    def _keyword_sentiment(self, text: str) -> dict:
        """Rule-based keyword sentiment as fallback / augmentation."""
        text_lower = text.lower()
        pos_count = sum(1 for w in POSITIVE_KEYWORDS if w in text)
        neg_count = sum(1 for w in NEGATIVE_KEYWORDS if w in text)

        total = pos_count + neg_count
        if total == 0:
            return {"label": "neutral", "score": 0.50, "method": "keyword"}

        pos_ratio = pos_count / total
        if pos_ratio >= 0.6:
            return {"label": "positive", "score": round(0.5 + pos_ratio * 0.45, 3), "method": "keyword"}
        elif pos_ratio <= 0.4:
            return {"label": "negative", "score": round(1 - pos_ratio * 0.45, 3), "method": "keyword"}
        else:
            return {"label": "neutral", "score": 0.55, "method": "keyword"}

    def _normalise_label(self, raw_label: str) -> str:
        """Normalise model output labels to positive/negative/neutral."""
        label = raw_label.upper()
        if label in ["POSITIVE", "POS", "LABEL_2", "2"]:
            return "positive"
        elif label in ["NEGATIVE", "NEG", "LABEL_0", "0"]:
            return "negative"
        else:
            return "neutral"

    def analyse(self, text: str) -> dict:
        """
        Analyse sentiment of a single Korean text passage.

        Returns:
            {
                'label':  'positive' | 'negative' | 'neutral',
                'score':  float (0–1, confidence),
                'pos_keywords': int,
                'neg_keywords': int,
                'method': 'transformer' | 'keyword',
                'model':  str
            }
        """
        # Count keywords regardless of method
        pos_kw = sum(1 for w in POSITIVE_KEYWORDS if w in text)
        neg_kw = sum(1 for w in NEGATIVE_KEYWORDS if w in text)

        if self.pipe is None:
            result = self._keyword_sentiment(text)
            result.update({"pos_keywords": pos_kw, "neg_keywords": neg_kw, "model": "keyword-only"})
            return result

        try:
            # Chunk long texts (BERT max 512 tokens ≈ ~350 Korean chars)
            chunks = [text[i:i+300] for i in range(0, len(text), 300)] if len(text) > 300 else [text]

            scores = []
            for chunk in chunks:
                if not chunk.strip():
                    continue
                out = self.pipe(chunk)[0]
                label     = self._normalise_label(out["label"])
                raw_score = out["score"]
                # Normalise to positive-direction score
                signed = raw_score if label == "positive" else (1 - raw_score if label == "negative" else 0.5)
                scores.append(signed)

            avg_score = float(np.mean(scores)) if scores else 0.5
            final_label = (
                "positive" if avg_score >= 0.55
                else "negative" if avg_score <= 0.45
                else "neutral"
            )

            return {
                "label":        final_label,
                "score":        round(avg_score, 4),
                "pos_keywords": pos_kw,
                "neg_keywords": neg_kw,
                "method":       "transformer",
                "model":        self.model_name,
            }

        except Exception as e:
            print(f"  ⚠ Transformer inference failed: {e}. Falling back to keywords.")
            result = self._keyword_sentiment(text)
            result.update({"pos_keywords": pos_kw, "neg_keywords": neg_kw, "model": "keyword-fallback"})
            return result

    def analyse_dataframe(self, df: pd.DataFrame, text_col: str = "text_preview") -> pd.DataFrame:
        """
        Run sentiment analysis on an entire DataFrame of transcripts.

        Adds columns: sentiment_label, sentiment_score, pos_keywords, neg_keywords, method
        """
        print(f"\n🔍 Analysing sentiment for {len(df)} transcripts...")
        results = []
        for i, row in df.iterrows():
            text = str(row.get(text_col, ""))
            result = self.analyse(text)
            results.append(result)
            label_emoji = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(result["label"], "⚪")
            print(f"  {label_emoji} {row.get('company','?')} ({row.get('rcept_dt','?')}): "
                  f"{result['label']} ({result['score']:.2f})")

        results_df = pd.DataFrame(results).rename(columns={
            "label": "sentiment_label",
            "score": "sentiment_score",
        })
        return pd.concat([df.reset_index(drop=True), results_df], axis=1)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    kfs = KoreanFinancialSentiment()

    test_cases = [
        ("Samsung positive", "매출액이 전년 대비 15% 증가하였으며 영업이익도 크게 개선되었습니다. HBM 수요가 급증하고 있습니다."),
        ("SK negative",      "업황 악화로 인해 실적이 감소하였으며 재고 조정 압력이 지속되고 있습니다. 불확실성이 높습니다."),
        ("NAVER neutral",    "검색 광고 매출은 안정적이나 성장세가 둔화되고 있습니다. 신규 사업 투자를 지속합니다."),
    ]

    print("\n── Sentiment Analysis Test ──")
    for name, text in test_cases:
        result = kfs.analyse(text)
        print(f"\n{name}:")
        print(f"  Text:  {text[:60]}...")
        print(f"  Label: {result['label']} | Score: {result['score']:.3f} | "
              f"Pos KW: {result['pos_keywords']} | Neg KW: {result['neg_keywords']}")

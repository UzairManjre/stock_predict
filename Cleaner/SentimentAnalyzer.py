# --- DEPENDENCY SETUP ---
import os
import sys
import fitz  # PyMuPDF
import torch
import pandas as pd
from datetime import datetime
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Hardware Detection ---
device = 0 if torch.cuda.is_available() else -1
print(f"Hardware Acceleration: Using {'CUDA' if device == 0 else 'CPU'}")

# --- Load FinBERT Model ---
print("Booting Trading Desk AI: Financial Sentiment Engine (FinBERT)...")
try:
    financial_sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        device=device
    )
    print("Engine online. Ready to generate alpha.")
except Exception as e:
    print("FATAL ERROR: Could not initialize FinBERT:", e)
    sys.exit(1)


# --- PDF Text Extractor ---
def extract_text_from_pdf(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        print(f"Error reading PDF file at {pdf_path}: {e}")
        return None


# --- Retail Sentiment (VADER) ---
def analyze_sentiment_retail(text):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)['compound']


# --- Pro Sentiment (FinBERT with Chunks) ---
def analyze_sentiment_pro(text, chunk_size=512):
    scores = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        try:
            result = financial_sentiment_pipeline(chunk)[0]
            label, score = result['label'], result['score']
            if label == "positive":
                scores.append(score)
            elif label == "negative":
                scores.append(-score)
            else:
                scores.append(0.0)
        except Exception as e:
            print(f"  -> Error analyzing chunk: {e}")
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


# --- Hybrid Decision Engine ---
def hybrid_decision(vader_score, finbert_score, threshold=0.5):
    if finbert_score > threshold and vader_score > 0.05:
        return "STRONG BULLISH"
    elif finbert_score < -threshold and vader_score < -0.05:
        return "STRONG BEARISH"
    elif vader_score > 0.05 and abs(finbert_score) < 0.2:
        return "HYPE / WEAK BULLISH"
    elif vader_score < -0.05 and abs(finbert_score) < 0.2:
        return "HYPE / WEAK BEARISH"
    else:
        return "NEUTRAL / MIXED"


# --- Save to Parquet (Dataset Building) ---
def save_result(company, pdf_path, vader_score, finbert_score, hybrid_score, verdict, out_dir="sentiment_results"):
    os.makedirs(out_dir, exist_ok=True)
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    data = {
        "timestamp": [date],
        "company": [company],
        "file": [os.path.basename(pdf_path)],
        "vader_score": [vader_score],
        "finbert_score": [finbert_score],
        "hybrid_score": [hybrid_score],  # ✅ NEW NUMERIC FEATURE
        "verdict": [verdict],
    }

    df = pd.DataFrame(data)
    parquet_file = os.path.join(out_dir, "sentiment_history.parquet")

    if os.path.exists(parquet_file):
        old_df = pd.read_parquet(parquet_file)
        df = pd.concat([old_df, df], ignore_index=True)

    df.to_parquet(parquet_file, index=False)
    print(f"Result saved to {parquet_file}")


# --- Main Demo ---
if __name__ == '__main__':
    sample_pdf_path = "Cleaner/sample_announcement.pdf"
    company_name = "DemoCompany"  # later you can pass actual NSE company name

    if not os.path.exists(sample_pdf_path):
        print(f"ERROR: Sample PDF not found at '{sample_pdf_path}'")
        sys.exit(1)

    print(f"\nLoading and extracting text from '{sample_pdf_path}'...")
    document_text = extract_text_from_pdf(sample_pdf_path)

    if document_text:
        print(f"Successfully extracted {len(document_text)} characters of text.")

        # Run both engines
        vader_score = analyze_sentiment_retail(document_text)
        finbert_score = analyze_sentiment_pro(document_text)

        # Hybrid numeric score
        hybrid_score = (vader_score + finbert_score) / 2.0  # ✅ clean number

        # Hybrid verdict text
        verdict = hybrid_decision(vader_score, finbert_score)

        print("\n" + "="*60)
        print("INTEL BRIEFING: FULL PDF DOCUMENT ANALYSIS")
        print(f"  -> Retail Tool (VADER): {vader_score:.4f}")
        print(f"  -> Pro-Grade AI (FinBERT): {finbert_score:.4f}")
        print(f"  -> Hybrid Score (numeric): {hybrid_score:.4f}")
        print(f"  -> HYBRID VERDICT: {verdict}")
        print("="*60)

        # Save to parquet for training dataset
        save_result(company_name, sample_pdf_path, vader_score, finbert_score, hybrid_score, verdict)

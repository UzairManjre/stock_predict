import os
import fitz  # PyMuPDF
import torch
import pandas as pd
from datetime import datetime
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

# ================== CONFIG ==================
DATA_ROOT = r'data\market_data_20250920_225539\individual_data'
OUTPUT_CSV = 'all_announcements_with_sentiment.csv'
CHECKPOINT_FILE = 'sentiment_checkpoint1.csv'
FINBERT_MODEL = 'ProsusAI/finbert'
CHUNK_SIZE = 512
BATCH_SIZE = 32

# ================== DEVICE ==================
device = 0 if torch.cuda.is_available() else -1
print(f"Device set to use {'cuda:0' if device == 0 else 'CPU'}")

# ================== MODELS ==================
print("✅ Loading models...")
financial_sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=FINBERT_MODEL,
    device=device,
    batch_size=BATCH_SIZE,
    torch_dtype=torch.float16 if device == 0 else None
)
vader_analyzer = SentimentIntensityAnalyzer()
print("✅ Models loaded")

# ================== HELPERS ==================
def extract_text(pdf_path):
    try:
        with fitz.open(pdf_path) as doc:
            return ''.join(page.get_text() for page in doc)
    except Exception as e:
        print(f"⚠️ MuPDF error in {pdf_path}: {e}")
        return ''

def analyze_vader(text):
    return vader_analyzer.polarity_scores(text)['compound']

def hybrid_decision(vader_score, finbert_score, threshold=0.5):
    if vader_score is None or finbert_score is None:
        return None
    if finbert_score > threshold and vader_score > 0.05:
        return 'STRONG BULLISH'
    elif finbert_score < -threshold and vader_score < -0.05:
        return 'STRONG BEARISH'
    elif vader_score > 0.05 and abs(finbert_score) < 0.2:
        return 'HYPE / WEAK BULLISH'
    elif vader_score < -0.05 and abs(finbert_score) < 0.2:
        return 'HYPE / WEAK BEARISH'
    else:
        return 'NEUTRAL / MIXED'

# ================== COLLECT ANNOUNCEMENTS ==================
def collect_announcements():
    all_rows = []
    for folder in os.listdir(DATA_ROOT):
        folder_path = os.path.join(DATA_ROOT, folder)
        if not os.path.isdir(folder_path):
            continue
        ann_csv = os.path.join(folder_path, 'corporate_announcements.csv')
        pdf_dir = os.path.join(folder_path, 'PDFs')
        if not os.path.exists(ann_csv) or not os.path.exists(pdf_dir):
            continue
        df = pd.read_csv(ann_csv)
        for _, row in df.iterrows():
            pdf_link = str(row.get('PDF_Link', '')).strip()
            filename = pdf_link.split('/')[-1] if '/' in pdf_link else pdf_link
            full_pdf_path = os.path.join(pdf_dir, filename)
            if not os.path.exists(full_pdf_path):
                received_time = row.get('Received_Time', '')
                subject = str(row.get('Subject', '')).replace('/', '').replace(':','').replace(' ', '_')
                date_str = ''
                try:
                    date_obj = pd.to_datetime(received_time, errors='coerce', dayfirst=True)
                    if pd.notnull(date_obj):
                        date_str = date_obj.strftime('%Y-%m-%d')
                except:
                    pass
                for f in os.listdir(pdf_dir):
                    if date_str and date_str in f and subject[:20] in f:
                        full_pdf_path = os.path.join(pdf_dir, f)
                        break
            row_dict = row.to_dict()
            row_dict['pdf_file'] = full_pdf_path if os.path.exists(full_pdf_path) else None
            row_dict['stock_folder'] = folder
            all_rows.append(row_dict)
    return all_rows

# ================== MAIN ==================
def main():
    print('Collecting all announcements...')
    all_rows = collect_announcements()
    print(f'Total announcements found: {len(all_rows)}')

    # checkpoint
    done_files = set()
    results = []
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint_df = pd.read_csv(CHECKPOINT_FILE)
        if 'pdf_file' in checkpoint_df.columns:
            done_files = set(checkpoint_df['pdf_file'].dropna())
            results = checkpoint_df.to_dict(orient='records')
            print(f'Resuming from checkpoint, {len(done_files)} already processed...')

    # process announcements
    for idx, row in enumerate(tqdm(all_rows, desc="Processing PDFs")):
        pdf_path = row.get('pdf_file')
        if not pdf_path:
            print(f"⚠️ Skipping missing PDF: {row.get('PDF_Link')}")
            continue
        if pdf_path in done_files:
            continue

        text = extract_text(pdf_path)
        if not text.strip():
            print(f"⚠️ Skipping empty text: {pdf_path}")
            continue

        # split into chunks
        chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

        # batch process with FinBERT
        finbert_scores_list = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i+BATCH_SIZE]
            try:
                res = financial_sentiment_pipeline(batch)
                for r in res:
                    if r['label'] == 'positive':
                        finbert_scores_list.append(r['score'])
                    elif r['label'] == 'negative':
                        finbert_scores_list.append(-r['score'])
                    else:
                        finbert_scores_list.append(0.0)
            except RuntimeError as e:
                print(f"⚠️ GPU OOM for {pdf_path}, retrying on CPU...")
                cpu_pipe = pipeline("sentiment-analysis", model=FINBERT_MODEL, device=-1)
                res = cpu_pipe(batch)
                for r in res:
                    if r['label'] == 'positive':
                        finbert_scores_list.append(r['score'])
                    elif r['label'] == 'negative':
                        finbert_scores_list.append(-r['score'])
                    else:
                        finbert_scores_list.append(0.0)

        finbert_score = sum(finbert_scores_list)/len(finbert_scores_list) if finbert_scores_list else 0.0
        vader_score = analyze_vader(text)
        hybrid_score = (vader_score + finbert_score)/2
        verdict = hybrid_decision(vader_score, finbert_score)

        results.append({
            'pdf_file': pdf_path,
            'stock_folder': row.get('stock_folder'),
            'pdf_link': row.get('PDF_Link'),
            'received_time': row.get('Received_Time'),
            'vader_score': vader_score,
            'finbert_score': finbert_score,
            'hybrid_score': hybrid_score,
            'verdict': verdict,
            'subject': row.get('Subject')
        })

        # save every 50 results
        if len(results) % 50 == 0:
            pd.DataFrame(results).to_csv(CHECKPOINT_FILE, index=False)
            print(f'✅ Checkpoint saved: {len(results)} rows')

    # final save
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f'🎉 All results saved to {OUTPUT_CSV}')

if __name__ == '__main__':
    main()

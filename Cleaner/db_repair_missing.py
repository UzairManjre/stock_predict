import os
import sqlite3
import pandas as pd
import sys
from SentimentAnalyzer import extract_text_from_pdf, analyze_sentiment_retail, analyze_sentiment_pro, hybrid_decision

DB_PATH = os.path.join(os.path.dirname(__file__), 'stock_data.db')
DATA_ROOT = r'd:\IMP\NEED MONEY THINGS\stock_predict\data\market_data_20250920_225539\individual_data'

def repair_missing_sentiment():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''SELECT id, stock_id, pdf_link FROM announcements WHERE vader_score IS NULL OR finbert_score IS NULL OR hybrid_score IS NULL OR verdict IS NULL''')
    missing = cursor.fetchall()
    print(f"Found {len(missing)} announcements with missing sentiment fields.")
    for ann_id, stock_id, pdf_link in missing:
        # Find stock folder
        cursor.execute('SELECT folder_name FROM stocks WHERE id=?', (stock_id,))
        row = cursor.fetchone()
        if not row:
            continue
        folder = row[0]
        pdf_dir = os.path.join(DATA_ROOT, folder, 'PDFs')
        pdf_file = None
        if isinstance(pdf_link, str) and pdf_link:
            pdf_basename = os.path.basename(pdf_link)
            local_pdf = os.path.join(pdf_dir, pdf_basename)
            if os.path.exists(local_pdf):
                pdf_file = local_pdf
            else:
                # Try fuzzy match by date/subject
                for f in os.listdir(pdf_dir):
                    if pdf_basename.split('-')[0] in f:
                        pdf_file = os.path.join(pdf_dir, f)
                        break
        vader_score = finbert_score = hybrid_score = None
        verdict = None
        if pdf_file:
            text = extract_text_from_pdf(pdf_file)
            if text:
                vader_score = analyze_sentiment_retail(text)
                finbert_score = analyze_sentiment_pro(text)
                hybrid_score = (vader_score + finbert_score) / 2.0
                verdict = hybrid_decision(vader_score, finbert_score)
        cursor.execute('''UPDATE announcements SET vader_score=?, finbert_score=?, hybrid_score=?, verdict=? WHERE id=?''',
            (vader_score, finbert_score, hybrid_score, verdict, ann_id))
        conn.commit()
    conn.close()
    print('Sentiment repair complete.')

if __name__ == '__main__':
    repair_missing_sentiment()

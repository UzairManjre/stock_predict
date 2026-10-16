import os
import pandas as pd
from datetime import datetime, timedelta
import time
from jugaad_data.nse import NSELive

def fetch_nifty_pcr():
    """Fetches the Put-Call Ratio for NIFTY 50."""
    try:
        from jugaad_data.nse import NSELive
        n = NSELive()
        # Get live option chain
        payload = n.index_option_chain("NIFTY")
        
        # Payload contains 'filtered' and 'records'
        # We calculate the overall PCR (Total Put OI / Total Call OI)
        tot_ce_oi = payload['filtered']['CE']['totOI']
        tot_pe_oi = payload['filtered']['PE']['totOI']
        
        if tot_ce_oi == 0:
            return 1.0
            
        pcr = tot_pe_oi / tot_ce_oi
        return pcr
    except Exception as e:
        print(f"Error fetching Nifty PCR: {e}")
        return None

def fetch_fii_dii_flow():
    """Fetches the latest daily FII/DII Cash Market flow data."""
    try:
        n = NSELive()
        # n.fii_dii() returns latest day institutional flow
        fii_dii = n.fii_dii()
        
        # Extract net values (in Crores)
        fii_net = None
        dii_net = None
        
        for record in fii_dii:
            if 'FII' in record['category']:
                fii_net = float(record['net'])
            elif 'DII' in record['category']:
                dii_net = float(record['net'])
                
        return fii_net, dii_net
    except Exception as e:
        print(f"Error fetching FII/DII flow: {e}")
        return None, None

def update_macro_environment(db_conn):
    """
    Updates the macro-environment dataset in the database with daily NSE institutional data.
    """
    print("Fetching institutional market data from NSE...")
    pcr = fetch_nifty_pcr()
    fii_net, dii_net = fetch_fii_dii_flow()
    
    print(f"  Nifty PCR: {pcr}")
    print(f"  FII Net (Cr): {fii_net}")
    print(f"  DII Net (Cr): {dii_net}")
    
    # Store this in the DB for the daily run
    if pcr is not None and fii_net is not None:
        cur = db_conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS macro_daily (
                date DATE PRIMARY KEY,
                nifty_pcr FLOAT,
                fii_flow_cr FLOAT,
                dii_flow_cr FLOAT
            )
        """)
        
        today = datetime.now().date()
        cur.execute("""
            INSERT INTO macro_daily (date, nifty_pcr, fii_flow_cr, dii_flow_cr)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (date) DO UPDATE 
            SET nifty_pcr = EXCLUDED.nifty_pcr,
                fii_flow_cr = EXCLUDED.fii_flow_cr,
                dii_flow_cr = EXCLUDED.dii_flow_cr
        """, (today, pcr, fii_net, dii_net))
        
        db_conn.commit()
        print("Successfully updated macro_daily table.")
        
if __name__ == "__main__":
    pcr = fetch_nifty_pcr()
    print(f"PCR: {pcr}")

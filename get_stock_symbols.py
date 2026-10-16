

import pandas as pd
import ast

def get_stock_symbols(file_path):
    df = pd.read_csv(file_path, nrows=0)
    stocks = set()
    for col_str in df.columns:
        try:
            col_tuple = ast.literal_eval(col_str)
            if isinstance(col_tuple, tuple) and len(col_tuple) > 1 and col_tuple[1]:
                stocks.add(col_tuple[1])
        except (ValueError, SyntaxError):
            pass
    return list(stocks)

if __name__ == "__main__":
    file_path = "D:\\IMP\\NEED MONEY THINGS\\stock_predict\\data\\market_data_20250920_222108\\summary\\complete_market_data.csv"
    symbols = get_stock_symbols(file_path)
    print(symbols)


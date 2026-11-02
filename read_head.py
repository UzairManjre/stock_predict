
import os
import pandas as pd
import ast

df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "market_data_20250920_222108", "summary", "complete_market_data.csv"), nrows=0)
stocks = set()
for col_str in df.columns:
    try:
        col_tuple = ast.literal_eval(col_str)
        if isinstance(col_tuple, tuple) and len(col_tuple) > 1 and col_tuple[1]:
            stocks.add(col_tuple[1])
    except (ValueError, SyntaxError):
        pass
print(list(stocks))

import re

app_path = "dashboard/app.py"

with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace hardcoded strings
replacements = [
    ("'5-Day Predicted Return'", "'Predicted Return'"),
    ("'Median predicted 5 day return'", "'Median predicted return'"),
    ("5D Q50", "Q50"),
    ("Actual 5-Day Return", "Actual Return"),
    ("actual 5-trading-day window", "actual trading window"),
    ("next 5 days?", "target horizon?"),
    ("5-Day Quantile Forecast", "Multi-Horizon Quantile Forecast"),
    ("5-Day", "Predicted"),
]

for old, new in replacements:
    content = content.replace(old, new)

with open(app_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Dashboard text labels patched successfully!")

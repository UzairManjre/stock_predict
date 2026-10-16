import re

app_path = "dashboard/app.py"

with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace hardcoded strings
replacements = [
    ("Median predicted 5-day return", "Median predicted return"),
    ("5-day predicted return", "predicted return"),
    ("Predicted Predicted Return (%)", "Predicted Return (%)"),
    ("actual 5d return", "actual return"),
    ("Actual 5d Return", "Actual Return"),
]

for old, new in replacements:
    content = content.replace(old, new)

with open(app_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Dashboard text labels patched again successfully!")

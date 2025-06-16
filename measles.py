import pandas as pd

# URL of the page that contains the measles data table
URL = "https://www.cdc.gov/measles/data-research/index.html"

# Extract all tables from the page
tables = pd.read_html(URL, header=0)

# Print available tables and their column names
print("Found tables:")
for i, tbl in enumerate(tables):
    print(f"\nTable {i} columns: {tbl.columns.tolist()}")

# === Once you've run the script and identified the correct table index ===
# For example, if table 3 is the one with 'week_start' and 'cases', use:
TABLE_INDEX = 3  # Change this after inspecting printed output

df = tables[TABLE_INDEX]

# Preview the data
print("\nSelected Measles Table:")
print(df.head())

# === Optional: clean column names ===
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Optional: save to local file for future use
df.to_csv("measles_cases_2025.csv", index=False)
print("\nSaved cleaned table as 'measles_cases_2025.csv'")
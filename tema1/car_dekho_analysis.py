import os
import glob
import sys
import math
import textwrap
import pandas as pd
import matplotlib.pyplot as plt


PRICE_THRESHOLD_LAKHS = 10.0


MANUAL_DATASET_DIR = None  

dataset_dir = None

if MANUAL_DATASET_DIR and os.path.isdir(MANUAL_DATASET_DIR):
    dataset_dir = MANUAL_DATASET_DIR
else:
    try:
        import kagglehub  # type: ignore
        print("Downloading dataset with kagglehub…")
        dataset_dir = kagglehub.dataset_download("nehalbirla/vehicle-dataset-from-cardekho")
    except Exception as e:
        print("\n⚠️  Could not use kagglehub to download the dataset.")
        print("   Reason:", repr(e))
        print("   If you already have the dataset, set MANUAL_DATASET_DIR at the top of this script")
        print("   to the folder that contains the CSV, then re‑run this script.\n")
        sys.exit(1)

print("Dataset directory:", dataset_dir)

# ------------------------------
# 2) Load CSV (auto‑detect filename)
# ------------------------------
csv_candidates = glob.glob(os.path.join(dataset_dir, "*.csv"))
if not csv_candidates:
    print("No CSV files found in:", dataset_dir)
    sys.exit(1)

# Prefer "car data.csv" (the canonical filename for this dataset).
csv_path = None
for c in csv_candidates:
    if "car data" in os.path.basename(c).lower():
        csv_path = c
        break
if csv_path is None:
    # Fall back to the first csv we find
    csv_path = csv_candidates[0]

print("Using CSV:", os.path.basename(csv_path))

df_raw = pd.read_csv(csv_path)

# ------------------------------
# 3) Harmonize columns
# ------------------------------
# Expected columns in this dataset:
# Car_Name, Year, Selling_Price, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner
rename_map = {
    "Car_Name": "car_name",
    "Year": "year",
    "Selling_Price": "selling_price",
    "Present_Price": "present_price",
    "Kms_Driven": "kms_driven",
    "Fuel_Type": "fuel",
    "Seller_Type": "seller_type",
    "Transmission": "transmission",
    "Owner": "owner",
}
df = df_raw.rename(columns=rename_map)

# Ensure numeric dtypes where relevant
for col in ["year", "selling_price", "present_price", "kms_driven", "owner"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows without price or mileage
df = df.dropna(subset=["selling_price", "kms_driven"]).reset_index(drop=True)

# ------------------------------
# 4) Min & Max price with characteristics
# ------------------------------
idx_min = df["selling_price"].idxmin()
idx_max = df["selling_price"].idxmax()

min_row = df.loc[idx_min]
max_row = df.loc[idx_max]

def pretty_row(row: pd.Series) -> str:
    # Print a tidy view of a single row
    items = []
    for k, v in row.items():
        items.append(f"{k}: {v}")
    return textwrap.indent("\n".join(items), prefix="  ")

print("\n=== Cheapest car (by Selling_Price, in lakhs) ===")
print(pretty_row(min_row))

print("\n=== Most expensive car (by Selling_Price, in lakhs) ===")
print(pretty_row(max_row))

# ------------------------------
# 5) Scatter: mileage vs price
# ------------------------------
plt.figure()
plt.scatter(df["kms_driven"], df["selling_price"], alpha=0.6)
plt.xlabel("Mileage (Kms_Driven)")
plt.ylabel("Selling Price (lakhs)")
plt.title("Mileage vs Selling Price")
plt.tight_layout()
scatter_path = os.path.join(os.path.dirname(csv_path), "scatter_kms_vs_price.png")
plt.savefig(scatter_path, dpi=150)
plt.show()
print(f"\nSaved scatter plot to: {scatter_path}")

# ------------------------------
# 6) Count prices above threshold
# ------------------------------
count_above = (df["selling_price"] > PRICE_THRESHOLD_LAKHS).sum()
total = len(df)
pct = (count_above / total * 100.0) if total else math.nan
print(f"\nCount of cars with price > {PRICE_THRESHOLD_LAKHS} lakhs: {count_above} / {total} ({pct:.1f}%)")

# ------------------------------
# 7) Mean, Std, Boxplot, Violin plot
# ------------------------------
mean_price = df["selling_price"].mean()
std_price = df["selling_price"].std(ddof=1)

print(f"\nPrice stats (lakhs): mean = {mean_price:.3f}, std = {std_price:.3f}")

# Boxplot (matplotlib only)
plt.figure()
plt.boxplot(df["selling_price"].dropna().values, vert=True, showmeans=True)
plt.title("Selling Price — Boxplot")
plt.ylabel("Lakhs")
plt.tight_layout()
boxplot_path = os.path.join(os.path.dirname(csv_path), "boxplot_price.png")
plt.savefig(boxplot_path, dpi=150)
plt.show()
print(f"Saved boxplot to: {boxplot_path}")

# Violin plot (matplotlib)
plt.figure()
plt.violinplot(dataset=[df["selling_price"].dropna().values], showmeans=True, showextrema=True, showmedians=True)
plt.title("Selling Price — Violin Plot")
plt.ylabel("Lakhs")
plt.tight_layout()
violin_path = os.path.join(os.path.dirname(csv_path), "violin_price.png")
plt.savefig(violin_path, dpi=150)
plt.show()
print(f"Saved violin plot to: {violin_path}")

# ------------------------------
# 8) Optional: Save min/max rows to CSV for reference
# ------------------------------
out_csv = os.path.join(os.path.dirname(csv_path), "min_max_price_rows.csv")
pd.concat([min_row.to_frame().T, max_row.to_frame().T]).to_csv(out_csv, index=False)
print(f"\nSaved min/max rows to: {out_csv}")

print("\n Done.")

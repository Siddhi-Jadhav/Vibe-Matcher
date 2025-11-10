# src/data_prep.py
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PRODUCTS_CSV = DATA_DIR / "products.csv"
OUT_CSV = DATA_DIR / "products_prepped.csv"

def load_and_prep(csv_path=PRODUCTS_CSV):
    df = pd.read_csv(csv_path)
    df['tags'] = df['tags'].fillna("").apply(lambda s: s.split("|") if isinstance(s, str) and s else [])
    df['embed_text'] = df['name'].fillna('') + " " + df['desc'].fillna('') + " " + df['tags'].apply(lambda t: " ".join(t))
    return df

def main():
    df = load_and_prep()
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved prepped products to {OUT_CSV}")
    print(df[['id','name','embed_text']].head().to_string(index=False))

if __name__ == "__main__":
    main()

from src.processing.data_manager import load_bronze, clean_data, save_silver

def run_silver():
    for ds in ["combustibles", "pib"]:
        print(f"Processing {ds}...")
        df = load_bronze(ds)
        df_clean = clean_data(df, ds)
        save_silver(df_clean, ds)

if __name__ == "__main__":
    run_silver()

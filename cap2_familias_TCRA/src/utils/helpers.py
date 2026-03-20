import pandas as pd
from src.config import TABLES_DIR

def save_csv(df, name):
    path = TABLES_DIR / f"{name}.csv"
    df.to_csv(path)
    print(f"Saved CSV table: {path}")

def format_summary_table(results_dict):
    """
    Creates a summary of metrics or parameters for the report.
    """
    summary = pd.DataFrame(results_dict).T
    save_csv(summary, "execution_summary")
    return summary

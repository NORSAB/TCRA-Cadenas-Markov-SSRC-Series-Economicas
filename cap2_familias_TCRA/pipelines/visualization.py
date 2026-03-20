import pandas as pd
from src.config import DATASETS, TABLES_DIR
from src.visualization.plots import plot_multiseries_grid, plot_individual_comparison, plot_optimization_heatmap, plot_alphas_distribution, plot_cross_validation_splits, plot_variant_comparison_bubble

def run_viz():
    print("--- Generating Enhanced Visualizations ---")

    # 1. PIB VISUALIZATIONS
    config_pib = DATASETS['pib']
    df_pib = pd.read_csv(config_pib['silver_path'], index_col=0, parse_dates=True)
    df_alphas_pib = pd.read_csv(config_pib['gold_path'], index_col=0, parse_dates=True)
    
    print("Generating PIB Grids (Original and Adjusted)...")
    plot_multiseries_grid(df_pib, "pib", suffix="original", show_adjustment=False)
    plot_multiseries_grid(df_pib, "pib", suffix="ajustado", show_adjustment=True, df_alphas=df_alphas_pib)
    
    for col in config_pib['series_cols']:
        plot_individual_comparison("pib", col, df_pib[col], df_alphas_pib)
        plot_alphas_distribution("pib", col, df_alphas_pib)
    
    # 2. FUELS VISUALIZATIONS
    config_fuels = DATASETS['combustibles']
    df_fuels = pd.read_csv(config_fuels['silver_path'], index_col=0, parse_dates=True)
    df_alphas_fuels = pd.read_csv(config_fuels['gold_path'], index_col=0, parse_dates=True)
    
    # Selection of only the 4 main variables for the grid
    df_fuels_sub = df_fuels[config_fuels['series_cols']]
    
    print("Generating Fuel Grids (Original and Adjusted)...")
    plot_multiseries_grid(df_fuels_sub, "combustibles", suffix="original", show_adjustment=False)
    plot_multiseries_grid(df_fuels_sub, "combustibles", suffix="ajustado", show_adjustment=True, df_alphas=df_alphas_fuels)
    
    for col in config_fuels['series_cols']:
        plot_individual_comparison("combustibles", col, df_fuels[col], df_alphas_fuels)
        plot_alphas_distribution("combustibles", col, df_alphas_fuels)

    # 3. OPTIMIZATION HEATMAPS AND BUBBLES
    for ds in ["combustibles", "pib"]:
        path = TABLES_DIR / f"grid_search_detailed_{ds}.csv"
        if path.exists():
            df_detailed = pd.read_csv(path)
            plot_optimization_heatmap(df_detailed, ds)
            plot_variant_comparison_bubble(df_detailed, ds)

    # 4. CROSS VALIDATION METHODOLOGY PLOTS
    # Gráficos demostrativos de cómo se partió la data (Para TODAS las series)
    if not df_fuels.empty:
        for col in config_fuels['series_cols']:
            plot_cross_validation_splits(df_fuels, "combustibles", col, 55)
    
    if not df_pib.empty:
        for col in config_pib['series_cols']:
            plot_cross_validation_splits(df_pib, "pib", col, 10)

    print("--- Visualization Generation Completed ---")

if __name__ == "__main__":
    run_viz()

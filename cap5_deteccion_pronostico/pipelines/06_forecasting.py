import os
import sys
import pandas as pd
import numpy as np

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import REQUIRED_COLUMNS, TABLES_DIR, MODELS_DIR

def run_forecasting():
    print("\n--- STEP 16: Forecast for the Next Period ---")
    
    silver_path = "data/silver/fuel_data_standardized.csv"
    if not os.path.exists(silver_path): return

    silver_df = pd.read_csv(silver_path)
    last_date = pd.to_datetime(silver_df['Date']).max()
    next_date = last_date + pd.Timedelta(weeks=1)
    
    forecast_results = []
    
    for col in REQUIRED_COLUMNS:
        # Cargar modelo
        p_path = os.path.join(MODELS_DIR, f"{col}_P_kmeans.npy")
        c_path = os.path.join(MODELS_DIR, f"{col}_centroids.npy")
        n_path = os.path.join(MODELS_DIR, f"{col}_regime_names.npy")
        audit_path = os.path.join(TABLES_DIR, f"{col}_discretization_audit.csv")
        
        if os.path.exists(p_path) and os.path.exists(audit_path):
            P = np.load(p_path)
            centroids = np.load(c_path)
            regime_names = np.load(n_path)
            audit_df = pd.read_csv(audit_path)
            
            last_state = int(audit_df['State_KMeans'].iloc[-1])
            last_price = audit_df[col].iloc[-1]
            
            # Predict
            next_state_probs = P[:, last_state]
            next_state = np.argmax(next_state_probs)
            confidence = next_state_probs[next_state]
            
            next_alpha = centroids[next_state]
            next_price = last_price * (1 + next_alpha)
            
            forecast_results.append({
                'Fuel': col,
                'Current Price': last_price,
                'Current Regime': regime_names[last_state],
                'Predicted Regime': regime_names[next_state],
                'Confidence': f"{confidence:.1%}",
                'Predicted Alpha': next_alpha,
                'Forecasted Price': next_price
            })

    df_forecast = pd.DataFrame(forecast_results)
    df_forecast.to_csv(os.path.join(TABLES_DIR, "next_week_forecast.csv"), index=False)
    print(f"\nForecast for week of {next_date.strftime('%Y-%m-%d')}:")
    print(df_forecast)

if __name__ == "__main__":
    run_forecasting()

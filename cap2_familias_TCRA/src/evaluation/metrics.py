import numpy as np

def calculate_metrics(y_true, y_pred):
    """
    Calculates various error metrics for time series.
    """
    # Remove NaNs
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t = y_true[mask]
    y_p = y_pred[mask]
    
    if len(y_t) == 0:
        return {
            "RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "R2": np.nan
        }

    mse = np.mean((y_t - y_p)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_t - y_p))
    medae = np.median(np.abs(y_t - y_p))
    
    # MSLE (Handling only positive for log)
    if np.all(y_t > 0) and np.all(y_p > 0):
        msle = np.mean((np.log1p(y_t) - np.log1p(y_p))**2)
    else:
        msle = np.nan

    # Avoid division by zero for MAPE
    mape = np.mean(np.abs((y_t - y_p) / np.where(y_t == 0, 1, y_t))) * 100
    
    ss_res = np.sum((y_t - y_p)**2)
    ss_tot = np.sum((y_t - np.mean(y_t))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MedAE": medae,
        "MSLE": msle,
        "MAPE": mape,
        "R2": r2
    }

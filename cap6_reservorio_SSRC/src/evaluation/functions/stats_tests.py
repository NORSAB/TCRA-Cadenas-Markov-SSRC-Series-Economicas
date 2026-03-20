"""
===================================================================
Pruebas Estadísticas para Comparación de Predicciones
===================================================================
Implementación del Diebold-Mariano Test (DM Test).
"""
import numpy as np
from scipy import stats

def diebold_mariano_test(y_true, y_pred1, y_pred2, h=1, crit='MSE'):
    """
    Calcula el test de Diebold-Mariano para comparar dos pronósticos.
    H0: Ambos modelos tienen la misma precisión.
    H1: El modelo 2 es significativamente distinto/mejor que el modelo 1.
    """
    y_true = np.array(y_true)
    y_pred1 = np.array(y_pred1)
    y_pred2 = np.array(y_pred2)
    
    T = len(y_true)
    
    if crit == 'MSE':
        e1 = (y_true - y_pred1)**2
        e2 = (y_true - y_pred2)**2
    else: # MAE
        e1 = np.abs(y_true - y_pred1)
        e2 = np.abs(y_true - y_pred2)
        
    # Diferencia de pérdida
    d = e1 - e2
    d_mean = np.mean(d)
    
    # Auto-covarianza (simplificado para h=1)
    gamma0 = np.var(d)
    
    # DM Statistic
    if gamma0 == 0:
        return 0.0, 1.0 # No hay diferencia
        
    dm_stat = d_mean / np.sqrt(gamma0 / T)
    
    # P-value (two-sided)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return dm_stat, p_value

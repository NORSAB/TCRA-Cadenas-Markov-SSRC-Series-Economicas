"""
Pipeline completo para el Capítulo 4: Modelo Híbrido TCRA-Markov con NNLS.
Genera la Tabla 4.3 (rendimiento predictivo por serie de combustible).

Lógica extraída de: Posible Cap4/Ej Combustible NNLS.ipynb
                     Posible Cap4/NNLS Combustible.ipynb

Uso: python pipeline.py
"""
import os, sys, io, numpy as np, pandas as pd
from scipy.optimize import nnls
from numpy import kron, identity, ones, zeros, diag

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ─── Configuración ───────────────────────────────────────────────────
from config import (
    DATA_PATH, OUTPUT_DIR, W, LAMBDA, K, SEED,
    THRESHOLDS, STATE_NAMES, FUEL_ORDER, TRAIN_RATIOS
)

np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_TO_INDEX = {'s1': 0, 's2': 1, 's3': 2, 's4': 3}
INDEX_TO_LABEL = {0: 's1', 1: 's2', 2: 's3', 3: 's4'}


# ═══════════════════════════════════════════════════════════════════════
# FUNCIONES CORE  (extraídas de notebooks originales, sin modificaciones)
# ═══════════════════════════════════════════════════════════════════════

def load_fuel_data(path):
    """Carga el CSV de combustibles y retorna un dict {nombre: np.array}."""
    df = pd.read_csv(path)
    fuel_series = {}
    for col in FUEL_ORDER:
        fuel_series[col] = np.asarray(df[col].values, dtype=float)
    return fuel_series


def calcular_alphas(serie, W_val=W, lambda_=LAMBDA):
    """Calcula alpha_t (Ec. TCRA). Replica notebook Cell 1."""
    alphas = []
    for t in range(W_val, len(serie)):
        numerador = 0.0
        denominador = 0.0
        for tau in range(t - W_val + 1, t + 1):
            weight = lambda_ ** (t - tau)
            numerador += weight * serie[tau] * serie[tau - 1]
            denominador += weight * serie[tau - 1] ** 2
        alpha_t = -1 + numerador / denominador
        alphas.append(alpha_t)
    return alphas


def asignar_estado_codigo(alpha):
    """Discretiza alpha con umbrales fijos (Cap 4). Replica notebook Cell 1."""
    if alpha < THRESHOLDS[0]:
        return 's1'   # caída
    elif alpha <= THRESHOLDS[1]:
        return 's2'   # estable
    elif alpha <= THRESHOLDS[2]:
        return 's3'   # subida
    else:
        return 's4'   # alza fuerte


def genera_X0_X1_C(estados_label, num_estados=K):
    """Genera matrices one-hot X0, X1 y conteo C. Replica notebook Cell 3/9."""
    estados = [LABEL_TO_INDEX[e] for e in estados_label]
    T = len(estados)
    X0 = np.zeros((num_estados, T - 1))
    X1 = np.zeros((num_estados, T - 1))
    for t in range(T - 1):
        X0[estados[t], t] = 1
        X1[estados[t + 1], t] = 1
    C = X1 @ X0.T
    return X0, X1, C


def estimate_P_from_C(C):
    """Estima P (MLE) normalizando columnas de C."""
    col_sums = C.sum(axis=0)
    col_sums[col_sums == 0] = 1
    return C / col_sums


def SRep(datos, ss):
    """Estimación NNLS/SRep. Replica notebook Cell 2."""
    n0 = datos.shape[0]
    S0 = datos[:, :ss]
    S0 = kron(S0, identity(n0)).T
    S1 = S0.T @ ((datos[:, 1:(1 + ss)]).T).reshape(ss * n0)
    C_mat = kron(identity(n0), ones((1, n0)))
    Mr = zeros((n0**2 + n0, n0**2))
    Mr[:n0**2, :] = S0.T @ S0
    Mr[n0**2:, :] = C_mat
    rhs = zeros((n0**2 + n0))
    rhs[:n0**2] = S1
    rhs[n0**2:] = 1
    c = zeros((n0**2, 1))
    c[:, 0] = nnls(Mr, rhs)[0]
    Pr = c.reshape(n0, n0).T
    Pr = Pr @ diag(1 / sum(Pr))
    return Pr


def compute_Ap(P, num_estados=K):
    """Calcula Ap via simulación + SRep. Replica notebook Cell 10."""
    p0 = zeros((num_estados, 100))
    p0[0, 0] = 1  # estado inicial s1
    for j in range(99):
        p0[:, j + 1] = P @ p0[:, j]
    return SRep(p0, ss=num_estados)


def get_predictions_and_accuracy(matrix, test_states_labels):
    """
    Predicción ONE-STEP-AHEAD: usa el estado ACTUAL del test para predecir
    el siguiente. Replica notebook Ej Combustible NNLS.ipynb.
    """
    predictions = []
    actual_outcomes = []
    for i in range(len(test_states_labels) - 1):
        current_label = test_states_labels[i]
        current_idx = LABEL_TO_INDEX[current_label]

        prob_next = matrix[:, current_idx]
        predicted_idx = np.argmax(prob_next)
        predicted_label = INDEX_TO_LABEL[predicted_idx]

        predictions.append(predicted_label)
        actual_outcomes.append(test_states_labels[i + 1])

    if not predictions:
        return 0.0, 0, predictions, actual_outcomes

    correct = sum(p == a for p, a in zip(predictions, actual_outcomes))
    accuracy = correct / len(predictions)
    return accuracy, len(predictions), predictions, actual_outcomes


# ═══════════════════════════════════════════════════════════════════════
# EJECUCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("PIPELINE TCROC-NNLS — Capítulo 4: Validación Predictiva")
    print(f"Parámetros: W={W}, lambda={LAMBDA}, K={K}, umbrales={THRESHOLDS}")
    print("Método de predicción: one-step-ahead (estado actual real)")
    print("=" * 70)

    fuel_series = load_fuel_data(DATA_PATH)
    print(f"Datos cargados: {list(fuel_series.keys())}")

    all_results = {}
    summary_rows = []

    for fuel in FUEL_ORDER:
        series = fuel_series[fuel]
        alphas = calcular_alphas(series, W, LAMBDA)
        states_full = [asignar_estado_codigo(a) for a in alphas]

        # Matrices completas (100%)
        _, _, C_full = genera_X0_X1_C(states_full, K)
        P_full = estimate_P_from_C(C_full)
        Ap_full = compute_Ap(P_full, K)

        print(f"\n{'─'*60}")
        print(f"  {fuel}: {len(alphas)} alphas, {len(states_full)} estados")
        print(f"  Distribución: {dict((INDEX_TO_LABEL[i], int(s)) for i, s in enumerate(np.bincount([LABEL_TO_INDEX[x] for x in states_full], minlength=K)))}")
        print(f"  Matriz C (100%):\n{C_full.astype(int)}")

        # Validación predictiva por partición
        results = []
        for ratio in TRAIN_RATIOS:
            split_idx = int(len(states_full) * ratio)
            states_train = states_full[:split_idx]
            states_test = states_full[split_idx:]

            if len(states_test) < 2:
                continue

            # Matrices de ENTRENAMIENTO
            _, _, C_train = genera_X0_X1_C(states_train, K)
            P_train = estimate_P_from_C(C_train)
            Ap_train = compute_Ap(P_train, K)

            # Accuracy con P_hat (MLE) y con Ap (SRep)
            acc_P, n_pred_P, _, _ = get_predictions_and_accuracy(P_train, states_test)
            acc_Ap, n_pred_Ap, _, _ = get_predictions_and_accuracy(Ap_train, states_test)

            pct_train = int(ratio * 100)
            pct_test = 100 - pct_train
            results.append({
                "Partición": f"{pct_train}/{pct_test}",
                "N_predicciones": n_pred_P,
                "Accuracy_P": round(acc_P * 100, 1),
                "Accuracy_Ap": round(acc_Ap * 100, 1),
            })

        all_results[fuel] = results
        df_r = pd.DataFrame(results)
        print(f"\n  Rendimiento predictivo (one-step-ahead):")
        print(f"  {df_r.to_string(index=False)}")

        accs_P = [r["Accuracy_P"] for r in results]
        accs_Ap = [r["Accuracy_Ap"] for r in results]
        max_acc_P = max(accs_P)
        max_acc_Ap = max(accs_Ap)
        avg_acc_P = round(np.mean(accs_P), 1)
        avg_acc_Ap = round(np.mean(accs_Ap), 1)

        summary_rows.append({
            "Serie": fuel,
            "Max_Acc_P": max_acc_P,
            "Avg_Acc_P": avg_acc_P,
            "Max_Acc_Ap": max_acc_Ap,
            "Avg_Acc_Ap": avg_acc_Ap,
        })
        print(f"  P_hat:  MAX={max_acc_P}% | AVG={avg_acc_P}%")
        print(f"  Ap:     MAX={max_acc_Ap}% | AVG={avg_acc_Ap}%")

    # ─── Guardar resultados ──────────────────────────────────────────
    df_summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "tabla_4_3_rendimiento_combustibles.csv")
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    detail_path = os.path.join(OUTPUT_DIR, "detalle_particiones.xlsx")
    with pd.ExcelWriter(detail_path) as writer:
        for fuel, results in all_results.items():
            pd.DataFrame(results).to_excel(writer, sheet_name=fuel, index=False)

    print(f"\n{'='*70}")
    print("RESUMEN FINAL — Tabla 4.3:")
    print(df_summary.to_string(index=False))
    print(f"\nGuardado en: {summary_path}")
    print(f"Detalle en:  {detail_path}")
    print("=" * 70)

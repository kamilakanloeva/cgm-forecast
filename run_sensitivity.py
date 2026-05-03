"""
run_sensitivity.py
==================
Анализ устойчивости результатов XGBoost и LSTM к выбору зерна генератора.

Запускает оба алгоритма с зёрнами SEEDS и сохраняет результаты в
results_sensitivity/. Результаты ARIMA и Хольта берутся из основного
эксперимента и добавляются вручную для построения итоговых графиков.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# ─── Пути ────────────────────────────────────────────────────────────────────

ROOT_DIR        = Path(__file__).resolve().parent
RESULTS_DIR     = ROOT_DIR / "results_sensitivity"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT_DIR))

from src.config import HORIZONS, TRAIN_RATIO, WINDOW_K, TEST_LIMIT
from src.config import XGB_TREES, XGB_DEPTH, XGB_LR
from src.config import LSTM_HIDDEN, LSTM_LAYERS, LSTM_EPOCHS, LSTM_BATCH, LSTM_LR
from src.data_loader import load_series
from src.features import temporal_split
from src.metrics import compute_metrics

# ─── Параметры анализа ───────────────────────────────────────────────────────

SEEDS = [12, 7, 42, 99, 1111]

MINUTES = [h * 5 for h in HORIZONS]

# Результаты ARIMA и Хольта из основного эксперимента (results/metrics.csv)
# Вносятся вручную — эти модели детерминированы и не зависят от зерна
ARIMA_MAE = [2.672, 4.956, 6.748, 8.361, 9.810, 11.272,
             12.581, 13.611, 14.547, 15.401, 16.234, 17.066]
HOLT_MAE  = [2.897, 5.580, 8.060, 10.527, 13.192, 16.059,
             18.742, 21.396, 24.218, 27.163, 30.087, 33.096]


# ─── Вспомогательные функции ─────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Фиксирует зерно для numpy и PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_xgb(train: np.ndarray, test: np.ndarray, seed: int) -> list[float]:
    """Запускает XGBoost с заданным зерном, возвращает список MAE по горизонтам."""
    from xgboost import XGBRegressor
    from src.features import make_lag_matrix, make_train_windows

    mae_list = []
    test_limited = test[:TEST_LIMIT]

    for h in HORIZONS:
        X_train, y_train = make_train_windows(train, h=h)
        _, _, X_test, y_test = make_lag_matrix(train, test_limited, h=h)

        model = XGBRegressor(
            n_estimators=XGB_TREES,
            max_depth=XGB_DEPTH,
            learning_rate=XGB_LR,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            verbosity=0,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = compute_metrics(y_test, preds)
        mae_list.append(metrics["MAE"])
        print(f"    h={h}: MAE={metrics['MAE']:.3f}")

    return mae_list


def run_lstm(train: np.ndarray, test: np.ndarray, seed: int) -> list[float]:
    """Запускает LSTM с заданным зерном, возвращает список MAE по горизонтам."""
    from src.models.lstm_model import LSTMModel

    set_seed(seed)
    model = LSTMModel()
    model.fit(train)

    mae_list = []
    test_limited = test[:TEST_LIMIT]

    for h in HORIZONS:
        y_pred, y_true, _ = model.predict(test_limited, h)
        metrics = compute_metrics(y_true, y_pred)
        mae_list.append(metrics["MAE"])
        print(f"    h={h}: MAE={metrics['MAE']:.3f}")

    return mae_list


# ─── Построение графиков ─────────────────────────────────────────────────────

def plot_sensitivity(results: dict) -> None:
    """
    Строит график MAE(h) с диапазоном min–max для XGBoost и LSTM
    и фиксированными линиями для ARIMA и Хольта.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # ARIMA и Хольт — фиксированные линии
    ax.plot(MINUTES, ARIMA_MAE, color="black", linestyle="-",
            marker="o", linewidth=1.8, markersize=5, label="ARIMA")
    ax.plot(MINUTES, HOLT_MAE, color="black", linestyle="--",
            marker="s", linewidth=1.8, markersize=5, label="Хольт")

    # XGBoost — среднее + диапазон
    xgb_arr = np.array(results["XGBoost"])
    xgb_mean = xgb_arr.mean(axis=0)
    xgb_min  = xgb_arr.min(axis=0)
    xgb_max  = xgb_arr.max(axis=0)
    ax.plot(MINUTES, xgb_mean, color="black", linestyle="-.",
            marker="^", linewidth=1.8, markersize=5, label="XGBoost (среднее)")
    ax.fill_between(MINUTES, xgb_min, xgb_max,
                    color="black", alpha=0.12, label="XGBoost (min–max)")

    # LSTM — среднее + диапазон
    lstm_arr = np.array(results["LSTM"])
    lstm_mean = lstm_arr.mean(axis=0)
    lstm_min  = lstm_arr.min(axis=0)
    lstm_max  = lstm_arr.max(axis=0)
    ax.plot(MINUTES, lstm_mean, color="black", linestyle=":",
            marker="D", linewidth=1.8, markersize=5, label="LSTM (среднее)")
    ax.fill_between(MINUTES, lstm_min, lstm_max,
                    color="black", alpha=0.06, label="LSTM (min–max)")

    # Ориентир 30 минут
    ax.axvline(30, color="gray", linestyle="--", linewidth=1.2,
               label="30 мин (ориентир)")

    ax.set_xticks(MINUTES)
    ax.set_xlabel("Горизонт, мин", fontsize=11)
    ax.set_ylabel("MAE, mg/dL", fontsize=11)
    ax.set_title("Устойчивость MAE к выбору зерна (5 прогонов)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9)
    fig.tight_layout()

    path = RESULTS_DIR / "fig_sensitivity_mae.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Сохранён: {path}")


def save_sensitivity_csv(results: dict) -> None:
    """Сохраняет все прогоны в CSV."""
    rows = []
    for model_name, runs in results.items():
        for seed_idx, mae_list in enumerate(runs):
            seed = SEEDS[seed_idx]
            for i, h in enumerate(HORIZONS):
                rows.append({
                    "Модель": model_name,
                    "SEED":   seed,
                    "h":      h,
                    "Минуты": h * 5,
                    "MAE":    round(mae_list[i], 4),
                })

    df = pd.DataFrame(rows)
    path = RESULTS_DIR / "sensitivity_results.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[csv]  Сохранён: {path}")


def save_summary_csv(results: dict) -> None:
    """Сохраняет сводку: среднее, мин, макс MAE по горизонтам."""
    rows = []
    for model_name, runs in results.items():
        arr = np.array(runs)
        for i, h in enumerate(HORIZONS):
            rows.append({
                "Модель":     model_name,
                "h":          h,
                "Минуты":     h * 5,
                "MAE_mean":   round(arr[:, i].mean(), 4),
                "MAE_min":    round(arr[:, i].min(), 4),
                "MAE_max":    round(arr[:, i].max(), 4),
                "MAE_std":    round(arr[:, i].std(), 4),
            })

    df = pd.DataFrame(rows)
    path = RESULTS_DIR / "sensitivity_summary.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[csv]  Сохранён: {path}")
    print("\n=== Сводка по устойчивости (среднее MAE) ===")
    pivot = df.pivot_table(index="Модель", columns="Минуты",
                           values="MAE_mean").round(2)
    print(pivot.to_string())


# ─── Главная функция ─────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Анализ устойчивости: XGBoost и LSTM")
    print(f"  Зёрна: {SEEDS}")
    print("=" * 60)

    series = load_series()
    train, test = temporal_split(series)

    results = {"XGBoost": [], "LSTM": []}

    for seed in SEEDS:
        print(f"\n{'─'*60}")
        print(f"  SEED = {seed}")
        print(f"{'─'*60}")

        print("\n  [XGBoost]")
        set_seed(seed)
        xgb_mae = run_xgb(train, test, seed)
        results["XGBoost"].append(xgb_mae)

        print("\n  [LSTM]")
        set_seed(seed)
        lstm_mae = run_lstm(train, test, seed)
        results["LSTM"].append(lstm_mae)

    print("\n[saving] Сохранение результатов...")
    save_sensitivity_csv(results)
    save_summary_csv(results)
    plot_sensitivity(results)

    print("\n" + "=" * 60)
    print("  Анализ устойчивости завершён.")
    print(f"  Результаты: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

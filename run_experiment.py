import pandas as pd
from src.config import HORIZONS, RESULTS_DIR, DEBUG_MODE
from src.config import HORIZONS, RESULTS_DIR
from src.data_loader import load_series
from src.features import temporal_split
from src.metrics import compute_metrics, find_degradation_horizon
from src.models.arima_model import ARIMAModel
from src.models.hw_model import HWModel
from src.models.lstm_model import LSTMModel
from src.models.xgb_model import XGBModel
from src.plots import save_all


def build_results_dict() -> dict:
    """
    Создаёт пустой словарь для хранения результатов эксперимента.

    Структура: {имя_модели: {"mae": [], "rmse": [], "inf_time": []}}
    """
    model_names = ["ARIMA", "Хольт", "XGBoost", "LSTM"]
    return {
        name: {"mae": [], "rmse": [], "inf_time": []}
        for name in model_names
    }


def run_models(train, test, results: dict) -> None:
    """
    Обучает все модели и собирает метрики по всем горизонтам.

    Для каждой модели: fit(train), затем для каждого h: predict(test, h).
    Метрики добавляются в словарь results.

    Параметры:
        train   : обучающий массив
        test    : тестовый массив
        results : словарь для записи результатов
    """
    # Создаём все четыре модели
    models = {
        "ARIMA":         ARIMAModel(),
        "Хольт":          HWModel(),
        "XGBoost":        XGBModel(),
        "LSTM":           LSTMModel(),
    }

    for model_name, model in models.items():
        print(f"\n{'='*60}")
        print(f"  Модель: {model_name}")
        print(f"{'='*60}")

        # Обучаем модель один раз на train
        model.fit(train)

        # Прогнозируем на каждом горизонте
        for h in HORIZONS:
            print(f"\n  [h={h}, {h*5} мин]")
            y_pred, y_true, inf_time = model.predict(test, h)

            metrics = compute_metrics(y_true, y_pred)
            results[model_name]["mae"].append(metrics["MAE"])
            results[model_name]["rmse"].append(metrics["RMSE"])
            results[model_name]["inf_time"].append(inf_time)

            print(f"  MAE={metrics['MAE']:.3f}  "
                  f"RMSE={metrics['RMSE']:.3f}  "
                  f"inf={inf_time*1000:.3f} мс")


def save_metrics_csv(results: dict) -> None:
    """
    Сохраняет MAE и RMSE всех моделей по горизонтам в CSV.

    Каждая строка — одна комбинация модель × горизонт.

    Параметры:
        results : словарь результатов эксперимента
    """
    rows = []
    for model_name, data in results.items():
        for i, h in enumerate(HORIZONS):
            rows.append({
                "Модель":    model_name,
                "h":         h,
                "Минуты":    h * 5,
                "MAE":       round(data["mae"][i], 4),
                "RMSE":      round(data["rmse"][i], 4),
                "Инференс_мс": round(data["inf_time"][i] * 1000, 4),
            })

    df   = pd.DataFrame(rows)
    path = RESULTS_DIR / "metrics.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n[csv] Метрики сохранены: {path}")


def save_degradation_csv(results: dict) -> None:
    """
    Определяет горизонт деградации h* для каждой модели и сохраняет в CSV.

    Параметры:
        results : словарь результатов эксперимента
    """
    rows = []
    for model_name, data in results.items():
        h_star = find_degradation_horizon(data["mae"])
        rows.append({
            "Модель":         model_name,
            "h* (шагов)":     h_star if h_star else "не найден",
            "h* (минут)":     h_star * 5 if h_star else "—",
            "MAE(h*)":        round(data["mae"][h_star - 1], 2) if h_star else "—",
            "MAE(h=6, 30мин)": round(data["mae"][5], 2),
            "MAE(h=1)":       round(data["mae"][0], 2),
        })

    df   = pd.DataFrame(rows)
    path = RESULTS_DIR / "degradation.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[csv] Горизонты деградации сохранены: {path}")

    print("\n=== Горизонты деградации ===")
    print(df.to_string(index=False))


def main() -> None:
    """
    Точка входа. Запускает полный эксперимент по шагам:

    1. Загрузка данных
    2. Разбивка на train/test
    3. Обучение моделей и сбор метрик
    4. Сохранение CSV
    5. Построение графиков
    """
    print("=" * 60)
    print("  Эксперимент: оптимальный горизонт прогнозирования CGM")
    print("  Пациент 588, OhioT1DM")
    print("=" * 60)

    # Шаг 1 — загрузка данных
    print("\n--- Шаг 1: Загрузка данных ---")
    series = load_series()

    if DEBUG_MODE:
        series = series.iloc[:400]
        print("[DEBUG] Режим отладки: первые 400 точек")

    # Шаг 2 — разбивка на train/test
    print("\n--- Шаг 2: Разбивка на train/test ---")
    train, test = temporal_split(series)

    # Шаг 3 — обучение и прогнозирование
    print("\n--- Шаг 3: Обучение моделей и сбор метрик ---")
    results = build_results_dict()
    run_models(train, test, results)

    # Шаг 4 — сохранение CSV
    print("\n--- Шаг 4: Сохранение результатов ---")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_metrics_csv(results)
    save_degradation_csv(results)

    # Шаг 5 — графики
    print("\n--- Шаг 5: Построение графиков ---")
    save_all(results)

    print("\n" + "=" * 60)
    print("  Эксперимент завершён.")
    print(f"  Результаты: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

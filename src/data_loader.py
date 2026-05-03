import pandas as pd

from src.config import DATA_PATH, ROOT_DIR


def load_series() -> pd.Series:
    """
    Читает glucose_588.csv и возвращает временной ряд глюкозы.

    Возвращает pd.Series с:
      - индексом типа DatetimeIndex (временные метки)
      - значениями типа float (уровень глюкозы, mg/dL)

    Выбрасывает FileNotFoundError, если CSV не найден.
    Выбрасывает ValueError, если ряд пустой после загрузки.
    """
    csv_path = ROOT_DIR / "data" / "glucose_588.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Файл не найден: {csv_path}\n"
            "Запустите prepare_data.py для генерации CSV из XML."
        )

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

    if df.empty:
        raise ValueError("CSV загружен, но не содержит ни одной строки.")

    # Устанавливаем timestamp как индекс и оставляем только столбец glucose
    series = df.set_index("timestamp")["glucose"]

    print(f"[data] Загружено точек : {len(series)}")
    print(f"[data] Период          : {series.index[0]} — {series.index[-1]}")
    print(f"[data] Диапазон        : {series.min():.0f} — {series.max():.0f} mg/dL")

    return series

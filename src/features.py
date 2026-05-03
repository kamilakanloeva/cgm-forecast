import numpy as np
import pandas as pd

from src.config import TRAIN_RATIO, WINDOW_K


def temporal_split(
    series: pd.Series,
    ratio: float = TRAIN_RATIO,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Делит временной ряд на обучающую и тестовую части.

    Разбивка строго темпоральная — первые ratio*n точек идут на обучение,
    оставшиеся на тест. Перемешивание не применяется, порядок сохраняется.

    Параметры:
        series : исходный временной ряд глюкозы
        ratio  : доля обучающей выборки (по умолчанию 0.80 из config.py)

    Возвращает:
        train : np.ndarray, первые ratio*n наблюдений
        test  : np.ndarray, оставшиеся (1-ratio)*n наблюдений
    """
    n     = len(series)
    t_end = int(n * ratio)

    train = series.iloc[:t_end].to_numpy(dtype=float)
    test  = series.iloc[t_end:].to_numpy(dtype=float)

    print(f"[split] Всего точек : {n}")
    print(f"[split] Train       : {len(train)}  ({len(train)/n:.0%})")
    print(f"[split] Test        : {len(test)}   ({len(test)/n:.0%})")

    return train, test


def make_lag_matrix(
    train: np.ndarray,
    test: np.ndarray,
    k: int = WINDOW_K,
    h: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Формирует лаговые матрицы признаков и векторы целей для горизонта h.

    Индексация согласно Главе 2:
      Train объекты : t = k ... T-h,   число = T - h - k + 1
      Test объекты  : t = T+k ... n-h, число = n - h - (T+k) + 1

    Для первого тестового окна нужны последние k точек train —
    они подклеиваются к test внутри функции.

    Параметры:
        train : обучающий массив длины T
        test  : тестовый массив длины n-T
        k     : длина окна истории (по умолчанию WINDOW_K из config.py)
        h     : горизонт прогнозирования в шагах

    Возвращает:
        X_train : матрица признаков для обучения,  shape (T-h-k+1, k)
        y_train : вектор целей для обучения,        shape (T-h-k+1,)
        X_test  : матрица признаков для теста,      shape (n-h-(T+k)+1, k)
        y_test  : вектор целей для теста,           shape (n-h-(T+k)+1,)
    """
    # Последние k точек train нужны как история для первого тестового окна
    test_extended = np.concatenate([train[-k:], test])

    X_train, y_train = _build_windows(train, k, h)
    X_test,  y_test  = _build_windows(test_extended, k, h)

    return X_train, y_train, X_test, y_test


def _build_windows(
    arr: np.ndarray,
    k: int,
    h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Нарезает массив на скользящие окна длины k с целью на шаг h вперёд.

    Для каждого индекса i:
      X[i] = arr[i : i+k]        — окно истории
      y[i] = arr[i + k - 1 + h]  — целевое значение через h шагов

    Параметры:
        arr : одномерный массив
        k   : длина окна
        h   : горизонт прогнозирования

    Возвращает:
        X : матрица окон,   shape (n_samples, k)
        y : вектор целей,   shape (n_samples,)
    """
    n        = len(arr)
    n_samples = n - k - h + 1

    if n_samples <= 0:
        raise ValueError(
            f"Недостаточно данных для построения окон: "
            f"len(arr)={n}, k={k}, h={h}. "
            f"Нужно минимум {k + h} точек."
        )

    X = np.array([arr[i : i + k]           for i in range(n_samples)])
    y = np.array([arr[i + k - 1 + h]       for i in range(n_samples)])

    return X, y


def make_train_windows(
    train: np.ndarray,
    k: int = WINDOW_K,
    h: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Формирует лаговую матрицу только из обучающего массива.

    Используется в XGBoost и LSTM для обучения модели на горизонте h
    без необходимости передавать тестовый массив.

    Параметры:
        train : обучающий массив наблюдений
        k     : длина окна истории
        h     : горизонт прогнозирования в шагах

    Возвращает:
        X_train : матрица признаков, shape (T-h-k+1, k)
        y_train : вектор целей,      shape (T-h-k+1,)
    """
    return _build_windows(train, k, h)

import numpy as np

from src.config import KAPPA, HORIZONS


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычисляет среднюю абсолютную ошибку (MAE).

    MAE = среднее |y_true - y_pred|

    Параметры:
        y_true : массив истинных значений
        y_pred : массив предсказанных значений

    Возвращает:
        MAE в единицах измерения ряда (mg/dL)
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Вычисляет среднеквадратичную ошибку (RMSE).

    RMSE = корень из среднего (y_true - y_pred)²

    Параметры:
        y_true : массив истинных значений
        y_pred : массив предсказанных значений

    Возвращает:
        RMSE в единицах измерения ряда (mg/dL)
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Вычисляет MAE и RMSE одновременно.

    Параметры:
        y_true : массив истинных значений
        y_pred : массив предсказанных значений

    Возвращает:
        словарь {"MAE": ..., "RMSE": ...}
    """
    return {
        "MAE":  mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
    }


def find_degradation_horizon(
    mae_list: list[float],
    kappa: float = KAPPA,
) -> int | None:
    """
    Определяет горизонт деградации h* по критерию скачка ΔMAE.

    Горизонт h* — первый горизонт где выполняется условие:
        ΔMAE(h) > κ × среднее(ΔMAE(j)), j = 2 ... h-1

    Проверка начинается с h=3, так как ΔMAE(1) не определён,
    а для сравнения нужен хотя бы один предыдущий прирост ΔMAE(2).

    Параметры:
        mae_list : список MAE по горизонтам, mae_list[0] = MAE(h=1)
        kappa    : порог скачка (по умолчанию KAPPA из config.py)

    Возвращает:
        h* — номер горизонта деградации (1-based)
        None — если скачка не обнаружено
    """
    # Вычисляем приросты ΔMAE(h) для h = 2, 3, ..., 12
    # deltas[0] = ΔMAE(h=2), deltas[1] = ΔMAE(h=3), ...
    deltas = [mae_list[i] - mae_list[i - 1] for i in range(1, len(mae_list))]

    # Проверяем начиная с h=3 — нужен хотя бы один предыдущий прирост
    for i in range(1, len(deltas)):
        h            = i + 2                         # текущий горизонт (1-based)
        delta_h      = deltas[i]                     # ΔMAE(h)
        prev_mean    = float(np.mean(deltas[:i]))    # среднее ΔMAE(j), j=2..h-1

        # Среднее предыдущих приростов должно быть положительным —
        # иначе ошибка не росла и сравнение не имеет смысла
        if prev_mean > 0 and delta_h > kappa * prev_mean:
            return h

    return None

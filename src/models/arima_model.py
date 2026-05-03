import time

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from src.config import ARIMA_MAX_ORDER, TEST_LIMIT
from src.models.base import BaseModel


class ARIMAModel(BaseModel):
    """
    Модель ARIMA(p, d, q) для прогнозирования уровня глюкозы.

    Порядок d определяется тестом ADF на стационарность.
    Параметры p и q подбираются перебором по критерию AIC
    из диапазона {0, 1, 2, 3} x {0, 1, 2, 3}.

    Порядок (p, d, q) фиксируется в fit и не меняется между горизонтами —
    чтобы рост ошибки отражал только увеличение h, а не смену модели.
    """

    def __init__(self) -> None:
        self.order: tuple[int, int, int] | None = None
        self._train: np.ndarray | None = None

    def fit(self, train: np.ndarray) -> None:
        """
        Определяет порядок (p, d, q) и сохраняет обучающий массив.

        Шаг 1: тест ADF определяет d.
        Шаг 2: перебор p и q по AIC определяет оптимальные параметры.

        Параметры:
            train : обучающий массив наблюдений
        """
        self._train = train
        d = self._find_d(train)
        p, q = self._find_pq(train, d)
        self.order = (p, d, q)
        print(f"[ARIMA] Подобранный порядок: {self.order}")

    def predict(
        self,
        test: np.ndarray,
        h: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Строит прогнозы для каждой тестовой точки на горизонте h.

        Тест ограничивается TEST_LIMIT точками для всех моделей одинаково —
        чтобы сравнение оставалось честным.

        Для каждой точки i история расширяется на одну точку:
            история = train + test[:i]
        Затем модель переобучается на этой истории и прогнозирует
        ровно h шагов вперёд. Истинное значение — test[i + h - 1].

        Параметры:
            test : тестовый массив наблюдений
            h    : горизонт прогнозирования в шагах

        Возвращает:
            y_pred         : массив прогнозов
            y_true         : массив истинных значений
            inference_time : среднее время одного прогноза в секундах
        """
        if self._train is None:
            raise RuntimeError("Сначала вызовите fit() для обучения модели.")

        # Ограничиваем тест одинаково для всех моделей
        test = test[:TEST_LIMIT]

        n_test    = len(test)
        n_samples = n_test - h + 1

        y_pred = np.empty(n_samples)
        y_true = np.empty(n_samples)
        times  = np.empty(n_samples)

        for i in range(n_samples):
            history = np.concatenate([self._train, test[:i]])

            t_start = time.perf_counter()

            model     = ARIMA(history, order=self.order).fit()
            forecast  = model.forecast(steps=h)
            y_pred[i] = float(forecast[-1])

            times[i] = time.perf_counter() - t_start

            y_true[i] = test[i + h - 1]

        return y_pred, y_true, float(np.mean(times))

    def _find_d(self, train: np.ndarray) -> int:
        """
        Определяет порядок интегрирования d тестом ADF.

        Если p-value < 0.05 — ряд стационарен, d = 0.
        Иначе — ряд нестационарен, d = 1.

        Параметры:
            train : обучающий массив наблюдений

        Возвращает:
            d : порядок интегрирования (0 или 1)
        """
        _, p_value, *_ = adfuller(train, autolag="AIC")
        d = 0 if p_value < 0.05 else 1
        print(f"[ARIMA] Тест ADF: p-value={p_value:.4f}, d={d}")
        return d

    def _find_pq(self, train: np.ndarray, d: int) -> tuple[int, int]:
        """
        Перебирает комбинации (p, q) и выбирает лучшую по AIC.

        Перебор ведётся по сетке p, q в {0, 1, 2, 3} — итого 16 комбинаций.
        Комбинации при которых statsmodels выбрасывает ошибку пропускаются.

        Параметры:
            train : обучающий массив наблюдений
            d     : фиксированный порядок интегрирования

        Возвращает:
            (p, q) : оптимальные параметры по критерию AIC
        """
        best_aic       = np.inf
        best_p, best_q = 1, 1

        for p in range(ARIMA_MAX_ORDER + 1):
            for q in range(ARIMA_MAX_ORDER + 1):
                try:
                    result = ARIMA(train, order=(p, d, q)).fit()
                    if result.aic < best_aic:
                        best_aic       = result.aic
                        best_p, best_q = p, q
                except Exception:
                    pass

        print(f"[ARIMA] Лучший (p, q) = ({best_p}, {best_q}), AIC={best_aic:.2f}")
        return best_p, best_q

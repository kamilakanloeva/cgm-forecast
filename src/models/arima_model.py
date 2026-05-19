import time

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from src.config import ARIMA_MAX_ORDER, MAX_H, TEST_LIMIT
from src.models.base import BaseModel


class ARIMAModel(BaseModel):
    """
    Модель ARIMA(p, d, q) для прогнозирования уровня глюкозы.

    Порядок d определяется тестом ADF на стационарность.
    Параметры p и q подбираются перебором по критерию AIC
    из диапазона {0, 1, 2, 3} x {0, 1, 2, 3}.

    Порядок (p, d, q) фиксируется в fit и не меняется между горизонтами —
    чтобы рост ошибки отражал только увеличение h, а не смену модели.

    Online-режим
    ------------
    Параметры модели подбираются один раз на обучающей части ряда.
    При прогнозировании на тестовых точках параметры не переоцениваются:
    для каждой стартовой точки i создаётся новый объект результатов на
    расширенной истории train + test[:i] с фиксированными параметрами,
    после чего выполняется многошаговый прогноз forecast(MAX_H).

    Этот режим соответствует реалистичной эксплуатации CGM-системы,
    в которой параметры модели калибруются заранее, а на каждом новом
    измерении выполняется только ассимиляция нового наблюдения и
    построение прогноза.

    Результат полного многошагового прогноза кэшируется в self._forecast_cache,
    чтобы извлечение прогнозов для разных горизонтов h из одной и той же
    стартовой точки i выполнялось без повторной обработки.
    """

    def __init__(self) -> None:
        self.order: tuple[int, int, int] | None = None
        self._train: np.ndarray | None = None

        # Результаты подгонки ARIMA на обучающей части ряда.
        # Используются для online-применения к расширенным историям
        # через apply(history, refit=False).
        self._results = None

        # Кэш многошаговых прогнозов для стартовых точек тестовой выборки.
        self._forecast_cache: dict[int, np.ndarray] = {}
        self._refit_time_cache: dict[int, float] = {}

    def fit(self, train: np.ndarray) -> None:
        """
        Определяет порядок (p, d, q), выполняет одноразовую подгонку
        параметров на обучающей части ряда.

        Шаг 1: тест ADF определяет d.
        Шаг 2: перебор p и q по AIC определяет оптимальные параметры.
        Шаг 3: финальная подгонка ARIMA(p, d, q) на train —
               её результаты используются в online-режиме при прогнозе.

        Параметры:
            train : обучающий массив наблюдений
        """
        self._train = train
        self._forecast_cache.clear()
        self._refit_time_cache.clear()

        d = self._find_d(train)
        p, q = self._find_pq(train, d)
        self.order = (p, d, q)

        # Финальная подгонка параметров на train.
        # В дальнейшем эти параметры применяются к расширенным историям
        # без повторной оптимизации (online-режим).
        self._results = ARIMA(train, order=self.order).fit()

        print(f"[ARIMA] Подобранный порядок: {self.order}")

    def predict(
        self,
        test: np.ndarray,
        h: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Строит прогнозы для каждой тестовой точки на горизонте h
        в online-режиме: параметры модели не переоцениваются,
        новые наблюдения ассимилируются через apply(refit=False).

        Параметры:
            test : тестовый массив наблюдений
            h    : горизонт прогнозирования в шагах

        Возвращает:
            y_pred         : массив прогнозов
            y_true         : массив истинных значений
            inference_time : среднее время на одну стартовую точку в секундах
        """
        if self._train is None or self._results is None:
            raise RuntimeError("Сначала вызовите fit() для обучения модели.")

        # Ограничиваем тест одинаково для всех моделей
        test = test[:TEST_LIMIT]

        n_test    = len(test)
        n_samples = n_test - h + 1

        y_pred = np.empty(n_samples)
        y_true = np.empty(n_samples)
        times  = np.empty(n_samples)

        for i in range(n_samples):
            forecast_arr, elapsed = self._forecast_for_start(test, i)

            y_pred[i] = forecast_arr[h - 1]
            times[i]  = elapsed
            y_true[i] = test[i + h - 1]

        return y_pred, y_true, float(np.mean(times))

    def _forecast_for_start(
        self,
        test: np.ndarray,
        i: int,
    ) -> tuple[np.ndarray, float]:
        """
        Возвращает массив прогнозов длины MAX_H для стартовой точки i
        в online-режиме.

        При первом обращении: применяет ранее подобранные параметры
        к истории train + test[:i] через apply(refit=False),
        выполняет forecast(steps=MAX_H), кэширует.
        """
        if i in self._forecast_cache:
            return self._forecast_cache[i], 0.0

        history = np.concatenate([self._train, test[:i]])

        t_start = time.perf_counter()
        # apply с refit=False создаёт новый объект результатов на новой
        # истории, используя параметры, оценённые на train. Это и есть
        # online-режим: фильтр Калмана пропускает новые наблюдения через
        # state space модели, но параметры не пересчитываются.
        results_i = self._results.apply(history, refit=False)
        forecast_arr = np.asarray(
            results_i.forecast(steps=MAX_H), dtype=float,
        )
        elapsed = time.perf_counter() - t_start

        self._forecast_cache[i]   = forecast_arr
        self._refit_time_cache[i] = elapsed

        return forecast_arr, elapsed

    def _find_d(self, train: np.ndarray) -> int:
        """
        Определяет порядок интегрирования d тестом ADF.

        Если p-value < 0.05 — ряд стационарен, d = 0.
        Иначе — ряд нестационарен, d = 1.
        """
        _, p_value, *_ = adfuller(train, autolag="AIC")
        d = 0 if p_value < 0.05 else 1
        print(f"[ARIMA] Тест ADF: p-value={p_value:.4f}, d={d}")
        return d

    def _find_pq(self, train: np.ndarray, d: int) -> tuple[int, int]:
        """
        Перебирает комбинации (p, q) и выбирает лучшую по AIC.
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

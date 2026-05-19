import time

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.config import MAX_H, TEST_LIMIT
from src.models.base import BaseModel


class HWModel(BaseModel):
    """
    Модель Хольта (двухпараметрическое экспоненциальное сглаживание).

    Учитывает уровень и тренд без сезонной компоненты. Сезонность
    исключена ввиду высокой вычислительной стоимости переобучения
    модели Хольта-Уинтерса с периодом m=288 на каждом шаге скользящего
    окна — это делало бы эксперимент вычислительно неприемлемым.

    Online-режим
    ------------
    Параметры α и β подбираются один раз на обучающей части ряда.
    Конечное состояние (уровень l_T и тренд b_T) сохраняется, и при
    прогнозировании на тестовых точках обновляется инкрементально по
    рекуррентным формулам Хольта:

        l_t = α · y_t + (1 - α) · (l_{t-1} + b_{t-1})
        b_t = β · (l_t - l_{t-1}) + (1 - β) · b_{t-1}

    Прогноз на h шагов вперёд от момента t:
        ŷ_{t+h} = l_t + h · b_t

    Такой режим соответствует реалистичной эксплуатации: параметры
    калибруются заранее, на каждом новом измерении состояние обновляется
    за O(1) операций, прогноз на MAX_H горизонтов — за O(MAX_H).
    Это согласуется с теоретической оценкой инференса O(1) из раздела 1.4.
    """

    def __init__(self) -> None:
        self._train: np.ndarray | None = None

        # Параметры сглаживания, оценённые на train. После fit неизменны.
        self._alpha: float = 0.0
        self._beta: float = 0.0

        # Конечное состояние модели после обучения на train.
        # Используется как стартовая точка для обновления на тестовых
        # наблюдениях.
        self._level_train_end: float = 0.0
        self._trend_train_end: float = 0.0

        # Кэш состояния (l, b) и прогнозов для стартовых точек теста.
        # _state_cache[i] = (l_after_test_prefix_i, b_after_test_prefix_i),
        # то есть состояние модели после ассимиляции test[0..i-1].
        self._state_cache: dict[int, tuple[float, float]] = {}
        self._forecast_cache: dict[int, np.ndarray] = {}
        self._refit_time_cache: dict[int, float] = {}

    def fit(self, train: np.ndarray) -> None:
        """
        Подбирает α и β по обучающей части ряда (один раз), сохраняет
        конечное состояние (уровень и тренд) для последующего online-режима.

        Параметры:
            train : обучающий массив наблюдений
        """
        self._train = train
        self._state_cache.clear()
        self._forecast_cache.clear()
        self._refit_time_cache.clear()

        try:
            results = ExponentialSmoothing(
                train,
                trend="add",
                seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)
            # Извлекаем оптимизированные параметры
            self._alpha = float(results.params["smoothing_level"])
            self._beta  = float(results.params["smoothing_trend"])
            # Конечное состояние = уровень и тренд после обработки последнего
            # наблюдения train. У statsmodels это level[-1] и trend[-1].
            self._level_train_end = float(results.level[-1])
            self._trend_train_end = float(results.trend[-1])
        except Exception:
            # Fallback: вырожденные параметры (наивный one-step)
            self._alpha = 1.0
            self._beta  = 0.0
            self._level_train_end = float(train[-1])
            self._trend_train_end = 0.0

        # Стартовое состояние для тестовой выборки (i=0): модель только что
        # обработала весь train, тестовые наблюдения ещё не ассимилированы.
        self._state_cache[0] = (self._level_train_end, self._trend_train_end)

        print(f"[HW] Параметры α={self._alpha:.4f}, β={self._beta:.4f}; "
              f"l_T={self._level_train_end:.2f}, b_T={self._trend_train_end:.4f}")
        print(f"[HW] Обучающий массив сохранён ({len(train)} точек).")

    def predict(
        self,
        test: np.ndarray,
        h: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Строит прогнозы для каждой тестовой точки на горизонте h в
        online-режиме: состояние обновляется инкрементально по рекуррентным
        формулам Хольта, параметры α и β не переоцениваются.

        Параметры:
            test : тестовый массив наблюдений
            h    : горизонт прогнозирования в шагах

        Возвращает:
            y_pred         : массив прогнозов
            y_true         : массив истинных значений
            inference_time : среднее время на одну стартовую точку в секундах
        """
        if self._train is None:
            raise RuntimeError("Сначала вызовите fit() для обучения модели.")

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
        Возвращает массив прогнозов длины MAX_H для стартовой точки i.

        Состояние (l_i, b_i) получается инкрементальным обновлением от
        состояния (l_{i-1}, b_{i-1}) одним шагом рекуррентной формулы.
        Прогноз: forecast[h-1] = l_i + h · b_i, h = 1..MAX_H.
        """
        if i in self._forecast_cache:
            return self._forecast_cache[i], 0.0

        t_start = time.perf_counter()

        # Получаем (или строим) состояние после ассимиляции test[0..i-1]
        l, b = self._state_at(test, i)

        # Прогноз на MAX_H шагов вперёд от момента i:
        # forecast[h-1] = l + h * b, для h = 1..MAX_H
        steps        = np.arange(1, MAX_H + 1, dtype=float)
        forecast_arr = l + steps * b

        elapsed = time.perf_counter() - t_start

        self._forecast_cache[i]   = forecast_arr
        self._refit_time_cache[i] = elapsed

        return forecast_arr, elapsed

    def _state_at(self, test: np.ndarray, i: int) -> tuple[float, float]:
        """
        Возвращает состояние модели (l, b) после ассимиляции test[0..i-1].

        Использует инкрементальное обновление: если состояние при i-1
        уже посчитано, для перехода к i выполняется один шаг рекуррентных
        формул Хольта. Иначе восстанавливает состояние с ближайшей
        известной точки.
        """
        if i in self._state_cache:
            return self._state_cache[i]

        # Найти ближайшую меньшую кэшированную позицию
        j = i - 1
        while j > 0 and j not in self._state_cache:
            j -= 1
        l, b = self._state_cache[j]

        # Прокрутить рекуррентные формулы от j до i (ассимилируя test[j..i-1])
        a, beta = self._alpha, self._beta
        for k in range(j, i):
            y_k   = float(test[k])
            l_new = a * y_k + (1.0 - a) * (l + b)
            b_new = beta * (l_new - l) + (1.0 - beta) * b
            l, b  = l_new, b_new
            self._state_cache[k + 1] = (l, b)

        return l, b

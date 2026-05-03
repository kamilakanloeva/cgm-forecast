import time

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.config import TEST_LIMIT
from src.models.base import BaseModel


class HWModel(BaseModel):
    """
    Модель Хольта (двухпараметрическое экспоненциальное сглаживание).

    Учитывает уровень и тренд без сезонной компоненты.
    Сезонность исключена ввиду высокой вычислительной стоимости
    переобучения модели Хольта-Уинтерса с периодом m=288 на каждом
    шаге скользящего окна — это делало бы эксперимент вычислительно
    неприемлемым. Модель Хольта сохраняет представление класса
    экспоненциального сглаживания в сравнительном анализе.
    """

    def __init__(self) -> None:
        # Обучающий массив — сохраняется в fit для расширения истории в predict
        self._train: np.ndarray | None = None

    def fit(self, train: np.ndarray) -> None:
        """
        Сохраняет обучающий массив.

        Параметры:
            train : обучающий массив наблюдений
        """
        self._train = train
        print(f"[HW] Обучающий массив сохранён ({len(train)} точек).")

    def predict(
        self,
        test: np.ndarray,
        h: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Строит прогнозы для каждой тестовой точки на горизонте h.

        Тест ограничивается TEST_LIMIT точками для всех моделей одинаково.
        Для каждой точки i история расширяется на одну точку и модель
        Хольта переобучается, затем прогнозирует ровно h шагов вперёд.

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

            try:
                # Модель Хольта: уровень + тренд, без сезонности
                model = ExponentialSmoothing(
                    history,
                    trend="add",
                    seasonal=None,
                    initialization_method="estimated",
                ).fit(optimized=True)
                forecast  = model.forecast(steps=h)
                y_pred[i] = float(forecast[-1])
            except Exception:
                # Если модель не сошлась — берём последнее известное значение
                y_pred[i] = float(history[-1])

            times[i] = time.perf_counter() - t_start

            y_true[i] = test[i + h - 1]

        return y_pred, y_true, float(np.mean(times))

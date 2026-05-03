from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    """
    Абстрактный базовый класс для всех моделей прогнозирования.

    Определяет единый интерфейс: fit + predict.
    Каждая модель обязана реализовать оба метода.
    Это позволяет в run_experiment.py обращаться со всеми
    моделями одинаково, не зная деталей их реализации.
    """

    @abstractmethod
    def fit(self, train: np.ndarray) -> None:
        """
        Обучает модель на обучающем массиве.

        Параметры:
            train : одномерный массив обучающих наблюдений
        """

    @abstractmethod
    def predict(
        self,
        test: np.ndarray,
        h: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Строит прогнозы на тестовом массиве для горизонта h.

        Параметры:
            test : одномерный массив тестовых наблюдений
            h    : горизонт прогнозирования в шагах

        Возвращает:
            y_pred         : массив прогнозов
            y_true         : массив истинных значений
            inference_time : среднее время одного прогноза в секундах
        """

import time

import numpy as np
from xgboost import XGBRegressor

from src.config import SEED, TEST_LIMIT, WINDOW_K, XGB_DEPTH, XGB_LR, XGB_TREES
from src.features import make_lag_matrix, make_train_windows
from src.models.base import BaseModel


class XGBModel(BaseModel):
    """
    Модель градиентного бустинга XGBoost для прогнозирования глюкозы.

    Входные признаки — лаговая матрица длины k=12 (последние 60 минут).
    Глубина деревьев и их число фиксированы для всех горизонтов —
    чтобы рост ошибки отражал только увеличение h, а не изменение
    устройства модели.

    Для каждого горизонта h обучается отдельная модель,
    так как цель y зависит от h.
    """

    def __init__(self) -> None:
        # Обучающий массив — сохраняется в fit, используется в predict
        # для формирования лаговой матрицы тестовой выборки
        self._train: np.ndarray | None = None

        # Словарь обученных моделей: ключ — горизонт h, значение — XGBRegressor
        # Модели обучаются лениво в predict при первом обращении к горизонту
        self._models: dict[int, XGBRegressor] = {}

    def fit(self, train: np.ndarray) -> None:
        """
        Сохраняет обучающий массив для последующего использования в predict.

        Модели для каждого горизонта обучаются лениво в predict,
        так как лаговая матрица зависит от h.

        Параметры:
            train : обучающий массив наблюдений
        """
        self._train = train
        print(f"[XGB] Обучающий массив сохранён. "
              f"Модели будут обучены при первом вызове predict(h).")

    def predict(
        self,
        test: np.ndarray,
        h: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Обучает XGBoost для горизонта h и строит прогнозы на test.

        Если модель для горизонта h уже обучена — использует её повторно.
        Прогнозы строятся батчем (все тестовые точки сразу).

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

        # Обучаем модель для горизонта h если ещё не обучена
        if h not in self._models:
            self._models[h] = self._fit_for_horizon(h)

        model = self._models[h]

        # Формируем лаговую матрицу для тестовой выборки
        _, _, X_test, y_true = make_lag_matrix(self._train, test, h=h)

        # Прогнозируем батчем и замеряем время инференса
        t_start = time.perf_counter()
        y_pred  = model.predict(X_test)
        t_end   = time.perf_counter()

        # Среднее время на один прогноз
        inference_time = (t_end - t_start) / len(X_test)

        return y_pred, y_true, inference_time

    def _fit_for_horizon(self, h: int) -> XGBRegressor:
        """
        Формирует лаговую матрицу и обучает XGBRegressor для горизонта h.

        Параметры:
            h : горизонт прогнозирования в шагах

        Возвращает:
            обученный XGBRegressor
        """
        X_train, y_train = make_train_windows(
            self._train,
            h=h,
        )

        model = XGBRegressor(
            n_estimators=XGB_TREES,
            max_depth=XGB_DEPTH,
            learning_rate=XGB_LR,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            verbosity=0,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        print(f"[XGB] Горизонт h={h}: модель обучена на {len(X_train)} объектах.")
        return model

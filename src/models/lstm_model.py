import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    TEST_LIMIT,
    LSTM_BATCH,
    LSTM_EPOCHS,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    LSTM_LR,
    SEED,
    WINDOW_K,
)
from src.features import make_lag_matrix, make_train_windows
from src.models.base import BaseModel

# Фиксируем зерно для воспроизводимости инициализации весов
torch.manual_seed(SEED)


class _LSTMNetwork(nn.Module):
    """
    Архитектура нейросети: один рекуррентный слой LSTM
    и один полносвязный слой на выходе.

    Входная последовательность: k измерений глюкозы.
    Выход: одно число — прогноз через h шагов.
    """

    def __init__(self) -> None:
        super().__init__()

        # Рекуррентный слой LSTM
        # input_size=1 — на каждом шаге подаётся одно значение глюкозы
        # batch_first=True — размерность входа (batch, seq, features)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            batch_first=True,
        )

        # Полносвязный слой: из скрытого состояния получаем один прогноз
        self.fc = nn.Linear(LSTM_HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через сеть.

        Параметры:
            x : тензор формы (batch, seq, 1)

        Возвращает:
            тензор формы (batch,) — прогнозы
        """
        # out: (batch, seq, hidden) — берём только последний шаг последовательности
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class LSTMModel(BaseModel):
    """
    Модель LSTM для прогнозирования уровня глюкозы.

    Входные признаки — лаговая матрица длины k=12 (последние 60 минут).
    Перед обучением признаки нормализуются: вычитается среднее и делится
    на стандартное отклонение обучающей выборки. Параметры нормализации
    вычисляются только на train и применяются к test — чтобы избежать
    утечки информации из тестовой выборки.

    Для каждого горизонта h обучается отдельная модель,
    так как целевое значение y зависит от h.
    """

    def __init__(self) -> None:
        # Обучающий массив — сохраняется в fit, используется в predict
        self._train: np.ndarray | None = None

        # Словарь обученных сетей: ключ — горизонт h, значение — _LSTMNetwork
        self._models: dict[int, _LSTMNetwork] = {}

        # Параметры нормализации — вычисляются на train в fit
        self._mu: float = 0.0
        self._sigma: float = 1.0

    def fit(self, train: np.ndarray) -> None:
        """
        Сохраняет обучающий массив и вычисляет параметры нормализации.

        Нейросети для каждого горизонта обучаются лениво в predict,
        так как лаговая матрица зависит от h.

        Параметры:
            train : обучающий массив наблюдений
        """
        self._train = train

        # Параметры нормализации вычисляются только на train
        self._mu    = float(np.mean(train))
        self._sigma = float(np.std(train)) + 1e-8  # +1e-8 защита от деления на ноль

        print(f"[LSTM] Параметры нормализации: mu={self._mu:.2f}, sigma={self._sigma:.2f}")
        print(f"[LSTM] Модели будут обучены при первом вызове predict(h).")

    def predict(
        self,
        test: np.ndarray,
        h: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Обучает LSTM для горизонта h и строит прогнозы на test.

        Если модель для горизонта h уже обучена — использует её повторно.
        Прогнозы строятся батчем (все тестовые точки сразу).

        Параметры:
            test : тестовый массив наблюдений
            h    : горизонт прогнозирования в шагах

        Возвращает:
            y_pred         : массив прогнозов (в исходных единицах mg/dL)
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

        # Нормализуем тестовые признаки параметрами из train
        X_test_n = (X_test - self._mu) / self._sigma

        # Прогнозируем батчем и замеряем время инференса
        x_tensor = self._to_tensor(X_test_n)
        model.eval()
        with torch.no_grad():
            t_start   = time.perf_counter()
            preds_n   = model(x_tensor).numpy()
            t_end     = time.perf_counter()

        # Денормализуем прогнозы обратно в mg/dL
        y_pred = preds_n * self._sigma + self._mu

        inference_time = (t_end - t_start) / len(X_test)

        return y_pred, y_true, inference_time

    def _fit_for_horizon(self, h: int) -> _LSTMNetwork:
        """
        Формирует лаговую матрицу, нормализует и обучает сеть для горизонта h.

        Функция потерь — MSE, оптимизатор — Adam.

        Параметры:
            h : горизонт прогнозирования в шагах

        Возвращает:
            обученная _LSTMNetwork
        """
        X_train, y_train = make_train_windows(
            self._train,
            h=h,
        )

        # Нормализация признаков и целей параметрами из train
        X_train_n = (X_train - self._mu) / self._sigma
        y_train_n = (y_train - self._mu) / self._sigma

        # Формируем датасет и загрузчик батчей
        dataset = TensorDataset(
            self._to_tensor(X_train_n),
            torch.tensor(y_train_n, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=LSTM_BATCH, shuffle=True)

        network   = _LSTMNetwork()
        optimizer = torch.optim.Adam(network.parameters(), lr=LSTM_LR)
        loss_fn   = nn.MSELoss()

        network.train()
        for epoch in range(LSTM_EPOCHS):
            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = loss_fn(network(x_batch), y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Выводим прогресс каждые 10 эпох
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(loader)
                print(f"[LSTM] h={h}, эпоха {epoch + 1}/{LSTM_EPOCHS}, "
                      f"loss={avg_loss:.4f}")

        print(f"[LSTM] Горизонт h={h}: модель обучена на {len(X_train)} объектах.")
        return network

    @staticmethod
    def _to_tensor(X: np.ndarray) -> torch.Tensor:
        """
        Конвертирует numpy массив в тензор PyTorch формы (batch, seq, 1).

        LSTM ожидает трёхмерный вход: батч, длина последовательности, размер признака.
        У нас один признак на каждый шаг — поэтому последнее измерение равно 1.
        """
        return torch.tensor(X, dtype=torch.float32).unsqueeze(-1)

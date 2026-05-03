from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from src.config import HORIZONS, RESULTS_DIR

# Минуты соответствующие горизонтам в шагах
MINUTES = [h * 5 for h in HORIZONS]

# Клинический ориентир — 30 минут (h=6)
CLINICAL_HORIZON_MIN = 30

# Стили линий и маркеров для академического чёрно-белого стиля.
# Каждая модель получает уникальную комбинацию — различимы при печати.
MODEL_STYLES = {
    "ARIMA": {
        "linestyle": "-",
        "marker":    "o",
        "color":     "black",
    },
    "Хольт": {
        "linestyle": "--",
        "marker":    "s",
        "color":     "black",
    },
    "XGBoost": {
        "linestyle": "-.",
        "marker":    "^",
        "color":     "black",
    },
    "LSTM": {
        "linestyle": ":",
        "marker":    "D",
        "color":     "black",
    },
}


def _setup_axes(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """
    Применяет единое академическое оформление к осям графика.

    Параметры:
        ax     : объект осей matplotlib
        title  : заголовок графика
        xlabel : подпись оси X
        ylabel : подпись оси Y
    """
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(MINUTES)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _add_clinical_line(ax: plt.Axes) -> None:
    """
    Добавляет вертикальную линию клинического ориентира 30 минут.

    Параметры:
        ax : объект осей matplotlib
    """
    ax.axvline(
        CLINICAL_HORIZON_MIN,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label="30 мин (ориентир)",
    )


def _save(fig: plt.Figure, filename: str) -> None:
    """
    Сохраняет график в папку results/ и закрывает фигуру.

    Параметры:
        fig      : объект фигуры matplotlib
        filename : имя файла без пути
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Сохранён: {path}")


def plot_mae(results: dict) -> None:
    """
    Строит график MAE(h) для всех моделей.

    По оси X — горизонт в минутах, по оси Y — MAE в mg/dL.
    Вертикальная линия отмечает клинический ориентир 30 минут.

    Параметры:
        results : словарь результатов {имя_модели: {"mae": [...], ...}}
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name, data in results.items():
        style = MODEL_STYLES[model_name]
        ax.plot(
            MINUTES,
            data["mae"],
            label=model_name,
            **style,
            linewidth=1.8,
            markersize=6,
        )

    _add_clinical_line(ax)
    _setup_axes(
        ax,
        title="Зависимость MAE от горизонта прогнозирования",
        xlabel="Горизонт, мин",
        ylabel="MAE, mg/dL",
    )
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, "fig1_mae.png")


def plot_rmse(results: dict) -> None:
    """
    Строит график RMSE(h) для всех моделей.

    Параметры:
        results : словарь результатов {имя_модели: {"rmse": [...], ...}}
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name, data in results.items():
        style = MODEL_STYLES[model_name]
        ax.plot(
            MINUTES,
            data["rmse"],
            label=model_name,
            **style,
            linewidth=1.8,
            markersize=6,
        )

    _add_clinical_line(ax)
    _setup_axes(
        ax,
        title="Зависимость RMSE от горизонта прогнозирования",
        xlabel="Горизонт, мин",
        ylabel="RMSE, mg/dL",
    )
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, "fig2_rmse.png")


def plot_delta_mae(results: dict) -> None:
    """
    Строит график ΔMAE(h) — прироста ошибки при увеличении горизонта.

    ΔMAE(h) = MAE(h) - MAE(h-1). Для h=1 значение не определено (NaN).
    Вертикальная линия отмечает клинический ориентир 30 минут.

    Параметры:
        results : словарь результатов {имя_модели: {"mae": [...], ...}}
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name, data in results.items():
        mae_list = data["mae"]

        # ΔMAE(h=1) не определён — ставим NaN чтобы линия начиналась с h=2
        deltas = [np.nan] + [
            mae_list[i] - mae_list[i - 1] for i in range(1, len(mae_list))
        ]
        style = MODEL_STYLES[model_name]
        ax.plot(
            MINUTES,
            deltas,
            label=model_name,
            **style,
            linewidth=1.8,
            markersize=6,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    _add_clinical_line(ax)
    _setup_axes(
        ax,
        title="Прирост MAE при увеличении горизонта (ΔMAE)",
        xlabel="Горизонт, мин",
        ylabel="ΔMAE, mg/dL",
    )
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, "fig3_delta_mae.png")


def plot_inference_time(results: dict) -> None:
    """
    Строит график среднего времени инференса одного прогноза.

    Ось Y в логарифмическом масштабе — модели различаются
    на несколько порядков по скорости.

    Параметры:
        results : словарь результатов {имя_модели: {"inf_time": [...], ...}}
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name, data in results.items():
        # Переводим секунды в миллисекунды для удобства чтения
        times_ms = [t * 1000 for t in data["inf_time"]]
        style = MODEL_STYLES[model_name]
        ax.plot(
            MINUTES,
            times_ms,
            label=model_name,
            **style,
            linewidth=1.8,
            markersize=6,
        )

    ax.set_yscale("log")
    _setup_axes(
        ax,
        title="Среднее время инференса одного прогноза",
        xlabel="Горизонт, мин",
        ylabel="Время, мс (лог. шкала)",
    )
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, "fig4_inference.png")


def plot_heatmap(results: dict) -> None:
    """
    Строит тепловую карту MAE: строки — модели, столбцы — горизонты.

    Позволяет одним взглядом сравнить все модели на всех горизонтах.

    Параметры:
        results : словарь результатов {имя_модели: {"mae": [...], ...}}
    """
    model_names = list(results.keys())
    mae_matrix  = np.array([results[m]["mae"] for m in model_names])

    fig, ax = plt.subplots(figsize=(12, 4))

    im = ax.imshow(mae_matrix, aspect="auto", cmap="Greys")

    # Подписи осей
    ax.set_xticks(range(len(MINUTES)))
    ax.set_xticklabels([f"{m}" for m in MINUTES], fontsize=9)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=10)
    ax.set_xlabel("Горизонт, мин", fontsize=11)
    ax.set_title(
        "Тепловая карта MAE по моделям и горизонтам (mg/dL)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )

    # Значения MAE внутри ячеек
    for i in range(len(model_names)):
        for j in range(len(MINUTES)):
            ax.text(
                j, i,
                f"{mae_matrix[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if mae_matrix[i, j] > mae_matrix.mean() else "black",
            )

    plt.colorbar(im, ax=ax, label="MAE, mg/dL")
    fig.tight_layout()
    _save(fig, "fig5_heatmap.png")


def save_all(results: dict) -> None:
    """
    Строит и сохраняет все пять графиков.

    Вызывается из run_experiment.py после завершения эксперимента.

    Параметры:
        results : полный словарь результатов эксперимента
    """
    print("\n[plots] Генерация графиков...")
    plot_mae(results)
    plot_rmse(results)
    plot_delta_mae(results)
    plot_inference_time(results)
    plot_heatmap(results)
    print("[plots] Все графики сохранены.")

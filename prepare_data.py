"""
prepare_data.py
===============
Извлечение CGM-ряда пациента 588 из датасета OhioT1DM (XML-формат)
и сохранение в CSV без каких-либо преобразований значений.

Принципы:
  - Данные сохраняются как есть: целочисленные mg/dL, оригинальные метки.
  - Никакая интерполяция, сглаживание или нормировка не применяются.
  - Разрывы сохраняются и явно документируются.
  - Все параметры качества выводятся на экран для §2.1 и §2.2.

Использование:
  python prepare_data.py
  python prepare_data.py --xml путь/к/файлу.xml --out glucose_588.csv
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Добавляем корень проекта в sys.path, чтобы импортировать src/config.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import DATA_PATH, ROOT_DIR


# ─────────────────────────────────────────────────────────────────────────────
# 1. Парсинг XML
# ─────────────────────────────────────────────────────────────────────────────

def load_glucose_from_xml(xml_path: str) -> pd.DataFrame:
    """
    Извлекает блок <glucose_level> из XML OhioT1DM.
    Возвращает DataFrame: timestamp (datetime), glucose (float).
    Никаких преобразований значений не выполняется.
    """
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {xml_path}")

    tree = ET.parse(str(path))
    root = tree.getroot()

    glucose_block = root.find("glucose_level")
    if glucose_block is None:
        raise ValueError(
            "Блок <glucose_level> не найден. "
            "Проверьте, что передан XML из датасета OhioT1DM."
        )

    records = []
    skipped = 0

    for event in glucose_block.findall("event"):
        ts_str  = event.attrib.get("ts")
        val_str = event.attrib.get("value")

        if ts_str is None or val_str is None:
            skipped += 1
            continue

        try:
            timestamp = datetime.strptime(ts_str, "%d-%m-%Y %H:%M:%S")
            glucose   = float(val_str)
        except (ValueError, TypeError):
            skipped += 1
            continue

        records.append({"timestamp": timestamp, "glucose": glucose})

    if not records:
        raise ValueError("Не удалось извлечь ни одной записи глюкозы.")

    df = (
        pd.DataFrame(records)
          .sort_values("timestamp")
          .reset_index(drop=True)
    )

    print(f"  Извлечено записей : {len(df)}")
    print(f"  Пропущено записей : {skipped}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Проверка качества ряда
# ─────────────────────────────────────────────────────────────────────────────

def check_quality(df: pd.DataFrame) -> dict:
    """
    Вычисляет и печатает характеристики качества ряда.
    Возвращает словарь для использования в итоговой сводке.
    """
    diffs = df["timestamp"].diff().dropna().dt.total_seconds() / 60.0

    stats = {
        "n_points"         : len(df),
        "start"            : df["timestamp"].min(),
        "end"              : df["timestamp"].max(),
        "mean_step_min"    : round(float(diffs.mean()), 4),
        "frac_not_5"       : round(float((diffs != 5).sum() / len(diffs)), 6),
        "gaps_gt_10"       : int((diffs > 10).sum()),
        "gaps_gt_30"       : int((diffs > 30).sum()),
        "glucose_min"      : float(df["glucose"].min()),
        "glucose_max"      : float(df["glucose"].max()),
        "glucose_mean"     : round(float(df["glucose"].mean()), 4),
        "glucose_std"      : round(float(df["glucose"].std()), 4),
        "is_integer_valued": bool((df["glucose"] == df["glucose"].round(0)).all()),
    }

    sep = "=" * 54
    print(f"\n{sep}")
    print("  КАЧЕСТВО ВРЕМЕННОГО РЯДА (CGM)")
    print(sep)
    print(f"  Число точек        : {stats['n_points']}")
    print(f"  Начало ряда        : {stats['start']}")
    print(f"  Конец ряда         : {stats['end']}")
    print(f"  Средний шаг        : {stats['mean_step_min']} мин")
    print(f"  Доля шагов != 5 мин: {stats['frac_not_5']:.2%}")
    print(f"  Разрывы > 10 мин   : {stats['gaps_gt_10']}")
    print(f"  Разрывы > 30 мин   : {stats['gaps_gt_30']}")
    print(f"  Мин. глюкоза       : {stats['glucose_min']:.0f} mg/dL")
    print(f"  Макс. глюкоза      : {stats['glucose_max']:.0f} mg/dL")
    print(f"  Среднее +- std     : {stats['glucose_mean']} +- {stats['glucose_std']} mg/dL")
    print(f"  Целые значения     : {stats['is_integer_valued']}")

    print()
    print("  Распределение по клиническим диапазонам:")
    ranges = [
        ("< 54   (тяжёлая гипо) ", df["glucose"] < 54),
        ("54-69  (гипогликемия) ", (df["glucose"] >= 54)  & (df["glucose"] < 70)),
        ("70-99  (норма низкая) ", (df["glucose"] >= 70)  & (df["glucose"] < 100)),
        ("100-139 (норма)       ", (df["glucose"] >= 100) & (df["glucose"] < 140)),
        ("140-179 (выше нормы)  ", (df["glucose"] >= 140) & (df["glucose"] < 180)),
        ("180-249 (гипергл.)    ", (df["glucose"] >= 180) & (df["glucose"] < 250)),
        (">= 250 (выс. гипергл.)", df["glucose"] >= 250),
    ]
    for label, mask in ranges:
        cnt = int(mask.sum())
        pct = cnt / len(df) * 100
        print(f"    {label}: {cnt:5d}  ({pct:.1f}%)")

    print(sep)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 3. Детализация разрывов
# ─────────────────────────────────────────────────────────────────────────────

def describe_gaps(df: pd.DataFrame, threshold_min: float = 10.0) -> pd.DataFrame:
    """
    Возвращает таблицу разрывов длиннее threshold_min минут.
    Каждая строка: gap_start, gap_end, duration_min, missing_steps.
    """
    diffs = df["timestamp"].diff().dt.total_seconds() / 60.0
    mask  = diffs > threshold_min

    gaps = []
    for idx in df.index[mask]:
        gap_start    = df.loc[idx - 1, "timestamp"]
        gap_end      = df.loc[idx,     "timestamp"]
        duration_min = float(diffs.loc[idx])
        missing      = int(duration_min // 5) - 1
        gaps.append({
            "gap_start"    : gap_start,
            "gap_end"      : gap_end,
            "duration_min" : duration_min,
            "missing_steps": missing,
        })

    if not gaps:
        print(f"\n  Разрывов > {threshold_min:.0f} мин не обнаружено.")
        return pd.DataFrame()

    gaps_df = pd.DataFrame(gaps)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(f"\n  Разрывы > {threshold_min:.0f} мин ({len(gaps_df)} шт.):")
    print(gaps_df.to_string(index=False))
    return gaps_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Сохранение CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(df: pd.DataFrame, out_path: str) -> None:
    """Сохраняет DataFrame в CSV без индекса."""
    df.to_csv(out_path, index=False)
    print(f"  Файл записан      : {out_path}")
    print(f"  Записей сохранено : {len(df)}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Верификация CSV
# ─────────────────────────────────────────────────────────────────────────────

def verify_csv(csv_path: str, original_df: pd.DataFrame) -> bool:
    """
    Читает CSV обратно и сверяет с оригинальным DataFrame.
    Проверяет: число строк, временные метки, значения глюкозы.
    """
    loaded = pd.read_csv(csv_path, parse_dates=["timestamp"])
    ok = True

    if len(loaded) != len(original_df):
        print(f"  [!] Число строк: CSV={len(loaded)}, оригинал={len(original_df)}")
        ok = False

    ts_match = (loaded["timestamp"].values == original_df["timestamp"].values).all()
    if not ts_match:
        n_diff = (loaded["timestamp"].values != original_df["timestamp"].values).sum()
        print(f"  [!] Временные метки расходятся в {n_diff} позициях")
        ok = False

    max_diff = float(
        np.abs(loaded["glucose"].values - original_df["glucose"].values).max()
    )
    if max_diff > 1e-6:
        print(f"  [!] Расхождение значений глюкозы: макс. {max_diff:.2e}")
        ok = False

    if ok:
        print("  [OK] CSV верифицирован: полностью совпадает с источником")

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Главная функция
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Извлечение CGM-ряда из XML OhioT1DM в CSV без преобразований"
    )
    parser.add_argument(
        "--xml",
        default=str(DATA_PATH),
        help="Путь к исходному XML-файлу (по умолчанию: data/588-ws-training.xml)"
    )
    parser.add_argument(
        "--out",
        default=str(ROOT_DIR / "data" / "glucose_588.csv"),
        help="Путь к выходному CSV-файлу (по умолчанию: data/glucose_588.csv)"
    )
    parser.add_argument(
        "--gaps-threshold",
        type=float,
        default=10.0,
        help="Порог для детализации разрывов в минутах (по умолчанию: 10)"
    )
    args = parser.parse_args()

    # Шаг 1
    print("\n--- Шаг 1: Парсинг XML -------------------------------------------")
    df = load_glucose_from_xml(args.xml)

    # Шаг 2
    print("\n--- Шаг 2: Проверка качества ряда --------------------------------")
    stats = check_quality(df)

    # Шаг 3
    print("\n--- Шаг 3: Детализация разрывов -----------------------------------")
    describe_gaps(df, threshold_min=args.gaps_threshold)

    # Шаг 4
    print("\n--- Шаг 4: Сохранение CSV -----------------------------------------")
    save_csv(df, args.out)

    # Шаг 5
    print("\n--- Шаг 5: Верификация CSV ----------------------------------------")
    ok = verify_csv(args.out, df)

    if not ok:
        print("\n  ОШИБКА: CSV не прошёл верификацию.")
        sys.exit(1)

    # Итоговая сводка
    print("\n--- Итоговая сводка (для параграфов 2.1-2.2) --------------------")
    print(f"  Файл источника     : {args.xml}")
    print(f"  Файл результата    : {args.out}")
    print(f"  Число точек        : {stats['n_points']}")
    print(f"  Период             : {stats['start']} -- {stats['end']}")
    print(f"  Диапазон глюкозы   : {stats['glucose_min']:.0f}--"
          f"{stats['glucose_max']:.0f} mg/dL")
    print(f"  Средний шаг        : {stats['mean_step_min']} мин")
    print(f"  Разрывов > 30 мин  : {stats['gaps_gt_30']}")
    print(f"  Нормировка         : не применялась")
    print(f"  Интерполяция       : не применялась")
    print()


if __name__ == "__main__":
    main()
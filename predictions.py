# -*- coding: utf-8 -*-

import csv
import math
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# 1. Чтение данных из CSV
# ------------------------------------------------------------
def read_csv(path: Path):
    """
    Возвращает словарь, где ключ – имя столбца,
    а значение – массив float с данными этого столбца.
    """
    with path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        # Инициализируем списки для всех столбцов
        columns = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name, value in row.items():
                # Приводим к float, пустые ячейки заменяем на 0.0
                try:
                    columns[name].append(float(value.replace(',', '.')))
                except (ValueError, AttributeError):
                    columns[name].append(0.0)
    # Преобразуем списки в numpy‑массивы
    return {k: np.array(v, dtype=float) for k, v in columns.items()}


# ------------------------------------------------------------
# 2. Функция, находящая коэффициенты a и b для y = a * x**b
# ------------------------------------------------------------
def power_law_fit(x: np.ndarray, y: np.ndarray):
    """Возвращает (a, b) для модели y = a·x**b."""
    # Убираем нулевые и отрицательные значения, которые ломают логарифм
    mask = (x > 0) & (y > 0)
    log_x = np.log(x[mask])
    log_y = np.log(y[mask])

    # Линейная регрессия: log_y = log_a + b * log_x
    b, log_a = np.polyfit(log_x, log_y, 1)
    a = np.exp(log_a)
    return a, b


# ------------------------------------------------------------
# 3. Основная часть – чтение, расчёт коэффициентов, интерактив
# ------------------------------------------------------------
def main():
    csv_path = Path("data.csv")          # <-- укажите путь к вашему файлу
    data = read_csv(csv_path)

    # Столбец, который будем использовать как «нагрузку» (x)
    x = data["log generation"]

    # Список интересующих нас зависимостей (можно добавить свои)
    targets = ["RX", "TX", "Read", "Write", "CPU load"]

    # Вычисляем коэффициенты для каждого целевого столбца
    coeff = {}
    for name in targets:
        if name not in data:
            print(f"В файле нет столбца «{name}», он будет пропущен.")
            continue
        coeff[name.lower()] = power_law_fit(x, data[name])

    # Выводим найденные коэффициенты
    print("\nНайденные коэффициенты (y = a·x**b):")
    for name, (a, b) in coeff.items():
        print(f"  {name}: a = {a:.6e}, b = {b:.6f}")
    print("-" * 50)

    # --------------------------------------------------------
    # 4. Интерактивный цикл предсказаний
    # --------------------------------------------------------
    def calc(y_name: str, x_val: float) -> float:
        a, b = coeff[y_name]
        return a * math.pow(x_val, b)

    print("\n=== Прогноз значений по нагрузке ===")
    while True:
        raw = input("\nВведите значение нагрузки (или пустую строку для выхода): ")
        if raw == "":
            break
        try:
            x_val = float(raw.replace(",", "."))
        except ValueError:
            print("Ошибка: введите корректное числовое значение.")
            continue

        print(f"\nДля нагрузки x = {x_val:.3f} получаем:")
        for name in targets:
            key = name.lower()
            if key not in coeff:
                continue
            y_val = calc(key, x_val)
            # Форматируем вывод в зависимости от типа метрики
            if "cpu" in key:
                suffix = " CPU"
            elif "read" in key or "write" in key or "rx" in key or "tx" in key:
                suffix = " MB/s"
            else:
                suffix = ""
            print(f"  {name.lower()} = {y_val:.5f}{suffix}")

    print("\nПрограмма завершена.")


if __name__ == "__main__":
    main()
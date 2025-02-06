"""
Создать CSV предобработанных файлов для данных
"""
import json
import numpy as np
import pandas as pd


def featurization():
    # Загрузка данных
    print("Загрузка датасетов...")
    train_data = pd.read_csv('./data/train_data.csv', header=None, dtype=float)
    test_data = pd.read_csv('./data/test_data.csv', header=None, dtype=float)
    print("Выполнено.")

    # Нормализация тренировочных данных
    print("Нормализация данных...")
    # Убираем первую колонку с labels
    train_mean = train_data.values[:, 1:].mean()
    train_std = train_data.values[:, 1:].std()

    #  Стандартизируем и нормализуем данные
    train_data.values[:, 1:] -= train_mean
    train_data.values[:, 1:] /= train_std
    test_data.values[:, 1:] -= train_mean
    test_data.values[:, 1:] /= train_std

    print("Выполнено.")

    print("Сохранение предобработанных датасетов и параметров...")
    # Сохранение данных
    np.save('./data/processed_train_data', train_data)
    np.save('./data/processed_test_data', test_data)

    # Сохранеям среднее и стд откл на будущее
    with open('./data/norm_params.json', 'w') as f:
        json.dump({'mean': train_mean, 'std': train_std}, f)

    print("Выполнено.")


if __name__ == '__main__':
    featurization()

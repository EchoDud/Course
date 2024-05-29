import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Функция для генерации выборок из нормального распределения
def generate_samples(n, mu=0, sigma=1):
    return np.random.normal(mu, sigma, n)

# Функция для добавления выбросов модели Тьюки
def add_tukey_outliers(samples, proportion, symmetric=True):
    n_outliers = int(proportion * len(samples))
    if symmetric:
        # Симметричные выбросы
        outliers = np.random.normal(np.mean(samples), 5 * np.std(samples), n_outliers)
    else:
        # Асимметричные выбросы
        n_high = n_outliers // 2
        n_low = n_outliers - n_high
        outliers_high = np.random.normal(np.mean(samples) + 3 * np.std(samples), np.std(samples), n_high)
        outliers_low = np.random.normal(np.mean(samples) - 3 * np.std(samples), np.std(samples), n_low)
        outliers = np.concatenate((outliers_high, outliers_low))
    return np.concatenate((samples, outliers))

# Ввод параметров
n_sample = int(input("Введите размер выборки: "))
n_bootstrap = int(input("Введите количество бутстреп-ревыборок: "))
mu = float(input("Введите среднее значение (mu) распределения: "))
sigma = float(input("Введите стандартное отклонение (sigma) распределения: "))
add_outliers_flag = input("Добавить выбросы? (да/нет): ").strip().lower() == 'да'
if add_outliers_flag:
    proportion_outliers = float(input("Введите долю выбросов (например, 0.1 для 10%): "))
    symmetric_outliers = input("Симметричные выбросы? (да/нет): ").strip().lower() == 'да'

# Генерация выборок
sample = generate_samples(n_sample, mu, sigma)

# Добавление выбросов, если выбрано
if add_outliers_flag:
    sample = add_tukey_outliers(sample, proportion_outliers, symmetric_outliers)

# Функция для вычисления оценки Джини
def gini_index(samples):
    n = len(samples)
    sum_abs_diff = np.sum(np.abs(samples[:, None] - samples[None, :]))
    return sum_abs_diff / (n * (n - 1))

# Вычисление оценки Джини
gini = gini_index(sample)

print(f"Оценка Джини для выборки размером {n_sample}: {gini}")

# Функция для вычисления оптимальных комплексных оценок на порядковых статистиках
def optimal_complex_estimate(samples):
    n = len(samples)
    sorted_samples = np.sort(samples)
    k1 = int(0.0668 * n)
    k2 = int(0.2912 * n)
    k3 = int(0.7088 * n)
    k4 = int(0.9332 * n)
    sigma = 0.2581 * (sorted_samples[k4] - sorted_samples[k1]) + 0.2051 * (sorted_samples[k3] - sorted_samples[k2])
    return sigma

# Вычисление оптимальных комплексных оценок
optimal = optimal_complex_estimate(sample)

print(f"Оптимальная комплексная оценка для выборки размером {n_sample}: {optimal}")

# Функция для оценки с использованием бутстреп-метода
def bootstrap_estimates(samples, estimate_func, n_iterations=1000):
    bootstrap_samples = np.random.choice(samples, (n_iterations, len(samples)), replace=True)
    estimates = np.array([estimate_func(sample) for sample in bootstrap_samples])
    mean_estimate = np.mean(estimates)
    var_estimate = np.var(estimates)
    std_estimate = np.std(estimates)
    conf_interval_mean = np.percentile(estimates, [2.5, 97.5])
    conf_interval_median = np.percentile(estimates, [50 - 34.1, 50 + 34.1])
    return estimates, mean_estimate, var_estimate, std_estimate, conf_interval_mean, conf_interval_median

# Бутстреп оценки для Gini
estimates_gini, mean_gini, var_gini, std_gini, conf_gini_mean, conf_gini_median = bootstrap_estimates(sample, gini_index, n_bootstrap)

# Бутстреп оценки для Optimal Complex Estimate
estimates_opt, mean_opt, var_opt, std_opt, conf_opt_mean, conf_opt_median = bootstrap_estimates(sample, optimal_complex_estimate, n_bootstrap)

print(f"Оценка Джини - Среднее: {mean_gini}, Дисперсия: {var_gini}, СКО: {std_gini}, 95% ДИ для среднего: {conf_gini_mean}, 95% ДИ для медианы: {conf_gini_median}")
print(f"Оптимальная комплексная оценка - Среднее: {mean_opt}, Дисперсия: {var_opt}, СКО: {std_opt}, 95% ДИ для среднего: {conf_opt_mean}, 95% ДИ для медианы: {conf_opt_median}")

# Оценка плотности для бутстреп-оценок
density_gini = gaussian_kde(estimates_gini)
density_opt = gaussian_kde(estimates_opt)

# График плотности бутстреп-оценок
x = np.linspace(min(min(estimates_gini), min(estimates_opt)), max(max(estimates_gini), max(estimates_opt)), 1000)
plt.figure(figsize=(10, 6))
plt.plot(x, density_gini(x), label='Оценка Джини', color='blue')
plt.plot(x, density_opt(x), label='Оптимальная комплексная оценка', color='green')
plt.legend()
plt.title('Плотность вероятности бутстреп-оценок')
plt.show()
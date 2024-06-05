import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Функция для генерации выборки из нормального распределения
def generate_samples(n, mu=0, sigma=1):
    return np.random.normal(mu, sigma, n)

# Функция для добавления выбросов модели Тьюки
def add_tukey_outliers(samples, proportion, outlier_mu, outlier_sigma):
    n_outliers = int(proportion * len(samples))
    outliers = np.random.normal(outlier_mu, outlier_sigma, n_outliers)
    return np.concatenate((samples, outliers))

# Ввод параметров
n_sample = int(input("Введите размер выборки: "))
n_bootstrap = int(input("Введите количество бутстреп-ревыборок: "))
mu = float(input("Введите среднее значение (mu) распределения: "))
sigma = float(input("Введите стандартное отклонение (sigma) распределения: "))
add_outliers_flag = input("Добавить выбросы? (да/нет): ").strip().lower() == 'да'
if add_outliers_flag:
    proportion_outliers = float(input("Введите долю выбросов (например, 0.1 для 10%): "))
    outlier_mu = float(input("Введите среднее значение (mu) для выбросов: "))
    outlier_sigma = float(input("Введите стандартное отклонение (sigma) для выбросов: "))

# Генерация выборки
sample = generate_samples(n_sample, mu, sigma)

# Добавление выбросов, если выбрано
if add_outliers_flag:
    sample = add_tukey_outliers(sample, proportion_outliers, outlier_mu, outlier_sigma)

# Функция для вычисления индекса Джини
def gini_index(samples):
    n = len(samples)
    abs_diff = np.abs(samples[:, None] - samples[None, :])
    sum_abs_diff = np.sum(np.triu(abs_diff, 1))
    return 1.77245 * sum_abs_diff / (n * (n - 1))

# Вычисление индекса Джини
gini = gini_index(sample)

# Функция для вычисления оптимальной комплексной оценки на порядковых статистиках
def optimal_complex_estimate(samples):
    n = len(samples)
    sorted_samples = np.sort(samples)
    k1 = int(0.0668 * n)
    k2 = int(0.2912 * n)
    k3 = int(0.7088 * n)
    k4 = int(0.9332 * n)
    sigma = 0.2581 * (sorted_samples[k4] - sorted_samples[k1]) + 0.2051 * (sorted_samples[k3] - sorted_samples[k2])
    return sigma

# Вычисление оптимальной комплексной оценки
optimal = optimal_complex_estimate(sample)

# Функция для оценок с использованием бутстреп-метода
def bootstrap_estimates(samples, estimate_func, true_value, n_iterations=1000):
    bootstrap_samples = np.random.choice(samples, (n_iterations, len(samples)), replace=True)
    estimates = np.array([estimate_func(sample) for sample in bootstrap_samples])
    mean_estimate = np.mean(estimates)
    var_estimate = np.var(estimates)    
    bias = (mean_estimate - true_value)**2  # Смещение
    std_estimate = bias + var_estimate
    conf_interval_mean = np.percentile(estimates, [2.5, 97.5])
    conf_interval_median = np.percentile(estimates, [50 - 34.1, 50 + 34.1])
    return estimates, bias, var_estimate, std_estimate, conf_interval_mean, conf_interval_median



# Бутстреп оценки для Джини
estimates_gini, bias_gini, var_gini, std_gini, conf_gini_mean, conf_gini_median = bootstrap_estimates(sample, gini_index, sigma, n_bootstrap)

# Бутстреп оценки для Оптимальной Комплексной Оценки
estimates_opt, bias_opt, var_opt, std_opt, conf_opt_mean, conf_opt_median = bootstrap_estimates(sample, optimal_complex_estimate, sigma, n_bootstrap)

# Оценка плотности для бутстреп-оценок
density_gini = gaussian_kde(estimates_gini)
density_opt = gaussian_kde(estimates_opt)

print(f"Оценка Джини для выборки размером {n_sample}: {gini}")
print(f"Оптимальная комплексная оценка для выборки размером {n_sample}: {optimal}")
print(f"Оценка Джини - Квадрат смещения: {bias_gini}, Дисперсия: {var_gini}, СКО: {std_gini}")
print(f"Оптимальная комплексная оценка - Квадрат смещения: {bias_opt}, Дисперсия: {var_opt}, СКО: {std_opt}")

# График плотности бутстреп-оценок
x = np.linspace(min(min(estimates_gini), min(estimates_opt)), max(max(estimates_gini), max(estimates_opt)), 1000)
plt.figure(figsize=(10, 6))
plt.plot(x, density_gini(x), label='Оценка Джини', color='blue')
plt.plot(x, density_opt(x), label='Оптимальная комплексная оценка', color='green')
plt.legend()
plt.title('Плотность вероятности бутстреп-оценок')
plt.show()

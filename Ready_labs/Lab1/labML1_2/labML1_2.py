import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data():
    """Загрузка и подготовка данных"""
    diabetes = datasets.load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target

    # Анализ корреляции для выбора признака
    print("\nКорреляция признаков с целевой переменной:")
    print(df.corr()['target'].sort_values(ascending=False))

    # Выбор BMI как наиболее коррелированного признака
    X = df[['bmi']].values
    y = df['target'].values

    # Нормализация данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.3, random_state=42)


class CustomLinearRegression:
    """Кастомная реализация линейной регрессии"""

    def __init__(self, learning_rate: float = 0.01, n_iter: int = 1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.w = None
        self.b = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Обучение модели"""
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        for _ in range(self.n_iter):
            y_pred = np.dot(X, self.w) + self.b
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание"""
        return np.dot(X, self.w) + self.b


if __name__ == "__main__":
    #Загрузка данных
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    #Обучение Sklearn модели
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test)

    #Обучение кастомной модели
    custom_model = CustomLinearRegression(learning_rate=0.1, n_iter=2000)
    custom_model.fit(X_train, y_train)
    y_pred_custom = custom_model.predict(X_test)

    #Вычисление метрик
    mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
    r2_sklearn = r2_score(y_test, y_pred_sklearn)
    mse_custom = mean_squared_error(y_test, y_pred_custom)
    r2_custom = r2_score(y_test, y_pred_custom)

    #Вывод результатов
    print("\nРезультаты:")
    print(f"Sklearn: w={sklearn_model.coef_[0]:.4f}, b={sklearn_model.intercept_:.4f}")
    print(f"MSE: {mse_sklearn:.2f}, R²: {r2_sklearn:.2f}")
    print(f"\nCustom: w={custom_model.w[0]:.4f}, b={custom_model.b:.4f}")
    print(f"MSE: {mse_custom:.2f}, R²: {r2_custom:.2f}")

    #Визуализация только регрессий
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='black', alpha=0.8, label='Реальные данные')
    plt.plot(X_test, y_pred_sklearn, color='red', linewidth=2, label='Sklearn')
    plt.plot(X_test, y_pred_custom, color='orange', linestyle='--', label='Custom')
    plt.xlabel('Нормализованный BMI')
    plt.ylabel('Прогрессирование диабета')
    plt.title('Сравнение моделей линейной регрессии')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Таблица результатов
    results = pd.DataFrame({
        'Реальные значения': y_test,
        'Предсказание Sklearn': y_pred_sklearn,
        'Предсказание Custom': y_pred_custom,
        'Разница Sklearn': np.abs(y_test - y_pred_sklearn),
        'Разница Custom': np.abs(y_test - y_pred_custom)
    })

    print("\nПример предсказаний (первые 10 строк):")
    print(results.head(10).round(2))
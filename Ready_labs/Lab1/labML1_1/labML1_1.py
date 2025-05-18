import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


class RegressionVisualizer:
    """
    Класс для визуализации линейной регрессии и анализа ошибок

    Attributes:
        df (pd.DataFrame): Загруженные данные
        fig (plt.Figure): Основная фигура для графиков
        ax1, ax2, ax3 (plt.Axes): Оси для графиков
        model (LinearRegression): Обученная модель регрессии
    """

    def __init__(self, data_path: str):
        """Инициализация с загрузкой данных и созданием графиков"""
        self.df = self._load_data(data_path)
        self.fig, ((self.ax1, self.ax2), (self.ax3, _)) = plt.subplots(2, 2)
        self.model = LinearRegression()

    def _load_data(self, path: str) -> pd.DataFrame:
        """Загрузка и валидация данных"""
        df = pd.read_csv(path)
        if len(df.columns) < 2:
            raise ValueError("Данные должны содержать как минимум 2 столбца")
        return df

    def calculate_regression(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """
        Вычисление параметров регрессии

        Args:
            x: Признаки для обучения
            y: Целевая переменная

        Returns:
            tuple: (intercept, slope) коэффициенты регрессии
        """
        x_reshaped = x.reshape(-1, 1)
        self.model.fit(x_reshaped, y)
        return self.model.intercept_, self.model.coef_[0]

    def evaluate_model(self, x: np.ndarray, y: np.ndarray) -> dict:
        """
        Оценка качества модели

        Args:
            x: Признаки для оценки
            y: Истинные значения

        Returns:
            dict: Метрики качества (MSE, R^2)
        """
        y_pred = self.model.predict(x.reshape(-1, 1))
        return {
            'mse': mean_squared_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }

    def plot_initial_data(self, x: np.ndarray, y: np.ndarray, labels: tuple[str, str]) -> None:
        """Визуализация исходных данных"""
        self.ax1.scatter(x, y, color='blue')
        self.ax1.set_title(f"Исходные данные: {labels[0]} vs {labels[1]}")
        self.ax1.set_xlabel(labels[0])
        self.ax1.set_ylabel(labels[1])

    def plot_regression_line(self, x: np.ndarray, y: np.ndarray, labels: tuple[str, str]) -> None:
        """Визуализация линии регрессии"""
        self.ax2.scatter(x, y, color='blue')
        self.ax2.set_xlabel(labels[0])
        self.ax2.set_ylabel(labels[1])

        # Генерация точек для линии регрессии
        x_line = np.linspace(min(x), max(x), 100)
        y_line = self.model.predict(x_line.reshape(-1, 1))
        self.ax2.plot(x_line, y_line, color='red')

    def plot_error_squares(self, x: np.ndarray, y: np.ndarray, labels: tuple[str, str]) -> None:
        """Визуализация квадратов ошибок"""
        self.ax3.scatter(x, y, color='blue')
        self.ax3.set_xlabel(labels[0])
        self.ax3.set_ylabel(labels[1])

        y_pred = self.model.predict(x.reshape(-1, 1))
        x_line = np.linspace(min(x), max(x), 100)
        y_line = self.model.predict(x_line.reshape(-1, 1))
        self.ax3.plot(x_line, y_line, color='red')

        # Масштабирование квадратов ошибок
        bbox = self.ax3.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        scale_factor = (self.ax3.get_xlim()[1] - self.ax3.get_xlim()[0]) / \
                       (self.ax3.get_ylim()[1] - self.ax3.get_ylim()[0]) / \
                       (bbox.width / bbox.height)

        for i in range(len(x)):
            error = y_pred[i] - y[i]
            if error > 0:
                rect = patches.Rectangle((x[i], y[i]), -abs(error) * scale_factor, abs(error),
                                         color='green', alpha=0.4)
            else:
                rect = patches.Rectangle((x[i], y_pred[i]), abs(error) * scale_factor, abs(error),
                                         color='green', alpha=0.4)
            self.ax3.add_patch(rect)


if __name__ == "__main__":
    try:
        # Инициализация визуализатора
        visualizer = RegressionVisualizer('student_scores.csv')

        print("Статистика данных:")
        print(visualizer.df.describe())

        print("\nВыберите вариант отображения:")
        columns = visualizer.df.columns
        variants = [(columns[0], columns[1]), (columns[1], columns[0])]
        for i, (x_col, y_col) in enumerate(variants):
            print(f"{i + 1}. X: {x_col}, Y: {y_col}")

        choice = int(input("Вариант:")) - 1
        x_col, y_col = variants[choice]
        x, y = visualizer.df[x_col].values, visualizer.df[y_col].values

        # Обучение модели и визуализация
        w0, w1 = visualizer.calculate_regression(x, y)
        metrics = visualizer.evaluate_model(x, y)

        print(f"\nРезультаты регрессии:")
        print(f"Уравнение: y = {w1:.2f}x + {w0:.2f}")
        print(f"MSE: {metrics['mse']:.2f}")
        print(f"R²: {metrics['r2']:.2f}")

        visualizer.plot_initial_data(x, y, (x_col, y_col))
        visualizer.plot_regression_line(x, y, (x_col, y_col))
        visualizer.plot_error_squares(x, y, (x_col, y_col))

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Ошибка: {str(e)}")
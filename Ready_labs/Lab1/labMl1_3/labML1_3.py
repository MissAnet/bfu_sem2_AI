"""
Implementation of Linear Regression with custom algorithm and scikit-learn
Includes calculation of MAE, R2, and MAPE metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score


# Custom MAPE implementation since scikit-learn doesn't have it directly
def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def load_and_prepare_data():
    """Load and prepare diabetes dataset"""
    diabetes = datasets.load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['target'] = diabetes.target

    # Analyze correlation to select feature
    print("\nFeature correlation with target:")
    print(df.corr()['target'].sort_values(ascending=False))

    # Select BMI as most correlated feature
    X = df[['bmi']].values
    y = df['target'].values

    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.3, random_state=42)


class CustomLinearRegression:
    """Custom implementation of linear regression with gradient descent"""

    def __init__(self, learning_rate=0.01, n_iter=1000):
        """
        Initialize model
        :param learning_rate: Learning rate for gradient descent
        :param n_iter: Number of iterations
        """
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """Train the model"""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            # Predictions
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute and store loss (MSE)
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """Make predictions"""
        return np.dot(X, self.weights) + self.bias


def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model using MAE, R2, and MAPE metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print(f"\n{model_name} Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"R2: {r2:.2f}")
    print(f"MAPE: {mape:.2f}%")

    return {'MAE': mae, 'R2': r2, 'MAPE': mape}


def plot_results(X_test, y_test, y_pred_sklearn, y_pred_custom):
    """Visualize regression results"""
    plt.figure(figsize=(12, 5))

    # Plot regression lines
    plt.subplot(1, 2, 1)
    plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Actual')
    plt.plot(X_test, y_pred_sklearn, color='red', linewidth=2, label='Scikit-Learn')
    plt.plot(X_test, y_pred_custom, color='green', linestyle='--', label='Custom')
    plt.xlabel('Normalized BMI')
    plt.ylabel('Diabetes Progression')
    plt.title('Regression Model Comparison')
    plt.legend()
    plt.grid(True)

    # Plot predictions vs actual
    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_pred_sklearn, color='red', alpha=0.5, label='Scikit-Learn')
    plt.scatter(y_test, y_pred_custom, color='green', alpha=0.5, label='Custom')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # Train scikit-learn model
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test)

    # Train custom model
    custom_model = CustomLinearRegression(learning_rate=0.1, n_iter=2000)
    custom_model.fit(X_train, y_train)
    y_pred_custom = custom_model.predict(X_test)

    # Print model coefficients
    print("\nModel Coefficients:")
    print(f"Scikit-Learn: w={sklearn_model.coef_[0]:.4f}, b={sklearn_model.intercept_:.4f}")
    print(f"Custom: w={custom_model.weights[0]:.4f}, b={custom_model.bias:.4f}")

    # Evaluate models
    sklearn_metrics = evaluate_model(y_test, y_pred_sklearn, "Scikit-Learn")
    custom_metrics = evaluate_model(y_test, y_pred_custom, "Custom")

    # Visualize results
    plot_results(X_test, y_test, y_pred_sklearn, y_pred_custom)

    # Display predictions table
    results = pd.DataFrame({
        'Actual': y_test,
        'Scikit-Learn': y_pred_sklearn,
        'Custom': y_pred_custom,
        'Scikit-Learn Error': np.abs(y_test - y_pred_sklearn),
        'Custom Error': np.abs(y_test - y_pred_custom)
    })

    print("\nSample Predictions (first 10):")
    print(results.head(10).round(2))
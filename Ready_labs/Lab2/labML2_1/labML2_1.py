import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#Загрузка и исследование данных
def load_and_explore_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['target_name'] = df['target'].map(lambda x: iris.target_names[x])

    print("Первые 5 строк данных:")
    print(df.head())
    print("\nИнформация о данных:")
    print(df.info())
    print("\nСтатистика данных:")
    print(df.describe())

    return df, iris

#Визуализация данных
def visualize_data(df, iris):
    plt.figure(figsize=(18, 6))

    # 1.Визуализация сепалов
    plt.subplot(1, 3, 1)
    sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)',
                    hue='target_name', palette='viridis')
    plt.title('Sepal Length vs Sepal Width')
    plt.legend(title='Class')

    # 2.Визуализация лепестков
    plt.subplot(1, 3, 2)
    sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)',
                    hue='target_name', palette='viridis')
    plt.title('Petal Length vs Petal Width')
    plt.legend(title='Class')

    # 3.График Generated Dataset
    plt.subplot(1, 3, 3)
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42,
        n_clusters_per_class=1
    )

    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='viridis')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Generated Dataset')
    plt.legend(title='Class')

    plt.tight_layout()
    plt.show()

    # 4.График Pairplot
    sns.pairplot(df, vars=iris.feature_names, hue='target_name', palette='viridis')
    plt.suptitle('Pairplot of Iris Dataset', y=1.02)
    plt.show()


#Подготовка данных и моделирование
def prepare_and_model(df, iris):
    # Создаем два подмножества данных
    df1 = df[df['target'].isin([0, 1])]
    df2 = df[df['target'].isin([1, 2])]

    #Функция для обучения и оценки модели
    def train_and_evaluate(X, y, name):
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        #Пайплайна с масштабированием и моделью
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=42, max_iter=1000)
        )

        #Обучение модели
        model.fit(X_train, y_train)

        #Предсказания
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        #Оценка модели
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        print(f"\nРезультаты для {name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print("\nМатрица ошибок:")
        print(conf_matrix)
        print("\nОтчет классификации:")
        print(class_report)

        return model

    #Обучение
    print("\n" + "=" * 50)
    print("Модель для различения setosa (0) и versicolor (1)")
    X1 = df1[iris.feature_names]
    y1 = df1['target']
    model1 = train_and_evaluate(X1, y1, "Setosa vs Versicolor")

    print("\n" + "=" * 50)
    print("Модель для различения versicolor (1) и virginica (2)")
    X2 = df2[iris.feature_names]
    y2 = df2['target']
    model2 = train_and_evaluate(X2, y2, "Versicolor vs Virginica")

    return model1, model2

#Работа с синтетическими данными
def synthetic_data_example():
    #Генерация данных
    X, y = make_classification(
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=42,
        n_clusters_per_class=1,
        flip_y=0.05
    )

    #Обучение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(random_state=42)
    )
    model.fit(X_train, y_train)

    #Оценка
    accuracy = model.score(X_test, y_test)
    print(f"\nAccuracy на синтетических данных: {accuracy:.4f}")

    return model



if __name__ == "__main__":
    df, iris = load_and_explore_data()
    visualize_data(df, iris)

    # Моделирование на реальных данных
    model1, model2 = prepare_and_model(df, iris)

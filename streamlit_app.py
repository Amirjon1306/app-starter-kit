import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Заголовок приложения
st.title('Diabetes Prediction using Decision Tree')

# # EDA (Exploratory Data Analysis)
# Загрузка данных
data = pd.read_csv('diabetes_prediction_dataset.csv.zip')
data = data[data['smoking_history'] != 'No Info']

# Кодирование категориальных признаков для корреляционной матрицы
label_encoder = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = label_encoder.fit_transform(data[col])

# Построение корреляционной матрицы
corr_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Корреляционная матрица')
st.pyplot(plt)

# # Preprocessing
# Удаление пропущенных значений
data.dropna(inplace=True)

# Разделение данных на признаки (X) и целевую переменную (y)
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Нормализация данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Modeling
# Обучение модели Decision Tree с заданными гиперпараметрами
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Оценка модели
y_pred = model.predict(X_test)
st.write('Confusion Matrix:')
st.write(confusion_matrix(y_test, y_pred))
st.write('Classification Report:')
st.text(classification_report(y_test, y_pred))

# Важность признаков
feature_importances = pd.Series(model.feature_importances_, index=data.drop('diabetes', axis=1).columns)
plt.figure(figsize=(12, 6))
feature_importances.sort_values(ascending=False).plot(kind='bar')
plt.title('Важность признаков')
st.pyplot(plt)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
import kagglehub
import json
import ast
from sklearn.preprocessing import MultiLabelBinarizer

path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")

df = pd.read_csv(path + '\\tmdb_5000_movies.csv')

df['profit'] = df['revenue'] - df['budget']
df['profit_margin'] = df['revenue'] / df['budget']
df['is_success'] = (df['profit_margin'] > 2).astype(int)

df_features = df[['budget', 'runtime', 'release_date', 'is_success', 'original_language']]
df_features['release_date'] = pd.to_datetime(df_features['release_date'])

df_features.dropna()
df_features['release_month'] = df_features['release_date'].dt.month
df_features['release_weekday'] = df_features['release_date'].dt.weekday
df_features['runtime'] = df_features['runtime']
df_features['release_month'] = df_features['release_month'].replace([np.inf, -np.inf], np.nan)
df_features['release_weekday'] = df_features['release_weekday'].replace([np.inf, -np.inf], np.nan)
df_features['runtime'] = df_features['runtime'].replace([np.inf, -np.inf], np.nan)
df_features = df_features.dropna()
df_features['release_month'] = df_features['release_month'].astype(int)
df_features['release_weekday'] = df_features['release_weekday'].astype(int)
df_features['runtime'] = df_features['runtime'].astype(int)

df_features['genres'] = df['genres'].apply(lambda x: [g['name'] for g in ast.literal_eval(x)] if pd.notna(x) else [])

mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(df_features['genres'])
genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)

df_features['is_english'] = (df['original_language'] == 'en').astype(int)

lang_counts = df['original_language'].value_counts()

plt.figure(figsize=(10, 6))
lang_counts.plot(kind='bar')
plt.title('Количество фильмов по языкам')
plt.xlabel('Язык')
plt.ylabel('Количество фильмов')
plt.xticks(rotation=45)
plt.show()


# Доля английского
english_share = (df['original_language'] == 'en').mean()
print(f'Доля фильмов на английском: {english_share:.1%}')

df_features = df_features.drop('release_date', axis=1)
df_features = df_features.drop('original_language', axis=1)

df_features = df_features.reset_index(drop=True)
genres_df = genres_df.reset_index(drop=True)

X = pd.concat([df_features[['budget', 'runtime', 'release_month', 'release_weekday', 'is_english']], 
               genres_df], axis=1)
Y = df_features['is_success']

print(X.shape) 
print(Y.shape) 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(y_test, y_pred)


print("РЕЗУЛЬТАТЫ МОДЕЛИ XGBoost")
print(f"Accuracy  (точность):     {accuracy:.4f} ({accuracy*100:.2f}%)")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Неуспешный', 'Успешный'],
            yticklabels=['Неуспешный', 'Успешный'])
plt.title(f'Матрица ошибок XGBoost\nAccuracy = {accuracy:.2%}')
plt.ylabel('Истинный класс')
plt.xlabel('Предсказанный класс')
plt.show()

importance_df = pd.DataFrame({
    'Признак': X.columns,
    'Важность': model.feature_importances_
}).sort_values('Важность', ascending=False)

plt.figure(figsize=(10,6))
plt.barh(importance_df['Признак'].head(10), importance_df['Важность'].head(10), color='steelblue')
plt.xlabel('Важность')
plt.title('Топ-10 важных признаков (XGBoost)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
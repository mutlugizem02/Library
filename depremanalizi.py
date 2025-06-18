import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import joblib

file_path = "turkey_earthquakes(1915-2023_may).csv"

df = pd.read_csv(file_path, encoding='ISO-8859-1')

features = ['xM', 'MD', 'ML', 'Mb', 'Ms']
train_df = df[df['Mw'].notnull()]
X_train = train_df[features]
y_train = train_df['Mw']
test_df = df[df['Mw'].isnull()]
X_test = test_df[features]
X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]
X_test = X_test.dropna()
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predicted_mw = model.predict(X_test)
df.loc[X_test.index, 'Mw'] = predicted_mw
df['Mw_is_imputed'] = df['Mw'].isnull().astype(int)
df['Olus tarihi'] = pd.to_datetime(df['Olus tarihi'])
df = df[['Olus tarihi', 'Olus zamani', 'Enlem', 'Boylam', 'Derinlik', 'Mw', 'ML', 'xM', 'Tip', 'Yer']]
df['Büyüklük'] = df['Mw'].fillna(df['ML']).fillna(df['xM'])
df['Timestamp'] = pd.to_datetime(df['Olus tarihi'].astype(str) + ' ' + df['Olus zamani'].astype(str))
df.drop(['Olus tarihi', 'Olus zamani', 'Mw', 'ML', 'xM'], axis=1, inplace=True)
df = df.rename(columns={'Timestamp': 'Oluş Zamanı'})
df['Log_Büyüklük'] = np.log10(df['Büyüklük'] + 0.1)
scaler = StandardScaler()
df['Log_Büyüklük_Std'] = scaler.fit_transform(df[['Log_Büyüklük']])
df['Büyüklük_Kategori'] = pd.cut(df['Büyüklük'], bins=[0, 3, 5, np.inf], labels=['Hafif', 'Orta', 'Şiddetli'])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(data=df, x='Büyüklük', ax=axes[0])
axes[0].set_title('Histogram')
sns.boxplot(data=df, x='Büyüklük', ax=axes[1])
axes[1].set_title('Boxplot')
sns.ecdfplot(data=df, x='Büyüklük', complementary=True, ax=axes[2])
axes[2].set_yscale('log')
axes[2].set_title('Kümülatif Dağılım (Log-Log)')
st.pyplot(fig)

def label_aftershocks(df, mainshock_mag_threshold=6.0, time_window_days=7, distance_km=50):
    df = df.sort_values("Oluş Zamanı").reset_index(drop=True)
    df['is_aftershock'] = 0
    for i, row in df.iterrows():
        if row['Büyüklük'] >= mainshock_mag_threshold:
            mainshock_time = row['Oluş Zamanı']
            mainshock_loc = (row['Enlem'], row['Boylam'])
            time_window = (df['Oluş Zamanı'] > mainshock_time) & \
                          (df['Oluş Zamanı'] <= mainshock_time + pd.Timedelta(days=time_window_days))
            for j in df[time_window].index:
                aftershock_loc = (df.loc[j, 'Enlem'], df.loc[j, 'Boylam'])
                distance = geodesic(mainshock_loc, aftershock_loc).km
                if distance <= distance_km and df.loc[j, 'Büyüklük'] < row['Büyüklük']:
                    df.at[j, 'is_aftershock'] = 1
    return df

df = label_aftershocks(df)

def ozellikleri_olustur(df):
    df['Saat'] = df['Oluş Zamanı'].dt.hour
    df['Haftanın Günü'] = df['Oluş Zamanı'].dt.dayofweek
    df = df.sort_values("Oluş Zamanı").reset_index(drop=True)
    df['Önceki Zaman Farkı (sn)'] = df['Oluş Zamanı'].diff().dt.total_seconds().fillna(0)
    df['Önceki Mesafe (km)'] = [0] + [geodesic((df.loc[i-1, 'Enlem'], df.loc[i-1, 'Boylam']), (df.loc[i, 'Enlem'], df.loc[i, 'Boylam'])).km for i in range(1, len(df))]
    return df

df = ozellikleri_olustur(df)

ozellikler = ['Büyüklük', 'Derinlik', 'Saat', 'Haftanın Günü', 'Önceki Zaman Farkı (sn)', 'Önceki Mesafe (km)']
X = df[ozellikler]
y = df['is_aftershock']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

X_train = X_train.dropna()
y_train = y_train.loc[X_train.index]

model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

modeller = {
    "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": XGBClassifier(random_state=42, scale_pos_weight=12),
    "LightGBM": LGBMClassifier(random_state=42, class_weight='balanced')
}

for isim, model in modeller.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))


df_cleaned = df.dropna(subset=ozellikler + ['is_aftershock']).copy()
X = df_cleaned[ozellikler]
y = df_cleaned['is_aftershock']


print("Tip kontrolü:\n", X.dtypes)

print("Eksik veri kontrolü:\n", X.isnull().sum())


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

X_train = X_train.dropna().reset_index(drop=True)
y_train = y_train.loc[X_train.index].reset_index(drop=True)

X_train = X_train.astype(float)
y_train = y_train.astype(int)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


df_cleaned = df.dropna(subset=ozellikler + ['is_aftershock']).copy()
X = df_cleaned[ozellikler]
y = df_cleaned['is_aftershock']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

X_train = X_train.dropna().reset_index(drop=True)
y_train = y_train.loc[X_train.index].reset_index(drop=True)

X_train = X_train.astype(float)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train_resampled, y_train_resampled)
y_pred = xgb.predict(X_test)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
grid = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='f1', cv=3, n_jobs=-1, verbose=1)
grid.fit(X_train_resampled, y_train_resampled)
best_xgb = grid.best_estimator_
y_pred_best = best_xgb.predict(X_test)
y_proba = best_xgb.predict_proba(X_test)[:, 1]

for thresh in [0.5, 0.4, 0.35, 0.3]:
    y_pred_thresh = (y_proba >= thresh).astype(int)
    print(confusion_matrix(y_test, y_pred_thresh))

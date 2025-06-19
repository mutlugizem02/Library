import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from geopy.distance import geodesic
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import joblib

st.set_page_config(page_title="Türkiye Deprem Analizi", layout="wide", page_icon="🌍")
st.title("🌍 Türkiye Deprem Analizi (1915-2023)")

@st.cache_data
def load_data():
    try:
        file_path = "turkey_earthquakes(1915-2023_may).csv"
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        return df
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {str(e)}")
        return None

def preprocess_data(df):
    features = ['xM', 'MD', 'ML', 'Mb', 'Ms']
    train_df = df[df['Mw'].notnull()]
    X_train_mw = train_df[features].dropna()
    y_train_mw = train_df['Mw'].loc[X_train_mw.index]
    
    test_df = df[df['Mw'].isnull()]
    X_test_mw = test_df[features].dropna()
    
    model_mw = RandomForestRegressor(n_estimators=100, random_state=42)
    model_mw.fit(X_train_mw, y_train_mw)
    predicted_mw = model_mw.predict(X_test_mw)
    df.loc[X_test_mw.index, 'Mw'] = predicted_mw
    
    df['Olus tarihi'] = pd.to_datetime(df['Olus tarihi'])
    df['Büyüklük'] = df['Mw'].fillna(df['ML']).fillna(df['xM'])
    df['Oluş Zamanı'] = pd.to_datetime(df['Olus tarihi'].astype(str) + ' ' + df['Olus zamani'].astype(str))
    
    df = label_aftershocks(df)
    df = create_features(df)
    
    return df

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

def create_features(df):
    df['Saat'] = df['Oluş Zamanı'].dt.hour
    df['Haftanın Günü'] = df['Oluş Zamanı'].dt.dayofweek
    df = df.sort_values("Oluş Zamanı").reset_index(drop=True)
    df['Önceki Zaman Farkı (sn)'] = df['Oluş Zamanı'].diff().dt.total_seconds().fillna(0)
    df['Önceki Mesafe (km)'] = [0] + [geodesic((df.loc[i-1, 'Enlem'], df.loc[i-1, 'Boylam']), 
                                    (df.loc[i, 'Enlem'], df.loc[i, 'Boylam'])).km 
                                    for i in range(1, len(df))]
    return df

def main():
    df = load_data()
    if df is None:
        return
    
    with st.expander("📊 Ham Veri Önizleme"):
        st.dataframe(df.head())
    
    with st.spinner('Veri işleniyor...'):
        df_processed = preprocess_data(df)
        
        features = ['Büyüklük', 'Derinlik', 'Saat', 'Haftanın Günü', 
                   'Önceki Zaman Farkı (sn)', 'Önceki Mesafe (km)']
        
        df_cleaned = df_processed.dropna(subset=features + ['is_aftershock']).copy()
        for col in features:
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
        df_cleaned = df_cleaned.dropna().reset_index(drop=True)
        
        X = df_cleaned[features]
        y = df_cleaned['is_aftershock']
    
    st.subheader("📈 Veri Analizi")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots()
        sns.histplot(data=df_cleaned, x='Büyüklük', kde=True, ax=ax)
        ax.set_title('Deprem Büyüklük Dağılımı')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots()
        y.value_counts().plot(kind='bar', ax=ax)
        ax.set_title('Artçı Şok Dağılımı')
        ax.set_xlabel('Artçı Şok (1) veya Değil (0)')
        ax.set_ylabel('Sayı')
        st.pyplot(fig)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
 
    # Örnekleme işlemi (şimdilik devre dışı bırakıldı)
    # try:
    #     ros = RandomOverSampler(random_state=42)
    #     X_train_resampled, y_train_resampled = ros.fit_resample(X_train_scaled, y_train)
    #     st.info("Veri örnekleme başarıyla uygulandı.")
    # except Exception as e:
    #     st.warning(f"Örnekleme hatası: {str(e)} - Örnekleme yapılmadan devam ediliyor...")
    #     X_train_resampled, y_train_resampled = X_train_scaled, y_train

    # Geçici çözüm: örnekleme yapılmadan devam
    X_train_resampled, y_train_resampled = X_train_scaled, y_train

    
    st.subheader("🤖 Model Eğitimi")
    with st.spinner('XGBoost modeli eğitiliyor...'):
        xgb = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1
        )
        xgb.fit(X_train_resampled, y_train_resampled)
        st.success("Model başarıyla eğitildi!")
    
    
    st.subheader("📊 Model Performansı")
    threshold = st.slider('Sınıflandırma Eşik Değeri', 0.1, 0.9, 0.5, 0.05)
    y_proba = xgb.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Karışıklık Matrisi")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)
    
    with col2:
        st.write("### Sınıflandırma Raporu")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
    
    st.subheader("🔍 Özellik Önemleri")
    feat_imp = pd.DataFrame({
        'Özellik': features,
        'Önem': xgb.feature_importances_
    }).sort_values('Önem', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feat_imp, x='Önem', y='Özellik', ax=ax)
    ax.set_title('XGBoost Özellik Önemleri')
    st.pyplot(fig)
    
    if st.button("💾 Modeli Kaydet"):
        joblib.dump(xgb, 'deprem_modeli.pkl')
        st.success("Model başarıyla kaydedildi!")

if __name__ == "__main__":
    main()

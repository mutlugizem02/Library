# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Streamlit sayfa ayarı
st.set_page_config(layout="wide", page_title="Deprem Analizi Uygulaması")

# Veri yükleme fonksiyonu
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("deprem_verisi.csv", encoding='utf-8-sig')
    except:
        df = pd.DataFrame(data)
        df['Log_Büyüklük'] = np.log10(df['Büyüklük'] + 0.1)
    return df

# Veri ön işleme fonksiyonları
def preprocess_data(df):
    # Büyüklük sınıfları
    df['Büyüklük_Sınıfı'] = pd.cut(df['Büyüklük'],
                                  bins=[0, 3, 5, 10],
                                  labels=['Küçük', 'Orta', 'Büyük'])
    
    # Log dönüşümü ve standardizasyon
    df['Log_Büyüklük'] = np.log10(df['Büyüklük'] + 0.1)
    scaler = StandardScaler()
    df['Log_Büyüklük_Std'] = scaler.fit_transform(df[['Log_Büyüklük']])
    
    # Tarih özellikleri
    df['Oluş Zamanı'] = pd.to_datetime(df['Oluş Zamanı'])
    df['Saat'] = df['Oluş Zamanı'].dt.hour
    df['Haftanın Günü'] = df['Oluş Zamanı'].dt.dayofweek
    
    return df

# Artçı şok tespit fonksiyonu
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

# Grafik oluşturma fonksiyonları
def create_histograms(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.histplot(df['Büyüklük'], bins=30, kde=True, ax=ax1)
    ax1.set_title('Orijinal Büyüklük Dağılımı')
    
    sns.histplot(df['Log_Büyüklük'], bins=30, kde=True, ax=ax2)
    ax2.set_title('Log10 Dönüşümlü Dağılım')
    
    fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15, wspace=0.25)
    return fig

def create_violin_plots(df):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    sns.violinplot(x=df['Büyüklük'], color='lightblue', ax=ax1)
    ax1.set_title('Orijinal Büyüklük Dağılımı')
    
    sns.violinplot(x=df['Log_Büyüklük'], color='lightgreen', ax=ax2)
    ax2.set_title('Log10 Dönüşümlü Dağılım')
    
    sns.ecdfplot(data=df, x='Büyüklük', complementary=True, ax=ax3)
    ax3.set_yscale('log')
    ax3.set_title('Kümülatif Dağılım')
    
    fig.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15, wspace=0.3)
    return fig

# Model eğitimi fonksiyonları
def train_models(X_train, y_train, X_test, y_test):
    modeller = {
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(random_state=42, scale_pos_weight=12),
        "LightGBM": LGBMClassifier(random_state=42, class_weight='balanced')
    }
    
    results = {}
    for isim, model in modeller.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[isim] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    return results

# Streamlit arayüzü
def main():
    st.title("🇹🇷 Türkiye Deprem Analizi Uygulaması")
    
    # Veri yükleme
    df = load_data()
    df = preprocess_data(df)
    
    # Sidebar
    st.sidebar.header("Analiz Parametreleri")
    min_magnitude = st.sidebar.slider("Minimum Büyüklük", 2.0, 7.0, 3.0)
    filtered_df = df[df['Büyüklük'] >= min_magnitude]
    
    # Ana sayfa
    tab1, tab2, tab3 = st.tabs(["Veri Analizi", "Görselleştirme", "Makine Öğrenmesi"])
    
    with tab1:
        st.header("Veri Özeti")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### İlk 5 Kayıt")
            st.dataframe(filtered_df.head())
            
        with col2:
            st.write("### Temel İstatistikler")
            st.dataframe(filtered_df.describe())
        
        st.write(f"### Toplam Kayıt Sayısı: {len(filtered_df)}")
    
    with tab2:
        st.header("Veri Görselleştirme")
        
        st.write("### Büyüklük Dağılımları")
        try:
            st.pyplot(create_histograms(filtered_df))
        except Exception as e:
            st.error(f"Grafik oluşturma hatası: {str(e)}")
        
        st.write("### Detaylı Dağılım Analizi")
        try:
            st.pyplot(create_violin_plots(filtered_df))
        except Exception as e:
            st.error(f"Grafik oluşturma hatası: {str(e)}")
        
        # Harita görselleştirme
        st.write("### Depremlerin Coğrafi Dağılımı")
        try:
            st.map(filtered_df[['Enlem', 'Boylam']].rename(columns={'Enlem': 'lat', 'Boylam': 'lon'}))
        except Exception as e:
            st.error(f"Harita oluşturma hatası: {str(e)}")
    
    with tab3:
        st.header("Makine Öğrenmesi Analizleri")
        
        # Artçı şok tespiti
        st.write("### Artçı Şok Tahmini")
        if st.button("Artçı Şokları İşaretle"):
            with st.spinner("Artçı şoklar işaretleniyor..."):
                df = label_aftershocks(df)
                st.success("Artçı şoklar başarıyla işaretlendi!")
                st.write(df['is_aftershock'].value_counts())
        
        # Model eğitimi
        if 'is_aftershock' in df.columns:
            st.write("#### Model Eğitimi")
            ozellikler = ['Büyüklük', 'Derinlik', 'Saat', 'Haftanın Günü']
            X = df[ozellikler]
            y = df['is_aftershock']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)
            
            # SMOTE uygulama
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
            if st.button("Modelleri Eğit"):
                with st.spinner("Modeller eğitiliyor..."):
                    results = train_models(X_train_resampled, y_train_resampled, X_test, y_test)
                    
                    for model_name, result in results.items():
                        st.write(f"#### {model_name} Sonuçları")
                        st.write(f"Doğruluk: {result['accuracy']:.2f}")
                        st.dataframe(pd.DataFrame(result['report']).transpose())
        
        # Model kaydetme
        st.write("#### Model Kaydetme")
        if st.button("XGBoost Modelini Kaydet"):
            try:
                xgb = XGBClassifier(random_state=42)
                xgb.fit(X_train_resampled, y_train_resampled)
                joblib.dump(xgb, 'xgboost_model.pkl')
                st.success("Model başarıyla kaydedildi!")
            except Exception as e:
                st.error(f"Model kaydetme hatası: {str(e)}")

if __name__ == "__main__":
    main()
    plt.close('all')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from geopy.distance import geodesic
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler  


st.set_page_config(page_title="Deprem Analizi", layout="wide")
st.title("Türkiye Deprem Analizi (1915-2023)")

@st.cache_data
def load_data():
    file_path = "turkey_earthquakes(1915-2023_may).csv"
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    return df

try:
    df = load_data()
    
    with st.expander("Ham Veri Görünümü"):
        st.dataframe(df.head())

   
    with st.spinner('Mw değerleri tahmin ediliyor...'):
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
        df['Mw_is_imputed'] = df['Mw'].isnull().astype(int)

    
    with st.spinner('Veri işleniyor...'):
        df['Olus tarihi'] = pd.to_datetime(df['Olus tarihi'])
        df = df[['Olus tarihi', 'Olus zamani', 'Enlem', 'Boylam', 'Derinlik', 'Mw', 'ML', 'xM', 'Tip', 'Yer']]
        df['Büyüklük'] = df['Mw'].fillna(df['ML']).fillna(df['xM'])
        df['Oluş Zamanı'] = pd.to_datetime(df['Olus tarihi'].astype(str) + ' ' + df['Olus zamani'].astype(str))
        df.drop(['Olus tarihi', 'Olus zamani', 'Mw', 'ML', 'xM'], axis=1, inplace=True)

        
        st.subheader("Deprem Büyüklük Dağılımı")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        sns.histplot(data=df, x='Büyüklük', ax=axes[0], kde=True)
        axes[0].set_title('Deprem Büyüklük Dağılımı')
        sns.boxplot(data=df, x='Büyüklük', ax=axes[1])
        axes[1].set_title('Büyüklük Kutu Grafiği')
        sns.scatterplot(data=df, x='Derinlik', y='Büyüklük', ax=axes[2])
        axes[2].set_title('Derinlik-Büyüklük İlişkisi')
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

    
    def create_features(df):
        df['Saat'] = df['Oluş Zamanı'].dt.hour
        df['Haftanın Günü'] = df['Oluş Zamanı'].dt.dayofweek
        df = df.sort_values("Oluş Zamanı").reset_index(drop=True)
        df['Önceki Zaman Farkı (sn)'] = df['Oluş Zamanı'].diff().dt.total_seconds().fillna(0)
        df['Önceki Mesafe (km)'] = [0] + [geodesic((df.loc[i-1, 'Enlem'], df.loc[i-1, 'Boylam']), 
                                      (df.loc[i, 'Enlem'], df.loc[i, 'Boylam'])).km 
                                      for i in range(1, len(df))]
        return df

    df = create_features(df)

    
    features = ['Büyüklük', 'Derinlik', 'Saat', 'Haftanın Günü', 
               'Önceki Zaman Farkı (sn)', 'Önceki Mesafe (km)']
    
    
    df_cleaned = df.dropna(subset=features + ['is_aftershock']).reset_index(drop=True).copy()
    
    
    for col in features:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    
    df_cleaned = df_cleaned.dropna().reset_index(drop=True)
    
    X = df_cleaned[features]
    y = df_cleaned['is_aftershock']

    
    st.subheader("Sınıf Dağılımı")
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

    
    with st.spinner('Veri örnekleme uygulanıyor...'):
        ros = RandomOverSampler(random_state=42)
        X_train_resampled, y_train_resampled = ros.fit_resample(X_train_scaled, y_train)
        st.success(f"Örnekleme sonrası sınıf dağılımı: \n{pd.Series(y_train_resampled).value_counts()}")

    
    with st.spinner('Model eğitiliyor...'):
        xgb = XGBClassifier(
            use_label_encoder=False, 
            eval_metric='logloss', 
            random_state=42,
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9
        )
        xgb.fit(X_train_resampled, y_train_resampled)
        st.success("Model başarıyla eğitildi!")

    
    st.subheader("Model Performansı")
    
    
    y_proba = xgb.predict_proba(X_test_scaled)[:, 1]
    
    threshold = st.slider('Sınıflandırma Eşik Değeri', 0.1, 0.9, 0.5, 0.05)
    y_pred = (y_proba >= threshold).astype(int)
    
    st.write("### Karışıklık Matrisi")
    st.write(confusion_matrix(y_test, y_pred))
    
    st.write("### Sınıflandırma Raporu")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    
    st.subheader("Özellik Önemleri")
    feat_imp = pd.DataFrame({
        'Özellik': features,
        'Önem': xgb.feature_importances_
    }).sort_values('Önem', ascending=False)
    
    fig, ax = plt.subplots()
    sns.barplot(data=feat_imp, x='Önem', y='Özellik', ax=ax)
    ax.set_title('XGBoost Özellik Önemleri')
    st.pyplot(fig)

except Exception as e:
    st.error(f"Bir hata oluştu: {str(e)}")
    st.error("Lütfen veri dosyasını ve bağımlılıkları kontrol edin.")

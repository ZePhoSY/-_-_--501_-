import streamlit as st
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Налаштування сторінки інтерфейсу
st.set_page_config(page_title="Прогноз опадів ML", page_icon="☔", layout="wide")

st.title("☔ Прототип сервісу прогнозування опадів на основі машинного навчання")
st.markdown("Застосунок завантажує історичні дані через Open-Meteo API, навчає оптимізовану модель Випадкового лісу та здійснює предиктивний аналіз.")

# --- БЛОК КЕШУВАННЯ ДАНИХ ТА НАВЧАННЯ МОДЕЛІ ---

@st.cache_data
def load_historical_data(lat, lon):
    """Завантаження 20-річного архіву погоди для формування вибірки"""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": "2014-01-01",
        "end_date": "2023-12-31",
        "daily": "precipitation_sum,temperature_2m_max,relative_humidity_2m_mean,wind_speed_10m_max,pressure_msl_mean",
        "timezone": "Europe/Kyiv"
    }
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df['target'] = (df['precipitation_sum'] > 0).astype(int)
    return df

@st.cache_resource
def train_final_model(df):
    """Навчання моделі за найкращими параметрами, знайденими через GridSearch"""
    features = ['temperature_2m_max', 'relative_humidity_2m_mean', 'wind_speed_10m_max', 'pressure_msl_mean']
    X = df[features]
    y = df['target']
    
    # Розподіл для генерації валідаційного звіту
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Використання оптимальних гіперпараметрів, знайдених під час дослідження
    model = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_leaf=4, random_state=42)
    model.fit(X_train, y_train)
    
    # Генеруємо звіт класифікації у вигляді словника для відображення в інтерфейсі
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, target_names=["Опадів немає", "Опади є"], output_dict=True)
    
    return model, report, features

# --- БІДЖЕТИ КЕРУВАННЯ НА БОКОВІЙ ПАНЕЛІ (SIDEBAR) ---
st.sidebar.header("📍 Географічні координати")
latitude = st.sidebar.number_input("Широта (Latitude):", value=50.45, step=0.01, format="%.2f")
longitude = st.sidebar.number_input("Довгота (Longitude):", value=30.52, step=0.01, format="%.2f")

# Ініціалізація процесів зчитування та навчання
historical_df = load_historical_data(latitude, longitude)
best_model, report_dict, selected_features = train_final_model(historical_df)

# --- ЕЛЕМЕНТ ІНТЕРФЕЙСУ 1: ПЕРЕГЛЯД ДАНИХ ---
st.header("1. Історичний набір метеоданих (2014-2023)")
with st.expander("Розгорнути для перегляду завантаженого датасету CSV"):
    st.dataframe(historical_df, use_container_width=True)
    st.download_button(
        label="Завантажити датасет як CSV-файл",
        data=historical_df.to_csv(index=False).encode('utf-8'),
        file_name='weather_daily.csv',
        mime='text/csv'
    )

# --- ЕЛЕМЕНТ ІНТЕРФЕЙСУ 2: МЕТРИКИ МОДЕЛІ ---
st.header("2. Оцінка якості фінальної моделі")
st.markdown("**Метрики класифікації оптимізованого Випадкового лісу (Тестова вибірка):**")

col1, col2, col3 = st.columns(3)
col1.metric("Загальна точність (Accuracy)", f"{report_dict['accuracy']:.2f}")
col2.metric("Precision (Клас: Опади є)", f"{report_dict['Опади є']['precision']:.2f}")
col3.metric("Recall (Клас: Опади є)", f"{report_dict['Опади є']['recall']:.2f}")

with st.expander("Переглянути повний classification_report"):
    st.json(report_dict)

# --- ЕЛЕМЕНТ ІНТЕРФЕЙСУ 3: ОПЕРАТИВНИЙ ПРОГНОЗ НА МАЙБУТНЄ ---
st.header("3. Оперативний ML-прогноз опадів")
forecast_days = st.slider("Кількість днів для прогнозування:", min_value=1, max_value=7, value=3)

if st.button("Згенерувати актуальний прогноз", type="primary"):
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    forecast_params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,relative_humidity_2m_mean,wind_speed_10m_max,pressure_msl_mean",
        "timezone": "Europe/Kyiv",
        "forecast_days": forecast_days
    }
    
    res = requests.get(forecast_url, params=forecast_params)
    data_f = res.json()
    df_f = pd.DataFrame(data_f["daily"])
    
    X_f = df_f[selected_features]
    preds_f = best_model.predict(X_f)
    probs_f = best_model.predict_proba(X_f)[:, 1]
    
    st.markdown("### Результати аналізу:")
    
    # Вивід картками для кожного дня
    cols = st.columns(forecast_days)
    for i in range(forecast_days):
        with cols[i]:
            date_str = str(df_f['time'][i])
            prob_percent = probs_f[i] * 100
            
            if preds_f[i] == 1:
                st.error(f"📅 {date_str}\n\n**Опади очікуються ☔**\n\nЙмовірність: {prob_percent:.1f}%")
            else:
                st.success(f"📅 {date_str}\n\n**Без опадів ☀️**\n\nЙмовірність: {prob_percent:.1f}%")
                
    # Зведена аналітична таблиця ознак для виконання умов "додаткової переваги"
    st.markdown("#### Зведена таблиця метеорологічних ознак прогнозованого періоду:")
    st.dataframe(df_f[['time'] + selected_features], use_container_width=True)

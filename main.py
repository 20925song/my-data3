import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="서울 기온 선형회귀 모델 분석", page_icon="📈", layout="wide")

st.title("📈 서울 연평균 기온 선형회귀 모델 평가 및 비교")
st.markdown("""
서울의 연평균 기온 데이터를 학습하여 선형회귀 모델을 생성하고,  
**1) 전체 데이터 평가**, **2) 과거 50년/100년 학습 데이터에 따른 최근 20년(2006~2025) 예측 성능**을 비교합니다.
""")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/seoul.csv"
    try:
        df = pd.read_csv(url, encoding='cp949')
    except Exception:
        df = pd.read_csv(url, encoding='utf-8')
    
    df.columns = df.columns.str.strip()
    col_map = {col: '날짜' if '날짜' in col else '평균기온' if '평균기온' in col else col for col in df.columns}
    df = df.rename(columns=col_map)
    
    df['날짜'] = pd.to_datetime(df['날짜'].astype(str).str.strip(), errors='coerce')
    df = df.dropna(subset=['날짜'])
    df['연도'] = df['날짜'].dt.year
    df['평균기온'] = pd.to_numeric(df['평균기온'], errors='coerce')
    
    # 300일 이상 존재하는 정상 연도만 추출
    valid_years = df.groupby('연도')['평균기온'].count()
    valid_years = valid_years[valid_years >= 300].index
    
    yearly_df = df[df['연도'].isin(valid_years)].groupby('연도')['평균기온'].mean().reset_index()
    yearly_df.columns = ['연도', '연평균기온']
    return yearly_df

yearly_df = load_data()

# Data Partitioning
train_50 = yearly_df[(yearly_df['연도'] >= 1956) & (yearly_df['연도'] <= 2005)]
train_100 = yearly_df[(yearly_df['연도'] >= 1906) & (yearly_df['연도'] <= 2005)]
test_20 = yearly_df[(yearly_df['연도'] >= 2006) & (yearly_df['연도'] <= 2025)]

# Helper function
def run_linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, slope, intercept, y_pred, mae, mse, r2

# 1. 전체 데이터 모델
X_all = yearly_df[['연도']].values
y_all = yearly_df['연평균기온'].values
m_all, slope_all, intercept_all, y_pred_all, mae_all, mse_all, r2_all = run_linear_regression(X_all, y_all, X_all, y_all)

# 2. 최근 50년 학습 (1956~2005) -> 테스트 (2006~2025)
X_tr50, y_tr50 = train_50[['연도']].values, train_50['연평균기온'].values
X_te20, y_te20 = test_20[['연도']].values, test_20['연평균기온'].values
m_50, slope_50, intercept_50, y_pred_50, mae_50, mse_50, r2_50 = run_linear_regression(X_tr50, y_tr50, X_te20, y_te20)

# 3. 최근 100년 학습 (1906~2005) -> 테스트 (2006~2025)
X_tr100, y_tr100 = train_100[['연도']].values, train_100['연평균기온'].values
m_100, slope_100, intercept_100, y_pred_100, mae_100, mse_100, r2_100 = run_linear_regression(X_tr100, y_tr100, X_te20, y_te20)

# Section 1: 전체 데이터 모델 성능
st.subheader("1️⃣ 전체 데이터 기준 선형회귀 모델")
c1, c2, c3, c4 = st.columns(4)
c1.metric("기울기 (Slope)", f"{slope_all:+.5f} °C/년")
c2.metric("MAE (평균 절대 오차)", f"{mae_all:.4f}")
c3.metric("MSE (평균 제곱 오차)", f"{mse_all:.4f}")
c4.metric("R² (결정계수)", f"{r2_all:.4f}")

# Section 2: 모델 비교 (50년 학습 vs 100년 학습)
st.subheader("2️⃣ 공통 테스트 데이터(2006~2025) 예측 성능 비교")

comp_df = pd.DataFrame({
    "구분": ["최근 50년 학습 (1956~2005)", "최근 100년 학습 (1906~2005)"],
    "회귀선 기울기 (°C/년)": [f"{slope_50:+.5f}", f"{slope_100:+.5f}"],
    "MAE (낮을수록 우수)": [f"{mae_50:.4f}", f"{mae_100:.4f}"],
    "MSE (낮을수록 우수)": [f"{mse_50:.4f}", f"{mse_100:.4f}"],
    "R² (높을수록 우수)": [f"{r2_50:.4f}", f"{r2_100:.4f}"]
})
st.table(comp_df)

# Section 3: 시각화
st.subheader("📉 회귀선 비교 시각화")

fig = go.Figure()

# 실제 데이터 점
fig.add_trace(go.Scatter(
    x=yearly_df['연도'], y=yearly_df['연평균기온'],
    mode='markers', name='실제 연평균기온',
    marker=dict(color='gray', size=6, opacity=0.6)
))

# 테스트 데이터 점 강조
fig.add_trace(go.Scatter(
    x=test_20['연도'], y=test_20['연평균기온'],
    mode='markers', name='테스트 데이터 (2006~2025)',
    marker=dict(color='red', size=8)
))

# 50년 모델 예측선
x_range_50 = np.arange(1956, 2026).reshape(-1, 1)
fig.add_trace(go.Scatter(
    x=x_range_50.flatten(), y=m_50.predict(x_range_50),
    mode='lines', name=f'50년 학습 회귀선 (기울기: {slope_50:+.4f})',
    line=dict(color='blue', width=3)
))

# 100년 모델 예측선
x_range_1

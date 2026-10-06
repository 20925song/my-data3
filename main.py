import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="서울 기온 선형회귀 평가", page_icon="📈", layout="wide")

st.title("📈 서울 연평균기온 선형회귀 모델 평가 및 학습 기간별 비교")

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
    
    valid_years = df.groupby('연도')['평균기온'].count()
    valid_years = valid_years[valid_years >= 300].index
    
    yearly_df = df[df['연도'].isin(valid_years)].groupby('연도')['평균기온'].mean().reset_index()
    yearly_df.columns = ['연도', '연평균기온']
    return yearly_df

yearly_df = load_data()

# 데이터 분할
train_50 = yearly_df[(yearly_df['연도'] >= 1956) & (yearly_df['연도'] <= 2005)]
train_100 = yearly_df[(yearly_df['연도'] >= 1906) & (yearly_df['연도'] <= 2005)]
test_20 = yearly_df[(yearly_df['연도'] >= 2006) & (yearly_df['연도'] <= 2025)]

def evaluate_model(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    slope = model.coef_[0]
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, slope, mae, mse, r2

# 1. 전체 데이터 모델
X_all, y_all = yearly_df[['연도']].values, yearly_df['연평균기온'].values
m_all, slope_all, mae_all, mse_all, r2_all = evaluate_model(X_all, y_all, X_all, y_all)

# 2. 최근 50년 학습 -> 테스트 20년
X_tr50, y_tr50 = train_50[['연도']].values, train_50['연평균기온'].values
X_te20, y_te20 = test_20[['연도']].values, test_20['연평균기온'].values
m_50, slope_50, mae_50, mse_50, r2_50 = evaluate_model(X_tr50, y_tr50, X_te20, y_te20)

# 3. 최근 100년 학습 -> 테스트 20년
X_tr100, y_tr100 = train_100[['연도']].values, train_100['연평균기온'].values
m_100, slope_100, mae_100, mse_100, r2_100 = evaluate_model(X_tr100, y_tr100, X_te20, y_te20)

# UI 영역
st.subheader("📌 1. 전체 데이터 기준 선형회귀 모델")
c1, c2, c3, c4 = st.columns(4)
c1.metric("회귀선 기울기", f"{slope_all:+.5f} °C/년")
c2.metric("MAE", f"{mae_all:.4f}")
c3.metric("MSE", f"{mse_all:.4f}")
c4.metric("R²", f"{r2_all:.4f}")

st.divider()

st.subheader("📊 2. 공통 테스트 데이터(2006~2025년) 예측 성능 비교")
st.write(f"- **최근 50년 훈련 데이터:** {len(train_50)}개 연도 (1956~2005)")
st.write(f"- **최근 100년 훈련 데이터:** {len(train_100)}개 연도 (1906~2005)")
st.write(f"- **공통 테스트 데이터:** {len(test_20)}개 연도 (2006~2025)")

comp_df = pd.DataFrame({
    "학습 기간": ["최근 50년 학습 (1956~2005)", "최근 100년 학습 (1906~2005)"],
    "회귀선 기울기 (°C/년)": [f"{slope_50:+.5f}", f"{slope_100:+.5f}"],
    "MAE (낮을수록 우수)": [f"{mae_50:.4f}", f"{mae_100:.4f}"],
    "MSE (낮을수록 우수)": [f"{mse_50:.4f}", f"{mse_100:.4f}"],
    "R² (높을수록 우수)": [f"{r2_50:.4f}", f"{r2_100:.4f}"]
})
st.table(comp_df)

st.divider()

st.subheader("📉 3. 학습 기간별 회귀선 비교 그래프")
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=yearly_df['연도'], y=yearly_df['연평균기온'],
    mode='markers', name='전체 실제 데이터', marker=dict(color='gray', size=6, opacity=0.5)
))

fig.add_trace(go.Scatter(
    x=test_20['연도'], y=test_20['연평균기온'],
    mode='markers', name='테스트 데이터 (2006~2025)', marker=dict(color='red', size=8)
))

x_r50 = np.arange(1956, 2026).reshape(-1, 1)
fig.add_trace(go.Scatter(
    x=x_r50.flatten(), y=m_50.predict(x_r50),
    mode='lines', name=f'50년 학습 회귀선 ({slope_50:+.4f}°C/년)', line=dict(color='blue', width=3)
))

x_r100 = np.arange(1906, 2026).reshape(-1, 1)
fig.add_trace(go.Scatter(
    x=x_r100.flatten(), y=m_100.predict(x_r100),
    mode='lines', name=f'100년 학습 회귀선 ({slope_100:+.4f}°C/년)', line=dict(color='green', width=3, dash='dash')
))

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    template="plotly_white",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

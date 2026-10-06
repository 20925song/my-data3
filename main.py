import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="서울 기온 다항회귀 분석", page_icon="📈", layout="wide")

st.title("📈 서울 연평균기온 다항회귀(1차·3차·9차) 모델 비교")

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

# 1. 훈련용 / 테스트용 데이터 분할 (2005년 기준)
train_df = yearly_df[yearly_df['연도'] < 2005].copy()
test_df = yearly_df[yearly_df['연도'] >= 2005].copy()

# 데이터 개수 화면 출력
st.subheader("📌 1. 데이터 분할 정보")
col_tr, col_te = st.columns(2)
col_tr.metric("훈련용 데이터 개수 (2005년 이전)", f"{len(train_df)}개 연도", f"{train_df['연도'].min()} ~ {train_df['연도'].max()}")
col_te.metric("테스트용 데이터 개수 (2005년 이후)", f"{len(test_df)}개 연도", f"{test_df['연도'].min()} ~ {test_df['연도'].max()}")

# 2. 연도 스케일링 (수치 안정성을 위해 연도 차이값 사용)
base_year = train_df['연도'].min()
X_train = (train_df[['연도']].values - base_year) / 100.0
y_train = train_df['연평균기온'].values

X_test = (test_df[['연도']].values - base_year) / 100.0
y_test = test_df['연평균기온'].values

X_2050 = np.array([[(2050 - base_year) / 100.0]])

# 3. 모델 학습 및 테스트 데이터 평가
degrees = [1, 3, 9]
results = []
models = {}

for deg in degrees:
    model = make_pipeline(PolynomialFeatures(degree=deg), LinearRegression())
    model.fit(X_train, y_train)
    models[deg] = model
    
    # 테스트 데이터 예측 및 오차 평가
    y_pred_test = model.predict(X_test)
    pred_2050 = model.predict(X_2050)[0]
    
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    results.append({
        "차수": f"{deg}차 모델",
        "테스트 오차 (MAE)": f"{mae:.2f} °C",
        "테스트 오차 (RMSE)": f"{rmse:.2f} °C",
        "2050년 예상 기온": f"{pred_2050:.2f} °C"
    })

st.divider()
st.subheader("📊 2. 차수별 테스트 성능 및 2050년 예측 비교표")
st.table(pd.DataFrame(results))

# 4. 시각화 그래프
st.divider()
st.subheader("📉 3. 회귀 곡선 및 2050년 예측 시각화")

fig = go.Figure()

# 실제 데이터 점 표시
fig.add_trace(go.Scatter(
    x=train_df['연도'], y=train_df['연평균기온'],
    mode='markers', name='훈련 데이터 (< 2005)', marker=dict(color='blue', opacity=0.6, size=6)
))
fig.add_trace(go.Scatter(
    x=test_df['연도'], y=test_df['연평균기온'],
    mode='markers', name='테스트 데이터 (≥ 2005)', marker=dict(color='red', size=8)
))

# 1908년~2050년까지의 예측 라인 생성
plot_years = np.linspace(train_df['연도'].min(), 2050, 300).reshape(-1, 1)
plot_X = (plot_years - base_year) / 100.0

colors = {1: 'green', 3: 'orange', 9: 'purple'}
for deg in degrees:
    pred_y = models[deg].predict(plot_X)
    fig.add_trace(go.Scatter(
        x=plot_years.flatten(), y=pred_y,
        mode='lines', name=f'{deg}차 곡선 모델', line=dict(color=colors[deg], width=2)
    ))

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    yaxis=dict(range=[5, 25]),
    template="plotly_white",
    height=550
)

st.plotly_chart(fig, use_container_width=True)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="서울 기온 선형회귀 모델 평가", page_icon="📈", layout="wide")

st.title("📈 서울 연평균 기온 선형회귀 모델 평가 및 비교")

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

# 훈련/테스트 데이터 분할
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

# 2. 최근 50년 학습 (1956~2005) -> 테스트 (2006~2025)
X_tr50, y_tr50 = train_50[['연도']].values, train_50['연평균기온'].values
X_te20, y_te20 = test_20[['연도']].values, test_20['연평균기온'].values

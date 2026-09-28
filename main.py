import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="서울 기온 예측기", layout="centered")

st.title("🌡️ 서울 연평균 기온 예측기")

# 데이터 불러오기
URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_and_process_data(url):
    df = pd.read_csv(url, encoding="utf-8")
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year

    # 연도별 관측일수 및 평균기온 계산
    yearly = df.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    ).reset_index()

    # 필터링: 2025년 이하 & 관측일수 300일 이상
    filtered = yearly[(yearly["연도"] <= 2025) & (yearly["관측일수"] >= 300)].copy()
    
    # 1908년 기준 경과 연수 계산 (독립 변수 X)
    filtered["지난연수"] = filtered["연도"] - 1908
    return filtered

try:
    df_filtered = load_and_process_data(URL)

    # 데이터 요약 정보
    num_years = len(df_filtered)
    start_year = int(df_filtered["연도"].min())
    end_year = int(df_filtered["연도"].max())

    # 1. 전체 기간 회귀 직선 계산
    X_full = df_filtered["지난연수"]
    y_full = df_filtered["연평균기온"]
    slope_full, intercept_full, r_full, _, _ = stats.linregress(X_full, y_full)

    # 2. 최근 20년 회귀 직선 계산
    recent_20_start = end_year - 19
    df_recent = df_filtered[df_filtered["연도"] >= recent_20_start]
    X_recent = df_recent["지난연수"]
    y_recent = df_recent["연평균기온"]
    slope_recent, intercept_recent, r_recent, _, _ = stats.linregress(X_recent, y_recent)

    st.markdown(f"**학습 데이터 정보:** 총 **{num_years}개** 연도 ({start_year}년 ~ {end_year}년)")

    # 100년당 기온 상승량 계산
    century_rate_full = slope_full * 100
    century_rate_recent = slope_recent * 100

    # 기울기 나란히 비교
    st.subheader("🔥 기온 상승 속도 비교 (100년당)")
    col_rate1, col_rate2 = st.columns(2)
    with col_rate1:
        st.metric(
            label=f"🌐 전체 기간 ({start_year}~{end_year}년)",
            value=f"+{century_rate_full:.2f} °C / 100년",
            help=f"상관계수 r = {r_full:.4f}"
        )
    with col_rate2:
        diff_rate = century_rate_recent - century_rate_full
        st.metric(
            label=f"⚡ 최근 20년 ({recent_20_start}~{end_year}년)",
            value=f"+{century_rate_recent:.2f} °C / 100년",
            delta=f"전체 대비 +{diff_rate:.2f} °C 더 빠름" if diff_rate > 0 else f"전체 대비 {diff_rate:.2f} °C",
            help=f"상관계수 r = {r_recent:.4f}"
        )

    st.divider()

    # 연도 선택 슬라이더 (1900년 ~ 2100년)
    selected_year = st.slider("예측할 연도를 선택하세요", min_value=1900, max_value=2100, value=2026)

    # 선택된 연도 기온 예측 (전체 기간 모델 기준)
    pred_passed_years = selected_year - 1908
    predicted_temp_full = slope_full * pred_passed_years + intercept_full
    predicted_temp_recent = slope_recent * pred_passed_years + intercept_recent

    # 예측 결과 크게 표시
    col_pred1, col_pred2 = st.columns(2)
    with col_pred1:
        st.metric(
            label=f"🎯 {selected_year}년 예상 기온 (전체 기간 모델)",
            value=f"{predicted_temp_full:.2f} °C"
        )
    with col_pred2:
        st.metric(
            label=f"📈 {selected_year}년 예상 기온 (최근 20년 추세 반영)",
            value=f"{predicted_temp_recent:.2f} °C"
        )

    # 회귀선 연도 범위 설정
    years_range = np.arange(start_year, 2101)
    passed_range = years_range - 1908
    
    reg_line_full = slope_full * passed_range + intercept_full
    reg_line_recent = slope_recent * passed_range + intercept_recent

    # Plotly 시각화
    fig = go.Figure()

    # 관측 데이터 산점도
    fig.add_trace(go.Scatter(
        x=df_filtered["연도"],
        y=df_filtered["연평균기온"],
        mode="markers",
        name="관측 데이터",
        marker=dict(color="royalblue", size=6, opacity=0.7),
        hovertemplate="연도: %{x}년<br>평균기온: %{y:.2f}°C<extra></extra>"
    ))

    # 전체 기간 회귀선
    fig.add_trace(go.Scatter(
        x=years_range,
        y=reg_line_full,
        mode="lines",
        name=f"전체 기간 회귀선 (r={r_full:.2f})",
        line=dict(color="firebrick", width=2),
        hovertemplate="연도: %{x}년<br>전체기준 예상: %{y:.2f}°C<extra></extra>"
    ))

    # 최근 20년 회귀선
    fig.add_trace(go.Scatter(
        x=years_range,
        y=reg_line_recent,
        mode="lines",
        name=f"최근 20년 회귀선 (r={r_recent:.2f})",
        line=dict(color="darkorange", width=2, dash="dash"),
        hovertemplate="연도: %{x}년<br>최근20년기준 예상: %{y:.2f}°C<extra></extra>"
    ))

    # 선택한 연도 포인트 강조 (전체 기간 모델)
    fig.add_trace(go.Scatter(
        x=[selected_year],
        y=[predicted_temp_full],
        mode="markers",
        name=f"선택 연도 예측점 ({selected_year}년)",
        marker=dict(color="green", size=12, symbol="star"),
        hovertemplate="선택 연도: %{x}년<br>예상기온: %{y:.2f}°C<extra></extra>"
    ))

    fig.update_layout(
        title="서울 연평균 기온 추이 및 회귀선 비교",
        xaxis_title="연도",
        yaxis_title="연평균 기온 (°C)",
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"데이터를 불러오거나 처리하는 중 오류가 발생했습니다: {e}")

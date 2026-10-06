import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="서울 100년 기온 변화 분석",
    page_icon="🌡️",
    layout="wide"
)

# 제목 및 설명
st.title("🌡️ 서울 100년 연평균 기온 변화 분석")
st.markdown("""
지난 100여 년간 서울의 연평균 기온 상승 추이를 한눈에 확인할 수 있는 대시보드입니다.  
* **데이터 출처:** 기상청 기후자료 (`seoul.csv`)
""")

# 데이터 로드 함수 (캐싱 적용)
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/seoul.csv"
    try:
        df = pd.read_csv(url, encoding='cp949')
    except Exception:
        df = pd.read_csv(url, encoding='utf-8')
    
    # 열 이름 공백 및 특수문자 정제
    df.columns = df.columns.str.strip()
    
    col_map = {}
    for col in df.columns:
        if '날짜' in col:
            col_map[col] = '날짜'
        elif '평균기온' in col:
            col_map[col] = '평균기온'
        elif '최저기온' in col:
            col_map[col] = '최저기온'
        elif '최고기온' in col:
            col_map[col] = '최고기온'
    
    df = df.rename(columns=col_map)
    
    # 날짜 및 기온 데이터 정제
    df['날짜'] = pd.to_datetime(df['날짜'].astype(str).str.strip(), errors='coerce')
    df = df.dropna(subset=['날짜'])
    df['연도'] = df['날짜'].dt.year
    
    for col in ['평균기온', '최저기온', '최고기온']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

with st.spinner("데이터를 불러오는 중입니다..."):
    raw_df = load_data()

# 연도별 집계 (데이터가 300일 이상 존재하는 연도만 집계)
valid_years = raw_df.groupby('연도')['평균기온'].count()
valid_years = valid_years[valid_years >= 300].index

yearly_df = raw_df[raw_df['연도'].isin(valid_years)].groupby('연도').agg(
    연평균기온=('평균기온', 'mean'),
    연최저기온평균=('최저기온', 'mean'),
    연최고기온평균=('최고기온', 'mean')
).reset_index()

# 사이드바 설정
st.sidebar.header("⚙️ 분석 옵션")

min_year = int(yearly_df['연도'].min())
max_year = int(yearly_df['연도'].max())

year_range = st.sidebar.slider(
    "조회 연도 범위",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

show_trendline = st.sidebar.checkbox("장기 추세선 (선형 회귀)", value=True)
show_ma = st.sidebar.checkbox("10년 이동평균선", value=True)

# 필터링 및 이동평균 계산
filtered_df = yearly_df[(yearly_df['연도'] >= year_range[0]) & (yearly_df['연도'] <= year_range[1])].copy()
filtered_df['10년이동평균'] = filtered_df['연평균기온'].rolling(window=10, min_periods=1).mean()

# 주요 지표 카드
col1, col2, col3, col4 = st.columns(4)

highest_row = filtered_df.loc[filtered_df['연평균기온'].idxmax()]
lowest_row = filtered_df.loc[filtered_df['연평균기온'].idxmin()]
temp_change = filtered_df['연평균기온'].iloc[-1] - filtered_df['연평균기온'].iloc[0]

with col1:
    st.metric("최고 연평균 기온", f"{highest_row['연평균기온']:.1f} °C", f"{int(highest_row['연도'])}년")
with col2:
    st.metric("최저 연평균 기온", f"{lowest_row['연평균기온']:.1f} °C", f"{int(lowest_row['연도'])}년")
with col3:
    st.metric("선택 기간 평균", f"{filtered_df['연평균기온'].mean():.1f} °C")
with col4:
    st.metric("기간 내 총 기온 변화", f"{temp_change:+.1f} °C")

# 시각화 그래프
st.subheader("📈 연도별 서울 평균 기온 변화")

fig = go.Figure()

# 연평균 기온 그래프
fig.add_trace(go.Scatter(
    x=filtered_df['연도'],
    y=filtered_df['연평균기온'],
    mode='lines+markers',
    name='연평균 기온',
    line=dict(color='#E74C3C', width=2),
    marker=dict(size=4),
    hovertemplate='<b>%{x}년</b><br>평균 기온: %{y:.2f} °C<extra></extra>'
))

# 10년 이동평균선
if show_ma:
    fig.add_trace(go.Scatter(
        x=filtered_df['연도'],
        y=filtered_df['10년이동평균'],
        mode='lines',
        name='10년 이동평균',
        line=dict(color='#2980B9', width=3, dash='dash'),
        hovertemplate='<b>%{x}년 (10년 이동평균)</b><br>기온: %{y:.2f} °C<extra></extra>'
    ))

# 장기 추세선
if show_trendline and len(filtered_df) > 1:
    z = np.polyfit(filtered_df['연도'], filtered_df['연평균기온'], 1)
    p = np.poly1d(z)
    
    fig.add_trace(go.Scatter(
        x=filtered_df['연도'],
        y=p(filtered_df['연도']),
        mode='lines',
        name='장기 추세선',
        line=dict(color='#27AE60', width=2, dash='dot'),
        hovertemplate='추세값: %{y:.2f} °C<extra></extra>'
    ))

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="기온 (°C)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# 데이터 상세 보기
st.subheader("📋 상세 데이터")
tab1, tab2 = st.tabs(["데이터 테이블", "통계 요약"])

with tab1:
    st.dataframe(
        filtered_df[['연도', '연평균기온', '연최저기온평균', '연최고기온평균']]
        .style.format({'연평균기온': '{:.2f}', '연최저기온평균': '{:.2f}', '연최고기온평균': '{:.2f}'}),
        use_container_width=True
    )

with tab2:
    st.write(
        filtered_df[['연평균기온', '연최저기온평균', '연최고기온평균']]
        .describe()
        .rename(columns={
            '연평균기온': '연평균 기온(°C)',
            '연최저기온평균': '연최저 기온 평균(°C)',
            '연최고기온평균': '연최고 기온 평균(°C)'
        })
    )

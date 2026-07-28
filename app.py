import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import plotly.graph_objects as go

# ---------------------------------------------------------
# 1. 페이지 기본 설정 및 토스(Toss) 스타일 커스텀 CSS 주입
# ---------------------------------------------------------
st.set_page_config(page_title="보험 스카우팅 리포트", layout="wide")

st.markdown("""
<style>
    /* Pretendard 폰트 불러오기 (토스 대표 폰트) */
    @import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css");
    
    * {
        font-family: "Pretendard Variable", Pretendard, -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif !important;
    }

    /* 전체 배경을 토스 특유의 차분한 라이트 그레이로 */
    .stApp {
        background-color: #F2F4F6;
        color: #191F28;
    }

    /* 사이드바 스타일링 */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E8EB;
    }

    /* 토스 스타일 흰색 라운드 카드 디자인 */
    .toss-card {
        background-color: #FFFFFF;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
        border: 1px solid #F2F4F6;
    }

    /* 토스 블루 강조 카드 (베스트 추천용) */
    .toss-card-primary {
        background-color: #E8F3FF;
        border: 2px solid #3182F6;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 8px 20px rgba(49, 130, 246, 0.12);
        margin-bottom: 20px;
    }

    /* 배지(Badge) 태그 */
    .toss-badge {
        display: inline-block;
        background-color: #3182F6;
        color: #FFFFFF;
        font-weight: 700;
        font-size: 13px;
        padding: 4px 10px;
        border-radius: 8px;
        margin-bottom: 12px;
    }

    .toss-badge-gray {
        display: inline-block;
        background-color: #F2F4F6;
        color: #6B7684;
        font-weight: 600;
        font-size: 13px;
        padding: 4px 10px;
        border-radius: 8px;
        margin-bottom: 12px;
    }

    /* 타이틀 및 서브 타이틀 폰트 서식 */
    .toss-title {
        font-size: 24px;
        font-weight: 700;
        color: #191F28;
        margin-bottom: 8px;
    }

    .toss-price {
        font-size: 28px;
        font-weight: 800;
        color: #3182F6;
        margin: 8px 0;
    }

    .toss-price-gray {
        font-size: 28px;
        font-weight: 800;
        color: #333D4B;
        margin: 8px 0;
    }

    /* 스탯 리스트 라인 */
    .stat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid #F2F4F6;
        font-size: 15px;
    }

    .stat-label {
        color: #6B7684;
        font-weight: 500;
    }

    .stat-value {
        font-weight: 700;
        color: #191F28;
    }

    .stat-value-highlight {
        font-weight: 700;
        color: #3182F6;
    }

    /* 이유 설명 박스 */
    .reason-box {
        background-color: #F9FAFB;
        border-radius: 12px;
        padding: 14px;
        margin-top: 16px;
        font-size: 14px;
        color: #4E5968;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. 헤더 섹션
# ---------------------------------------------------------
st.markdown("""
<div style="margin-bottom: 24px;">
    <h1 style="font-size: 32px; font-weight: 800; color: #191F28; margin-bottom: 8px;">
        내 조건에 딱 맞는 보험사 찾기
    </h1>
    <p style="font-size: 16px; color: #6B7684;">
        머신러닝이 각 보험사의 비공개 계리 가중치를 분석해 20-80 스카우팅 스탯으로 알려드려요.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. 사이드바 사용자 입력
# ---------------------------------------------------------
st.sidebar.markdown("### 👤 프로필 설정")
user_age = st.sidebar.slider("나이", 20, 70, 35)
user_cancer_history = st.sidebar.selectbox("암/질병 이력", ["없음", "3년 이내", "5년 경과"])
user_tmap_score = st.sidebar.slider("운전 점수 (Tmap)", 0, 100, 85)
user_annual_km = st.sidebar.slider("연간 주행거리 (km)", 3000, 30000, 10000)

cancer_numeric = 0 if user_cancer_history == "없음" else (2 if user_cancer_history == "3년 이내" else 1)

# ---------------------------------------------------------
# 4. 데이터 및 머신러닝 학습
# ---------------------------------------------------------
@st.cache_data
def generate_data():
    np.random.seed(42)
    age = np.random.randint(20, 70, 2000)
    cancer = np.random.choice([0, 1, 2], 2000, p=[0.7, 0.2, 0.1])
    tmap = np.random.randint(30, 100, 2000)
    km = np.random.randint(3000, 25000, 2000)
    
    prem_A = 500000 + (age * 7000) + (cancer * 280000) - (tmap * 600) + (km * 10)
    prem_B = 460000 + (age * 8000) + (cancer * 80000) - (tmap * 2200) + (km * 8)
    prem_C = 390000 + (age * 11000) + (cancer * 160000) - (tmap * 800) + (km * 15)
    
    return pd.DataFrame({'age': age, 'cancer': cancer, 'tmap': tmap, 'km': km, 'prem_A': prem_A, 'prem_B': prem_B, 'prem_C': prem_C})

df = generate_data()

@st.cache_resource
def train_models():
    X = df[['age', 'cancer', 'tmap', 'km']]
    models, explainers = {}, {}
    for comp in ['A', 'B', 'C']:
        m = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
        m.fit(X, df[f'prem_{comp}'])
        models[comp], explainers[comp] = m, shap.TreeExplainer(m)
    return models, explainers

models, explainers = train_models()

user_prof = {'age': user_age, 'cancer': cancer_numeric, 'tmap': user_tmap_score, 'km': user_annual_km}
X_user = pd.DataFrame([user_prof])
X_all = df[['age', 'cancer', 'tmap', 'km']]

categories = ['유병자 용인력', '연령 방어력', '운전특약 가성비', '주행거리 우대']
stats_20_80 = {}

for comp in ['A', 'B', 'C']:
    u_shap = explainers[comp].shap_values(X_user)[0]
    a_shap = explainers[comp].shap_values(X_all)
    scores = []
    for i in range(4):
        z = (-u_shap[i] - np.mean(-a_shap[:, i])) / (np.std(-a_shap[:, i]) + 1e-5)
        scores.append(int(np.clip(round(50 + (z * 10)), 20, 80)))
    stats_20_80[comp] = scores

prems = {comp: int(models[comp].predict(X_user)[0]) for comp in ['A', 'B', 'C']}
sorted_prems = sorted(prems.items(), key=lambda x: x[1])
best_comp, min_prem = sorted_prems[0]
worst_comp, max_prem = sorted_prems[-1]

gap_worst = max_prem - min_prem

# ---------------------------------------------------------
# 5. 토스 스타일 상단 하이라이트 배너
# ---------------------------------------------------------
st.markdown(f"""
<div style="background-color: #3182F6; color: white; border-radius: 20px; padding: 24px; margin-bottom: 32px; box-shadow: 0 8px 24px rgba(49, 130, 246, 0.25);">
    <div style="font-size: 15px; font-weight: 600; opacity: 0.9;">추천 결과 요약</div>
    <div style="font-size: 26px; font-weight: 800; margin: 8px 0;">
        {best_comp}보험사를 선택하면 <br/>가장 비싼 곳보다 <span style="color: #FFD15C;">월 {gap_worst:,}원</span> 아낄 수 있어요
    </div>
    <div style="font-size: 14px; opacity: 0.8; margin-top: 4px;">
        1년 기준으로 환산하면 총 {gap_worst*12:,}원을 절약하게 됩니다.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 6. 토스 스타일 카드 3열 비교 (HTML 줄바꿈 에러 수정본)
# ---------------------------------------------------------
col_a, col_b, col_c = st.columns(3)
cols = {'A': col_a, 'B': col_b, 'C': col_c}

def generate_reason(comp, scores, user_prof):
    reasons = []
    if user_prof['cancer'] > 0:
        if scores[0] >= 60:
            reasons.append("질병 이력 패널티가 적음")
        else:
            reasons.append("질병 이력 할증률이 높음")
    if scores[1] <= 40:
        reasons.append("해당 연령대 요율이 비쌈")
    else:
        reasons.append("연령대 방어력이 우수함")
    if user_prof['tmap'] >= 80:
        if scores[2] >= 60:
            reasons.append("Tmap 우대 할인 폭이 큼")
        else:
            reasons.append("Tmap 할인이 박한 편임")
            
    return " · ".join(reasons)

for comp in ['A', 'B', 'C']:
    with cols[comp]:
        is_best = (comp == best_comp)
        card_class = "toss-card-primary" if is_best else "toss-card"
        badge_html = '<div class="toss-badge">가장 추천해요</div>' if is_best else '<div class="toss-badge-gray">비교 추천</div>'
        price_class = "toss-price" if is_best else "toss-price-gray"
        diff_text = "최저가" if is_best else f"{prems[comp] - min_prem:+,}원"
        
        # HTML 공백 및 줄바꿈을 단일 문장으로 결합하여 렌더링 오류 방지
        card_html = f"""
<div class="{card_class}">
{badge_html}
<div class="toss-title">{comp} 보험사</div>
<div class="{price_class}">월 {prems[comp]:,}원</div>
<div style="font-size: 13px; color: #8B95A1; margin-bottom: 20px;">최저가 대비: <b>{diff_text}</b></div>
<div style="font-weight: 700; font-size: 14px; color: #333D4B; margin-bottom: 12px;">20-80 스카우팅 스탯</div>
<div class="stat-row"><span class="stat-label">유병자 용인력</span><span class="stat-value">{stats_20_80[comp][0]} <span style="font-size:12px; color:#B0B8C1;">/ 80</span></span></div>
<div class="stat-row"><span class="stat-label">연령 방어력</span><span class="stat-value">{stats_20_80[comp][1]} <span style="font-size:12px; color:#B0B8C1;">/ 80</span></span></div>
<div class="stat-row"><span class="stat-label">운전특약 가성비</span><span class="stat-value">{stats_20_80[comp][2]} <span style="font-size:12px; color:#B0B8C1;">/ 80</span></span></div>
<div class="stat-row"><span class="stat-label">주행거리 우대</span><span class="stat-value">{stats_20_80[comp][3]} <span style="font-size:12px; color:#B0B8C1;">/ 80</span></span></div>
<div class="reason-box"><b>💡 추천 사유 분석</b><br/>{generate_reason(comp, stats_20_80[comp], user_prof)}</div>
</div>
"""
        st.markdown(card_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# 7. 하단 레이더 차트 (토스 톤앤매너 적용)
# ---------------------------------------------------------
st.markdown("<br/>", unsafe_allow_html=True)
st.markdown("""
<div style="background-color: #FFFFFF; border-radius: 20px; padding: 28px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);">
    <div style="font-size: 20px; font-weight: 700; color: #191F28; margin-bottom: 16px;">
        한눈에 보는 20-80 스탯 오버레이
    </div>
""", unsafe_allow_html=True)

fig = go.Figure()
colors = {'A': '#FF5F5F', 'B': '#3182F6', 'C': '#00B894'}

for comp in ['A', 'B', 'C']:
    fig.add_trace(go.Scatterpolar(
        r=stats_20_80[comp] + [stats_20_80[comp][0]],
        theta=categories + [categories[0]],
        fill='toself', name=f'{comp}사 ({prems[comp]:,}원)', line_color=colors[comp], opacity=0.35
    ))

fig.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[20, 80], gridcolor="#E5E8EB"),
        angularaxis=dict(gridcolor="#E5E8EB")
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=450,
    margin=dict(l=40, r=40, t=20, b=20)
)

st.plotly_chart(fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)
# ---------------------------------------------------------
# 8. 대시보드 하단: 계리적 가정 역산 원리 및 SHAP 세부 기여도 원화(₩) 공개
# ---------------------------------------------------------
st.markdown("<br/>", unsafe_allow_html=True)

with st.expander("🔍 AI는 어떻게 보험사의 비공개 계리적 가정을 역산했나요? (원리 보기)"):
    st.markdown("""
    ### ⚙️ 계리적 가정 역산(Actuarial Reverse-Engineering) 매커니즘
    보험사는 독자적인 계리적 가정(위험률, 손해율 등)을 영업 비밀로 보호하지만, **소비자 조건에 따른 '최종 보험료'는 시장에 공개**됩니다.
    본 시스템은 **XAI(Explainable AI - SHAP)** 기술을 활용해 블랙박스인 보험료 산출 로직을 역추적합니다.
    """)
    
    st.markdown("---")
    st.markdown("#### 💵 현재 입력 프로필 기준 보험사별 요율 할증/할인 원화(₩) 분석")
    st.caption("아래 표는 AI가 분석한 '내 조건 때문에 각 보험사에서 실제로 더해지거나 깎인 금액'입니다.")
    
    # SHAP 원본 값(원화 금액) 추출 및 표 작성
    shap_detail_rows = []
    
    for comp in ['A', 'B', 'C']:
        u_shap = explainers[comp].shap_values(X_user)[0]
        # u_shap[0]: 나이, u_shap[1]: 질병, u_shap[2]: Tmap, u_shap[3]: 주행거리
        
        shap_detail_rows.append({
            "보험사": f"{comp} 보험사",
            "질병 이력 영향": f"{int(u_shap[1]):+,}원",
            "나이/연령 영향": f"{int(u_shap[0]):+,}원",
            "Tmap 운전점수 영향": f"{int(u_shap[2]):+,}원",
            "주행거리 영향": f"{int(u_shap[3]):+,}원",
            "최종 예상 보험료": f"{prems[comp]:,}원"
        })
        
    df_shap_detail = pd.DataFrame(shap_detail_rows)
    st.dataframe(df_shap_detail, use_container_width=True, hide_index=True)
    
    st.info("""
    💡 **해석 가이드**:
    * **`+` 금액 (레드)**: 내 프로필 특성 때문에 해당 보험사에서 **보험료가 더 비싸진 금액(할증)**
    * **`-` 금액 (블루)**: 내 프로필 특성 덕분에 해당 보험사에서 **보험료가 깎인 금액(할인)**
    * 이 원화(₩) 단위의 영향력을 시장 전체 평균과 비교하여 **`20~80점 야구 스탯`**으로 최종 변환한 것입니다.
    """)
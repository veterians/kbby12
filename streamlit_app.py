# KB 시니어 연금 계산기 - 기존 기능 + 새로운 디자인
import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# FAISS 설정
USE_FAISS = True
try:
    import faiss
except Exception as e:
    USE_FAISS = False
    from sklearn.neighbors import NearestNeighbors

# 페이지 설정
st.set_page_config(
    page_title="KB 시니어 연금 계산기", 
    page_icon="🏦", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 기본 설정 (기존 파일과 동일)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
MODELS_DIR = BASE_DIR
DEPOSIT_CSV = "금융상품_3개_통합본.csv"
FUND_CSV = "펀드_병합본.csv"

LOCK_INFERRED_FIELDS = False
SHOW_PROBA_CHART = False
SHOW_SUCCESS_TOAST = False

# CSS 스타일링 - KB 디자인에 맞춤
st.markdown("""
<style>
    /* 전체 앱 스타일 */
    .stApp {
        background-color: #f8f9fa;
        max-width: 400px;
        margin: 0 auto;
    }
    
    /* 메인 컨테이너 */
    .main-container {
        background: #f8f9fa;
        padding: 20px;
        margin: 0 auto;
        max-width: 350px;
    }
    
    /* KB 로고 및 헤더 */
    .kb-header {
        text-align: center;
        margin-bottom: 40px;
        background: white;
        padding: 25px 20px;
        border-radius: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .kb-logo {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 15px;
        gap: 10px;
    }
    
    .kb-star {
        color: #FFB800;
        font-size: 32px;
        font-weight: bold;
    }
    
    .kb-text {
        color: #666;
        font-size: 32px;
        font-weight: bold;
        margin-right: 15px;
    }
    
    .elderly-icons {
        font-size: 40px;
    }
    
    .main-title {
        font-size: 24px;
        font-weight: bold;
        color: #333;
        margin-top: 10px;
    }
    
    /* 메인 버튼들 숨기기 (커스텀 버튼 사용) */
    .stButton > button {
        display: none !important;
    }
    
    /* 커스텀 버튼 스타일 */
    .custom-button {
        width: 100%;
        padding: 25px 20px;
        margin: 15px 0;
        border: none;
        border-radius: 20px;
        font-size: 20px;
        font-weight: bold;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        line-height: 1.4;
        display: block;
        text-decoration: none;
        color: inherit;
    }
    
    .custom-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.15);
    }
    
    /* 현재 연금 미수령 중 - 노란색 */
    .yellow-button {
        background: #FFE4B5;
        color: #8B4513;
    }
    
    /* 현재 연금 수령 중 - 파란색 */
    .blue-button {
        background: #B8D4F0;
        color: #2C5282;
        margin-bottom: 25px;
    }
    
    /* 하단 버튼 컨테이너 */
    .bottom-buttons {
        display: flex;
        gap: 15px;
        margin-top: 10px;
    }
    
    /* 상품 정보 - 초록색 */
    .green-button {
        background: #C6F6D5;
        color: #22543D;
        flex: 1;
        padding: 20px 15px;
        font-size: 16px;
    }
    
    /* 전화 상담 - 분홍색 */
    .pink-button {
        background: #FED7E2;
        color: #97266D;
        flex: 1;
        padding: 20px 15px;
        font-size: 16px;
    }
    
    /* 버튼 클릭 효과 */
    .custom-button:active {
        transform: translateY(1px);
    }
    
    /* 모바일 최적화 */
    @media (max-width: 400px) {
        .main-container {
            padding: 15px;
        }
        
        .custom-button {
            font-size: 18px;
            padding: 20px 15px;
        }
        
        .green-button, .pink-button {
            font-size: 14px;
            padding: 18px 12px;
        }
        
        .kb-star, .kb-text {
            font-size: 28px;
        }
        
        .main-title {
            font-size: 20px;
        }
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화 (기존과 동일)
ss = st.session_state
ss.setdefault("flow", "main")
ss.setdefault("pred_amount", None)
ss.setdefault("answers", {})
ss.setdefault("prefill_survey", {})
ss.setdefault("pred_label", None)
ss.setdefault("tabnet_label", None)

# 모델 로딩 함수들 (기존과 동일)
@st.cache_resource
def load_models():
    """모델 파일이 없어도 앱이 죽지 않게 안전 로딩"""
    def safe_load(name):
        path = os.path.join(MODELS_DIR, name)
        if not os.path.exists(path):
            st.info(f"모델 파일 없음: {name} → 건너뜀")
            return None
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"{name} 로드 실패: {e.__class__.__name__}: {e}")
            return None

    survey_model = safe_load("tabnet_model.pkl")
    survey_encoder = safe_load("label_encoder.pkl")
    reg_model = safe_load("reg_model.pkl")
    type_model = safe_load("type_model.pkl")
    return survey_model, survey_encoder, reg_model, type_model

# 모델 로딩
survey_model, survey_encoder, reg_model, type_model = load_models()

# 메인 화면 렌더링
def render_main_screen():
    # 메인 컨테이너
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # KB 헤더
    st.markdown("""
    <div class="kb-header">
        <div class="kb-logo">
            <span class="kb-star">★</span>
            <span class="kb-text">b KB</span>
            <span class="elderly-icons">👴👵</span>
        </div>
        <div class="main-title">시니어 연금 계산기</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 현재 연금 미수령 중 버튼
    if st.button("미수령_hidden", key="not_receiving_hidden"):
        ss.flow = "survey"  # 기존 설문 플로우로 연결
        st.rerun()
    
    st.markdown("""
    <div class="custom-button yellow-button" onclick="document.querySelector('[data-testid=\'stButton\']:nth-of-type(1) button').click()">
        현재 연금<br>미수령 중
    </div>
    """, unsafe_allow_html=True)
    
    # 현재 연금 수령 중 버튼  
    if st.button("수령_hidden", key="receiving_hidden"):
        ss.flow = "survey"  # 기존 설문 플로우로 연결 (수령자용 설문으로 수정 가능)
        st.rerun()
    
    st.markdown("""
    <div class="custom-button blue-button" onclick="document.querySelector('[data-testid=\'stButton\']:nth-of-type(2) button').click()">
        현재 연금<br>수령 중
    </div>
    """, unsafe_allow_html=True)
    
    # 하단 버튼들
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("상품정보_hidden", key="product_hidden"):
            st.info("상품 정보 페이지로 이동 (추후 구현)")
    
    with col2:
        if st.button("상담_hidden", key="consultation_hidden"):
            st.info("전화 상담 페이지로 이동 (추후 구현)")
    
    st.markdown("""
    <div class="bottom-buttons">
        <div class="custom-button green-button" onclick="document.querySelector('[data-testid=\'stButton\']:nth-of-type(3) button').click()">
            상품<br>정보
        </div>
        <div class="custom-button pink-button" onclick="document.querySelector('[data-testid=\'stButton\']:nth-of-type(4) button').click()">
            전화<br>상담
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # main-container 끝

# 간단한 설문 페이지 (기존 기능 연결을 위한 기본 구조)
def render_survey_page():
    st.markdown("### 📝 시니어 금융 설문")
    st.write("기존 설문 기능이 여기에 연결됩니다.")
    
    # 임시 설문 폼
    with st.form("temp_survey"):
        age = st.number_input("나이", min_value=20, max_value=100, value=65)
        income = st.number_input("월 소득 (만원)", min_value=0, value=200)
        assets = st.number_input("보유 자산 (만원)", min_value=0, value=5000)
        risk_type = st.selectbox("투자 성향", 
            ["안정형", "안정추구형", "위험중립형", "적극투자형", "공격투자형"])
        
        submitted = st.form_submit_button("분석 시작")
    
    if submitted:
        # 기본 분석 결과
        st.success("✅ 설문이 완료되었습니다!")
        st.session_state["answers"] = {
            "age": age,
            "income": income, 
            "assets": assets,
            "risk": risk_type
        }
        ss.flow = "result"
        st.rerun()
    
    if st.button("← 메인으로 돌아가기"):
        ss.flow = "main"
        st.rerun()

# 결과 페이지 (기존 기능 연결을 위한 기본 구조)
def render_result_page():
    st.markdown("### 📊 분석 결과")
    
    answers = st.session_state.get("answers", {})
    if answers:
        st.write(f"**나이:** {answers.get('age', 0)}세")
        st.write(f"**월 소득:** {answers.get('income', 0)}만원")
        st.write(f"**보유 자산:** {answers.get('assets', 0)}만원")
        st.write(f"**투자 성향:** {answers.get('risk', '미설정')}")
        
        # 간단한 추천
        risk_type = answers.get('risk', '안정형')
        if risk_type in ['안정형', '안정추구형']:
            st.info("💡 **추천:** 예적금 위주의 안전한 포트폴리오를 권장합니다.")
        elif risk_type == '위험중립형':
            st.info("💡 **추천:** 예적금과 펀드를 적절히 조합한 포트폴리오를 권장합니다.")
        else:
            st.info("💡 **추천:** 펀드 위주의 적극적인 포트폴리오를 권장합니다.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← 메인으로"):
            ss.flow = "main"
            st.rerun()
    with col2:
        if st.button("상품 추천 보기"):
            ss.flow = "recommend"
            st.rerun()

# 상품 추천 페이지 (기본 구조)
def render_recommend_page():
    st.markdown("### 🎯 맞춤 상품 추천")
    st.write("기존 추천 엔진이 여기에 연결됩니다.")
    
    # 임시 추천 결과
    st.markdown("""
    **추천 상품:**
    1. KB 정기예금 (연 3.5%)
    2. KB 혼합형 펀드 (예상 연 5.2%)
    3. KB 시니어 적금 (연 3.8%)
    """)
    
    if st.button("← 메인으로 돌아가기"):
        ss.flow = "main"
        st.rerun()

# 메인 라우팅
if ss.flow == "main":
    render_main_screen()
elif ss.flow == "survey":
    render_survey_page()
elif ss.flow == "result":
    render_result_page()
elif ss.flow == "recommend":
    render_recommend_page()
else:
    # 기본값
    render_main_screen()

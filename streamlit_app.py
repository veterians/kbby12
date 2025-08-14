import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# 기존 코드의 필요한 부분들 통합
try:
    import faiss
    USE_FAISS = True
except:
    USE_FAISS = False
    from sklearn.neighbors import NearestNeighbors

# 페이지 설정
st.set_page_config(
    page_title="KB 시니어 연금 계산기",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =================================
# 데이터 로딩 및 모델 관련 함수들
# =================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

@st.cache_resource
def load_models():
    """모델 파일이 없어도 앱이 죽지 않게 안전 로딩"""
    def safe_load(name):
        path = os.path.join(BASE_DIR, name)
        if not os.path.exists(path):
            return None
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"{name} 로드 실패: {e}")
            return None

    survey_model = safe_load("tabnet_model.pkl")
    survey_encoder = safe_load("label_encoder.pkl")
    reg_model = safe_load("reg_model.pkl")
    return survey_model, survey_encoder, reg_model

survey_model, survey_encoder, reg_model = load_models()

# =================================
# 연금 계산 및 시뮬레이션 함수들
# =================================
def calculate_pension_estimate(monthly_income, years):
    """간단한 연금 추정 계산"""
    if reg_model is not None:
        try:
            X = pd.DataFrame([{"평균월소득(만원)": monthly_income, "가입기간(년)": years}])
            amount = round(float(reg_model.predict(X)[0]), 1)
            return amount
        except:
            pass
    
    # 모델이 없거나 실패시 간단한 공식 사용
    estimated_pension = (monthly_income * 0.015 * years)
    return round(estimated_pension, 1)

def retirement_simulation(current_age, end_age, current_assets, monthly_income, monthly_expense,
                          inflation_rate=0.03, investment_return=0.02):
    """노후 시뮬레이션"""
    asset = float(current_assets)
    yearly_log = []
    expense = float(monthly_expense)
    depletion_age = None

    for age in range(int(current_age), int(end_age) + 1):
        annual_income = float(monthly_income) * 12
        annual_expense = float(expense) * 12
        delta = annual_income - annual_expense
        asset += delta
        if asset > 0:
            asset *= (1 + float(investment_return))

        yearly_log.append({
            "나이": age,
            "수입": round(annual_income),
            "지출": round(annual_expense),
            "증감": round(delta),
            "잔액": round(asset)
        })

        if asset <= 0 and depletion_age is None:
            depletion_age = age
            break

        expense *= (1 + float(inflation_rate))

    return yearly_log, depletion_age

def get_financial_advice(current_age, monthly_income, monthly_expense, current_assets, investment_type):
    """투자성향에 따른 금융 조언"""
    surplus = monthly_income - monthly_expense
    
    advice = {
        "안정형": {
            "products": ["정기예금", "정기적금", "국채"],
            "expected_return": "연 2-3%",
            "description": "원금보장을 중시하는 안전한 투자 성향입니다."
        },
        "안정추구형": {
            "products": ["혼합형펀드", "채권형펀드", "CMA"],
            "expected_return": "연 3-4%", 
            "description": "수익과 안정의 균형을 추구하는 투자 성향입니다."
        },
        "위험중립형": {
            "products": ["인덱스펀드", "혼합형펀드", "리츠"],
            "expected_return": "연 4-6%",
            "description": "적정 수준의 위험을 감수할 수 있는 투자 성향입니다."
        },
        "적극투자형": {
            "products": ["주식형펀드", "성장주펀드", "해외펀드"],
            "expected_return": "연 6-8%",
            "description": "높은 수익을 위해 변동성을 감내할 수 있는 투자 성향입니다."
        }
    }
    
    return advice.get(investment_type, advice["위험중립형"])

def predict_financial_type(answers):
    """설문 답변을 바탕으로 금융 유형 예측"""
    if survey_model is not None and survey_encoder is not None:
        try:
            # 설문 답변을 모델 입력 형태로 변환
            gender = 0 if answers["gender"] == "남성" else 1
            dependents = 1 if answers["dependents"] == "예" else 0
            risk_map = {"안정형": 0, "안정추구형": 1, "위험중립형": 2, "적극투자형": 3}
            risk = risk_map.get(answers["risk"], 2)
            
            arr = np.array([[
                float(answers["age"]), gender, float(answers["family_size"]), dependents,
                float(answers["assets"]), float(answers["pension"]), float(answers["living_cost"]),
                float(answers["income"]), risk
            ]])
            
            pred = survey_model.predict(arr)
            financial_type = survey_encoder.inverse_transform(pred)[0].strip()
            return financial_type
        except Exception as e:
            st.warning(f"모델 예측 실패: {e}")
    
    # 모델이 없거나 실패시 투자성향 기반 간단 분류
    return answers.get("risk", "위험중립형")

# =================================
# CSS 스타일링
# =================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px 0;
        margin-bottom: 30px;
        background-color: #f8f9fa;
    }
    
    .kb-logo {
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 15px;
    }
    
    .kb-star {
        color: #FFB800;
        margin-right: 8px;
    }
    
    .kb-text {
        color: #666;
        margin-right: 15px;
    }
    
    .elderly-emoji {
        font-size: 48px;
        margin-left: 10px;
    }
    
    .title {
        font-size: 24px;
        font-weight: bold;
        color: #333;
        margin-top: 15px;
    }
    
    .stApp {
        max-width: 350px;
        margin: 0 auto;
        background-color: #f8f9fa;
        padding: 20px;
    }
    
    /* 메인화면 Streamlit 버튼 스타일링 */
    div[data-testid="stVerticalBlock"] > div:nth-child(1) .stButton > button {
        width: 100% !important;
        height: 80px !important;
        border-radius: 20px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        transition: all 0.2s ease !important;
        white-space: pre-line !important;
        background: #FFE4B5 !important;
        color: #8B4513 !important;
    }
    
    div[data-testid="stVerticalBlock"] > div:nth-child(1) .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    
    /* 두 번째 버튼 (수령중) - 파란색 */
    div[data-testid="stVerticalBlock"] > div:nth-child(3) .stButton > button {
        width: 100% !important;
        height: 80px !important;
        border-radius: 20px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        transition: all 0.2s ease !important;
        white-space: pre-line !important;
        background: #B8D4F0 !important;
        color: #2C5282 !important;
    }
    
    div[data-testid="stVerticalBlock"] > div:nth-child(3) .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    
    /* 세 번째 버튼 (상품정보) - 초록색 */
    div[data-testid="stVerticalBlock"] > div:nth-child(5) div:nth-child(1) .stButton > button {
        width: 100% !important;
        height: 60px !important;
        border-radius: 20px !important;
        font-size: 16px !important;
        font-weight: bold !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        transition: all 0.2s ease !important;
        white-space: pre-line !important;
        background: #C6F6D5 !important;
        color: #22543D !important;
    }
    
    div[data-testid="stVerticalBlock"] > div:nth-child(5) div:nth-child(1) .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    
    /* 네 번째 버튼 (전화상담) - 분홍색 */
    div[data-testid="stVerticalBlock"] > div:nth-child(5) div:nth-child(2) .stButton > button {
        width: 100% !important;
        height: 60px !important;
        border-radius: 20px !important;
        font-size: 16px !important;
        font-weight: bold !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        transition: all 0.2s ease !important;
        white-space: pre-line !important;
        background: #FED7E2 !important;
        color: #97266D !important;
    }
    
    div[data-testid="stVerticalBlock"] > div:nth-child(5) div:nth-child(2) .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
    
    /* 선택 버튼 스타일링 */
    div[data-testid="stVerticalBlock"] .stButton > button {
        background: #E8F4FD !important;
        color: #1E40AF !important;
        border: 2px solid #60A5FA !important;
        border-radius: 15px !important;
        font-size: 18px !important;
        font-weight: bold !important;
        padding: 20px !important;
        margin: 10px 0 !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="stVerticalBlock"] .stButton > button:hover {
        background: #DBEAFE !important;
        border-color: #3B82F6 !important;
        transform: translateY(-2px) !important;
    }
    
    /* 텍스트 입력 스타일링 */
    .stTextInput > div > div > input {
        border-radius: 15px !important;
        border: 2px solid #E5E7EB !important;
        padding: 15px 20px !important;
        font-size: 16px !important;
        text-align: center !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* 결과 화면 스타일링 */
    .result-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin: 10px 0;
    }
    
    .advice-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
    }
    
    /* 모바일 최적화 */
    @media (max-width: 400px) {
        .kb-logo {
            font-size: 32px;
        }
        
        .title {
            font-size: 20px;
        }
    }
</style>
""", unsafe_allow_html=True)

# =================================
# 세션 상태 초기화
# =================================
if 'page' not in st.session_state:
    st.session_state.page = 'main'
if 'question_step' not in st.session_state:
    st.session_state.question_step = 1
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'survey_answers' not in st.session_state:
    st.session_state.survey_answers = {}
if 'financial_type' not in st.session_state:
    st.session_state.financial_type = None

# =================================
# 헤더
# =================================
st.markdown("""
<div class="main-header">
    <div class="kb-logo">
        <span class="kb-star">★</span><span class="kb-text">b KB</span>
        <span class="elderly-emoji">👴👵</span>
    </div>
    <div class="title">시니어 연금 계산기</div>
</div>
""", unsafe_allow_html=True)

# =================================
# 메인 페이지
# =================================
if st.session_state.page == 'main':
    # 현재 연금 미수령 중 버튼
    if st.button("현재 연금\n미수령 중", key="pension_not_receiving", use_container_width=True):
        st.session_state.page = 'not_receiving'
        st.rerun()
    
    st.markdown('<div style="margin: 15px 0;"></div>', unsafe_allow_html=True)
    
    # 현재 연금 수령 중 버튼  
    if st.button("현재 연금\n수령 중", key="pension_receiving", use_container_width=True):
        st.session_state.page = 'receiving'
        st.rerun()
    
    st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)
    
    # 하단 버튼들
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("상품\n정보", key="product_info", use_container_width=True):
            st.session_state.page = 'product_info'
            st.rerun()
    
    with col2:
        if st.button("전화\n상담", key="phone_consultation", use_container_width=True):
            st.session_state.page = 'phone_consultation'
            st.rerun()

# =================================
# 현재 연금 미수령 중 페이지 - 통합된 설문
# =================================
elif st.session_state.page == 'not_receiving':
    st.markdown("### 📊 연금 계산을 위한 정보 입력")
    
    with st.form("pension_survey"):
        st.write("**기본 정보**")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("나이", min_value=20, max_value=100, value=45)
            gender = st.selectbox("성별", ["남성", "여성"])
            family_size = st.number_input("가구원 수", min_value=1, max_value=10, value=2)
        
        with col2:
            dependents = st.selectbox("피부양자가 있나요?", ["아니오", "예"])
            monthly_income = st.number_input("월 평균 소득 (만원)", min_value=0, value=300)
            career_years = st.number_input("국민연금 가입 예정 기간 (년)", min_value=1, max_value=50, value=20)
        
        st.write("**재정 정보**")
        col3, col4 = st.columns(2)
        with col3:
            current_assets = st.number_input("현재 보유 금융자산 (만원)", min_value=0, value=5000)
            current_pension = st.number_input("현재 받는 연금 (만원)", min_value=0, value=0)
        
        with col4:
            monthly_expense = st.number_input("월 평균 지출 (만원)", min_value=0, value=200)
            investment_type = st.selectbox("투자 성향", 
                ["안정형", "안정추구형", "위험중립형", "적극투자형"])
        
        submitted = st.form_submit_button("연금 계산 및 분석 시작", use_container_width=True)
    
    if submitted:
        # 설문 답변 저장
        survey_data = {
            "age": age,
            "gender": gender,
            "family_size": family_size,
            "dependents": dependents,
            "income": monthly_income,
            "assets": current_assets,
            "pension": current_pension,
            "living_cost": monthly_expense,
            "risk": investment_type
        }
        
        st.session_state.survey_answers = survey_data
        
        # 금융 유형 예측
        financial_type = predict_financial_type(survey_data)
        st.session_state.financial_type = financial_type
        
        # 연금 계산
        estimated_pension = calculate_pension_estimate(monthly_income, career_years)
        
        # 결과 페이지로 이동
        st.session_state.page = 'not_receiving_result'
        st.session_state.estimated_pension = estimated_pension
        st.rerun()
    
    if st.button("← 메인으로 돌아가기"):
        st.session_state.page = 'main'
        st.rerun()

# =================================
# 미수령자 결과 페이지
# =================================
elif st.session_state.page == 'not_receiving_result':
    survey_data = st.session_state.survey_answers
    financial_type = st.session_state.financial_type
    estimated_pension = st.session_state.estimated_pension
    
    st.markdown("### 📈 연금 계산 및 분석 결과")
    
    # 연금 계산 결과
    st.markdown(f"""
    <div class="metric-card">
        <h3>예상 월 연금액</h3>
        <h2>{estimated_pension:,.1f} 만원</h2>
        <p>국민연금 가입기간 기준</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 금융 유형 분석
    st.markdown(f"""
    <div class="result-card">
        <h4>🎯 당신의 금융 유형: {financial_type}</h4>
        <p>설문 답변을 바탕으로 분석된 결과입니다.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 투자 조언
    advice = get_financial_advice(
        survey_data["age"], 
        survey_data["income"], 
        survey_data["living_cost"], 
        survey_data["assets"], 
        survey_data["risk"]
    )
    
    st.markdown(f"""
    <div class="advice-card">
        <h4>💡 맞춤 투자 조언</h4>
        <p><strong>투자 성향:</strong> {survey_data["risk"]}</p>
        <p><strong>설명:</strong> {advice["description"]}</p>
        <p><strong>추천 상품:</strong> {", ".join(advice["products"])}</p>
        <p><strong>예상 수익률:</strong> {advice["expected_return"]}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 노후 시뮬레이션
    st.markdown("### 📊 노후 자금 시뮬레이션")
    
    monthly_total_income = survey_data["income"] + estimated_pension
    log_data, depletion_age = retirement_simulation(
        survey_data["age"], 100, survey_data["assets"], 
        monthly_total_income, survey_data["living_cost"]
    )
    
    if depletion_age:
        st.warning(f"⚠️ 예상 자산 고갈 나이: {depletion_age}세")
        st.write("추가적인 노후 준비가 필요합니다.")
    else:
        st.success("✅ 현재 계획으로는 자산 고갈 없이 노후를 보낼 수 있습니다.")
    
    # 연도별 자산 변화 차트
    if log_data:
        df_chart = pd.DataFrame(log_data)
        df_chart = df_chart.set_index('나이')
        st.line_chart(df_chart[['잔액']])
    
    # 상세 분석 표
    with st.expander("상세 재정 분석 보기"):
        surplus = monthly_total_income - survey_data["living_cost"]
        st.metric("월 예상 총수입", f"{monthly_total_income:,.0f}만원", f"소득 {survey_data['income']}만원 + 연금 {estimated_pension}만원")
        st.metric("월 지출", f"{survey_data['living_cost']:,.0f}만원")
        st.metric("월 수지", f"{surplus:,.0f}만원", "흑자" if surplus > 0 else "적자")
        st.metric("현재 자산", f"{survey_data['assets']:,.0f}만원")
    
    # 네비게이션 버튼
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← 메인으로 돌아가기"):
            st.session_state.page = 'main'
            st.rerun()
    with col2:
        if st.button("다른 시나리오 보기"):
            st.session_state.page = 'simulation'
            st.rerun()

# =================================
# 현재 연금 수령 중 페이지
# =================================
elif st.session_state.page == 'receiving':
    st.markdown("### 💰 현재 연금 수령 현황 분석")
    
    with st.form("receiving_survey"):
        st.write("**기본 정보**")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("나이", min_value=50, max_value=100, value=67)
            gender = st.selectbox("성별", ["남성", "여성"])
            family_size = st.number_input("가구원 수", min_value=1, max_value=10, value=2)
        
        with col2:
            dependents = st.selectbox("피부양자가 있나요?", ["아니오", "예"])
            current_pension = st.number_input("현재 수령 중인 연금 (만원)", min_value=0, value=100)
            other_income = st.number_input("기타 월 소득 (만원)", min_value=0, value=50)
        
        st.write("**재정 정보**")
        col3, col4 = st.columns(2)
        with col3:
            current_assets = st.number_input("현재 보유 금융자산 (만원)", min_value=0, value=10000)
            monthly_expense = st.number_input("월 평균 지출 (만원)", min_value=0, value=150)
        
        with col4:
            investment_type = st.selectbox("투자 성향", 
                ["안정형", "안정추구형", "위험중립형", "적극투자형"])
            start_year = st.number_input("연금 수령 시작 연도", min_value=1990, max_value=2024, value=2020)
        
        submitted = st.form_submit_button("현황 분석 시작", use_container_width=True)
    
    if submitted:
        # 분석 결과 계산
        total_monthly_income = current_pension + other_income
        surplus = total_monthly_income - monthly_expense
        years_receiving = 2024 - start_year
        total_received = current_pension * 12 * years_receiving
        
        st.markdown("### 📊 수령 현황 분석")
        
        # 현황 요약
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("월 총수입", f"{total_monthly_income:,.0f}만원")
        with col2:
            st.metric("월 지출", f"{monthly_expense:,.0f}만원")
        with col3:
            if surplus > 0:
                st.metric("월 잉여금", f"{surplus:,.0f}만원", "흑자")
            else:
                st.metric("월 부족금", f"{abs(surplus):,.0f}만원", "적자")
        
        # 수령 이력
        st.markdown(f"""
        <div class="result-card">
            <h4>📈 연금 수령 이력</h4>
            <p><strong>수령 기간:</strong> {years_receiving}년 ({start_year}년부터)</p>
            <p><strong>총 수령액:</strong> {total_received:,.0f}만원</p>
            <p><strong>월 평균:</strong> {current_pension:,.0f}만원</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 투자 조언
        advice = get_financial_advice(age, total_monthly_income, monthly_expense, current_assets, investment_type)
        
        st.markdown(f"""
        <div class="advice-card">
            <h4>💡 추가 투자 조언</h4>
            <p><strong>현재 상황:</strong> {"안정적" if surplus > 0 else "개선 필요"}</p>
            <p><strong>추천 전략:</strong> {advice["description"]}</p>
            <p><strong>적합 상품:</strong> {", ".join(advice["products"])}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 향후 전망
        log_data, depletion_age = retirement_simulation(
            age, 100, current_assets, total_monthly_income, monthly_expense
        )
        
        if depletion_age:
            st.warning(f"⚠️ 현재 지출 수준 유지시 {depletion_age}세에 자산 고갈 예상")
        else:
            st.success("✅ 현재 수준으로는 자산을 유지할 수 있습니다.")
    
    if st.button("← 메인으로 돌아가기"):
        st.session_state.page = 'main'
        st.rerun()

# =================================
# 상품 정보 페이지
# =================================
elif st.session_state.page == 'product_info':
    st.markdown("### 📋 KB 시니어 연금 상품 정보")
    
    # 상품 정보 카드들
    st.markdown("""
    <div class="result-card">
        <h4>🏦 KB 시니어 연금보험</h4>
        <p><strong>특징:</strong> 안정적인 노후 소득 보장</p>
        <p><strong>세제혜택:</strong> 연금저축 세액공제 혜택</p>
        <p><strong>수령방식:</strong> 종신연금, 확정연금 선택 가능</p>
        <p><strong>최소가입금액:</strong> 월 10만원부터</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="result-card">
        <h4>💰 KB 퇴직연금</h4>
        <p><strong>특징:</strong> 퇴직금을 연금으로 전환</p>
        <p><strong>세제혜택:</strong> 퇴직소득세 이연 혜택</p>
        <p><strong>운용방식:</strong> 원리금보장형, 실적배당형</p>
        <p><strong>수수료:</strong> 경쟁력 있는 수수료 구조</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="result-card">
        <h4>📈 KB 연금펀드</h4>
        <p><strong>특징:</strong> 다양한 자산군 분산투자</p>
        <p><strong>수익구조:</strong> 시장 수익률 연동</p>
        <p><strong>리스크:</strong> 투자위험 존재 (원금손실 가능)</p>
        <p><strong>적합대상:</strong> 중장기 투자자</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 가입 조건
    st.markdown("### 📝 가입 조건")
    st.markdown("""
    **공통 가입 조건:**
    - 만 18세 이상 65세 이하
    - 소득 증빙이 가능한 분
    - 건강 상태 양호 (일부 상품)
    
    **필요 서류:**
    - 신분증, 소득증빙서류
    - 건강진단서 (필요시)
    - 기타 심사 관련 서류
    """)
    
    # 상담 안내
    st.markdown("""
    <div class="advice-card">
        <h4>🎯 전문가 상담 추천</h4>
        <p>개인별 상황에 맞는 최적의 상품 선택을 위해 전문가 상담을 받아보세요.</p>
        <p><strong>상담 방법:</strong> 전화, 온라인, 지점 방문</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("전화 상담 신청"):
            st.session_state.page = 'phone_consultation'
            st.rerun()
    with col2:
        if st.button("← 메인으로 돌아가기"):
            st.session_state.page = 'main'
            st.rerun()

# =================================
# 전화 상담 페이지
# =================================
elif st.session_state.page == 'phone_consultation':
    st.markdown("### 📞 전화 상담 신청")
    
    st.markdown("""
    <div class="result-card">
        <h4>📋 KB 시니어 연금 상담센터</h4>
        <p><strong>상담 전화번호:</strong> 1588-9999</p>
        <p><strong>운영시간:</strong></p>
        <ul>
            <li>평일: 오전 9시 ~ 오후 6시</li>
            <li>토요일: 오전 9시 ~ 오후 1시</li>
            <li>일요일 및 공휴일: 휴무</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # 상담 신청 폼
    st.markdown("### ✍️ 상담 신청서")
    
    with st.form("consultation_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("성명 *")
            phone = st.text_input("연락처 *")
            age_group = st.selectbox("연령대", ["40대", "50대", "60대", "70대 이상"])
        
        with col2:
            email = st.text_input("이메일")
            preferred_time = st.selectbox("희망 상담 시간", 
                ["오전 9-12시", "오후 1-3시", "오후 3-6시", "시간 무관"])
            consultation_type = st.selectbox("상담 유형", 
                ["연금보험 상담", "퇴직연금 상담", "펀드 상담", "종합 상담"])
        
        inquiry = st.text_area("문의 내용", placeholder="상담받고 싶은 내용을 자세히 적어주세요.")
        
        # 개인정보 동의
        privacy_agree = st.checkbox("개인정보 수집·이용에 동의합니다. *")
        marketing_agree = st.checkbox("마케팅 목적의 개인정보 이용에 동의합니다.")
        
        submitted = st.form_submit_button("상담 신청하기", use_container_width=True)
    
    if submitted:
        if name and phone and privacy_agree:
            # 상담 신청 성공 처리
            st.success("✅ 상담 신청이 완료되었습니다!")
            
            st.markdown(f"""
            <div class="result-card">
                <h4>📋 신청 내용 확인</h4>
                <p><strong>성명:</strong> {name}</p>
                <p><strong>연락처:</strong> {phone}</p>
                <p><strong>상담 유형:</strong> {consultation_type}</p>
                <p><strong>희망 시간:</strong> {preferred_time}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("💡 영업일 기준 1-2일 내에 담당자가 연락드릴 예정입니다.")
            
        else:
            st.error("❌ 필수 항목을 모두 입력해주세요.")
    
    # 자주 묻는 질문
    with st.expander("❓ 자주 묻는 질문"):
        st.markdown("""
        **Q: 상담 비용이 있나요?**
        A: 모든 상담은 무료입니다.
        
        **Q: 가입을 강요하지는 않나요?**
        A: 고객의 필요에 따른 맞춤 상담을 제공하며, 가입을 강요하지 않습니다.
        
        **Q: 어떤 서류를 준비해야 하나요?**
        A: 신분증과 소득증빙서류를 준비해주시면 더 정확한 상담이 가능합니다.
        
        **Q: 온라인으로도 상담받을 수 있나요?**
        A: 네, 화상상담도 가능합니다. 상담 신청시 요청해주세요.
        """)
    
    if st.button("← 메인으로 돌아가기"):
        st.session_state.page = 'main'
        st.rerun()

# =================================
# 시뮬레이션 페이지 (추가)
# =================================
elif st.session_state.page == 'simulation':
    st.markdown("### 📊 노후 자금 시뮬레이션")
    
    # 기본값 설정 (이전 설문 데이터가 있으면 사용)
    survey_data = st.session_state.get('survey_answers', {})
    
    with st.form("simulation_form"):
        st.markdown("**시뮬레이션 조건 설정**")
        
        col1, col2 = st.columns(2)
        with col1:
            sim_age = st.number_input("현재 나이", 
                min_value=30, max_value=80, 
                value=survey_data.get('age', 45))
            sim_retire_age = st.number_input("은퇴 나이", 
                min_value=sim_age, max_value=80, 
                value=max(65, sim_age))
            sim_assets = st.number_input("현재 자산 (만원)", 
                min_value=0, 
                value=survey_data.get('assets', 5000))
        
        with col2:
            sim_monthly_save = st.number_input("월 저축액 (만원)", 
                min_value=0, value=100)
            sim_monthly_expense = st.number_input("은퇴 후 월 지출 (만원)", 
                min_value=0, 
                value=survey_data.get('living_cost', 200))
            sim_pension = st.number_input("예상 연금 (만원)", 
                min_value=0, 
                value=st.session_state.get('estimated_pension', 80))
        
        col3, col4 = st.columns(2)
        with col3:
            investment_return = st.slider("투자 수익률 (%)", 0.0, 15.0, 5.0, 0.5)
            inflation_rate = st.slider("물가상승률 (%)", 0.0, 10.0, 3.0, 0.5)
        
        with col4:
            life_expectancy = st.selectbox("기대수명", [85, 90, 95, 100], index=2)
            scenario = st.selectbox("시나리오", 
                ["보수적", "중간", "낙관적"], index=1)
        
        run_simulation = st.form_submit_button("시뮬레이션 실행", use_container_width=True)
    
    if run_simulation:
        # 시나리오별 수익률 조정
        scenario_multiplier = {"보수적": 0.7, "중간": 1.0, "낙관적": 1.3}
        adjusted_return = investment_return * scenario_multiplier[scenario] / 100
        
        # 은퇴 전 자산 축적 단계
        accumulated_assets = sim_assets
        for year in range(sim_retire_age - sim_age):
            accumulated_assets += sim_monthly_save * 12
            accumulated_assets *= (1 + adjusted_return)
        
        # 은퇴 후 시뮬레이션
        monthly_income_retirement = sim_pension
        log_data, depletion_age = retirement_simulation(
            sim_retire_age, life_expectancy, accumulated_assets,
            monthly_income_retirement, sim_monthly_expense,
            inflation_rate/100, adjusted_return
        )
        
        # 결과 표시
        st.markdown("### 📈 시뮬레이션 결과")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("은퇴시 예상 자산", f"{accumulated_assets:,.0f}만원")
        with col2:
            st.metric("월 연금 수입", f"{monthly_income_retirement:,.0f}만원")
        with col3:
            if depletion_age:
                st.metric("자산 고갈 나이", f"{depletion_age}세", "위험")
            else:
                st.metric("자산 상태", "고갈 없음", "안전")
        
        # 연도별 자산 변화 차트
        if log_data:
            df_chart = pd.DataFrame(log_data)
            df_chart = df_chart.set_index('나이')
            st.line_chart(df_chart[['잔액']])
        
        # 시나리오 분석
        st.markdown("### 🎯 시나리오 분석")
        
        if depletion_age and depletion_age < life_expectancy:
            shortage_years = life_expectancy - depletion_age
            additional_needed = sim_monthly_expense * 12 * shortage_years
            
            st.warning(f"""
            ⚠️ **추가 준비 필요**
            - 자산 부족 기간: {shortage_years}년
            - 추가 필요 자산: {additional_needed:,.0f}만원
            - 권장사항: 월 저축액 증액 또는 지출 절약 검토
            """)
            
            # 개선 방안 제시
            required_monthly_save = additional_needed / ((sim_retire_age - sim_age) * 12)
            st.info(f"💡 **개선 방안**: 월 {required_monthly_save:,.0f}만원 추가 저축 권장")
            
        else:
            st.success("""
            ✅ **안정적인 노후 계획**
            현재 계획으로 충분한 노후 자금 확보가 가능합니다.
            """)
        
        # 상세 분석표
        with st.expander("상세 분석 보기"):
            if log_data:
                df_detail = pd.DataFrame(log_data)
                st.dataframe(df_detail, use_container_width=True)
    
    # 네비게이션
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← 메인으로 돌아가기"):
            st.session_state.page = 'main'
            st.rerun()
    with col2:
        if st.button("다른 조건으로 재계산"):
            st.rerun()

# =================================
# 푸터
# =================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px; margin-top: 30px;">
    <p>KB 시니어 연금 계산기 | 고객센터: 1588-9999</p>
    <p>본 계산 결과는 예시이며, 실제 상품 가입시 조건이 다를 수 있습니다.</p>
</div>
""", unsafe_allow_html=True)

# KB 시니어 연금 계산기 - 완전한 기존 기능 + KB 디자인
import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# FAISS 설정 (기존과 동일)
USE_FAISS = True
try:
    import faiss  # pip: faiss-cpu
except Exception as e:
    USE_FAISS = False
    from sklearn.neighbors import NearestNeighbors

# =================================
# 기본 설정 (기존과 완전히 동일)
# =================================
st.set_page_config(page_title="KB 시니어 연금 계산기", page_icon="🏦", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
MODELS_DIR = BASE_DIR
DEPOSIT_CSV = "금융상품_3개_통합본.csv"  # 예·적금 CSV
FUND_CSV    = "펀드_병합본.csv"          # 펀드 CSV

# 예측→설문 프리필 값을 사용자가 수정 못 하게 잠글지 여부
LOCK_INFERRED_FIELDS = False  # True면 자동 채운 소득/연금 칸 비활성화

SHOW_PROBA_CHART = False  # 설문 예측 확률 막대차트 표시 여부
SHOW_SUCCESS_TOAST = False  # '예측된 금융 유형' 성공 메시지 표시 여부

# =================================
# 공통 유틸 (인덱스 빌드/검색)
# =================================
def build_index(X: np.ndarray):
    X = X.astype("float32")
    if USE_FAISS:
        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X)
        return index
    nn = NearestNeighbors(metric="euclidean")
    nn.fit(X)
    return nn

def index_search(index, q: np.ndarray, k: int):
    q = q.astype("float32")
    if USE_FAISS:
        return index.search(q, k)
    D, I = index.kneighbors(q, n_neighbors=k, return_distance=True)
    return D, I

# =================================
# 모델/데이터 로딩 (캐시)
# =================================
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

    survey_model   = safe_load("tabnet_model.pkl")
    survey_encoder = safe_load("label_encoder.pkl")
    reg_model      = safe_load("reg_model.pkl")
    type_model     = safe_load("type_model.pkl")
    return survey_model, survey_encoder, reg_model, type_model

@st.cache_data
def load_deposit_csv():
    path = os.path.join(BASE_DIR, DEPOSIT_CSV)
    if not os.path.exists(path):
        raise FileNotFoundError(f"예·적금 파일이 없습니다: {path}")
    for enc in ("utf-8-sig", "cp949"):
        try: return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError: pass
    return pd.read_csv(path)

@st.cache_data
def load_fund_csv():
    path = os.path.join(BASE_DIR, FUND_CSV)
    if not os.path.exists(path):
        raise FileNotFoundError(f"펀드 파일이 없습니다: {path}")
    for enc in ("utf-8-sig", "cp949"):
        try: return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError: pass
    return pd.read_csv(path)

survey_model, survey_encoder, reg_model, type_model = load_models()

# =================================
# 전처리 & 추천 유틸
# =================================
def preprocess_products(df: pd.DataFrame, group_name: str = "") -> pd.DataFrame:
    """CSV → 공통 전처리. group_name='예·적금' 또는 '펀드' 라벨."""
    np.random.seed(42)
    df = df.copy()
    df.columns = df.columns.str.strip()

    # 상품명
    if '상품명' in df.columns:
        names = df['상품명'].fillna('무명상품').astype(str)
    elif '펀드명' in df.columns:
        names = df['펀드명'].fillna('무명상품').astype(str)
    elif '출처파일명' in df.columns:
        names = df['출처파일명'].apply(lambda x: str(x).split('.')[0] if pd.notnull(x) else '무명상품')
    else:
        names = [f"무명상품_{i}" for i in range(len(df))]

    # 최소 투자금액
    if '최고한도' in df.columns:
        min_invest = pd.to_numeric(df['최고한도'], errors='coerce').fillna(0)
        zero_mask = (min_invest == 0)
        if zero_mask.any():
            min_invest.loc[zero_mask] = np.random.randint(100, 1000, zero_mask.sum())
    elif '최소가입금액' in df.columns:
        min_invest = pd.to_numeric(df['최소가입금액'], errors='coerce')
        miss = min_invest.isna()
        if miss.any():
            min_invest.loc[miss] = np.random.randint(100, 1000, miss.sum())
    else:
        min_invest = pd.Series(np.random.randint(100, 1000, len(df)), index=df.index)

    # 수익률 (%) → 소수
    cand_cols = [c for c in df.columns if any(k in c for k in ["기본금리", "이자율", "세전", "%", "수익률", "수익"])]
    rate_col = cand_cols[0] if cand_cols else None
    if rate_col:
        raw = (df[rate_col].astype(str)
                         .str.replace(",", "", regex=False)
                         .str.extract(r"([\d\.]+)")[0])
        est_return = pd.to_numeric(raw, errors="coerce")
        rand_series = pd.Series(np.random.uniform(1.0, 8.0, len(df)), index=df.index)
        est_return = (est_return.fillna(rand_series) / 100.0).astype(float).round(4)
    else:
        low, high = (0.01, 0.08) if group_name != "펀드" else (0.03, 0.15)
        est_return = pd.Series(np.round(np.random.uniform(low, high, len(df)), 4), index=df.index)

    # 리스크
    if '위험등급' in df.columns:
        raw_risk = df['위험등급'].astype(str)
        risk = raw_risk.apply(lambda x: '높음' if ('5' in x or '4' in x) else ('중간' if '3' in x else '낮음'))
    else:
        if group_name == "펀드":
            risk = pd.Series(np.random.choice(['낮음','중간','높음'], len(df), p=[0.2,0.4,0.4]), index=df.index)
        else:
            risk = pd.Series(np.random.choice(['낮음','중간','높음'], len(df), p=[0.6,0.3,0.1]), index=df.index)

    # 권장기간/투자성향(필터용)
    duration = pd.Series(np.random.choice([6, 12, 24, 36], len(df)), index=df.index)
    profile  = pd.Series(np.random.choice(['안정형','위험중립형','공격형'], len(df)), index=df.index)

    out = pd.DataFrame({
        '구분': group_name if group_name else '기타',
        '상품명': names,
        '최소투자금액': min_invest.astype(int),
        '예상수익률': est_return,
        '리스크': risk,
        '권장투자기간': duration,
        '투자성향': profile
    })
    return out[out['상품명'] != '무명상품'].drop_duplicates(subset=['상품명']).reset_index(drop=True)

def rule_based_filter(df: pd.DataFrame, user: dict) -> pd.DataFrame:
    # 방어: user 유효성
    if not isinstance(user, dict):
        st.warning("내부 경고: 사용자 선호 정보가 올바르지 않습니다.")
        return df.head(0)

    # 기본값 + 타입 정리
    risk_choice = (user.get('투자성향') or '위험중립형')
    invest_amt  = user.get('투자금액', 0) or 0
    invest_per  = user.get('투자기간', 0) or 0

    try:
        invest_amt = int(invest_amt)
    except Exception:
        invest_amt = 0
    try:
        invest_per = int(invest_per)
    except Exception:
        invest_per = 0

    # 리스크 허용 매핑
    risk_pref_map = {
        '안정형': ['낮음','중간'],
        '위험중립형': ['중간','낮음','높음'],
        '공격형': ['높음','중간']
    }
    allowed = risk_pref_map.get(risk_choice, ['낮음','중간','높음'])

    f = df[
        (pd.to_numeric(df['최소투자금액'], errors='coerce').fillna(10**9) <= invest_amt) &
        (pd.to_numeric(df['권장투자기간'], errors='coerce').fillna(10**9) <= invest_per) &
        (df['리스크'].isin(allowed))
    ]
    return f.sort_values('예상수익률', ascending=False).head(500).reset_index(drop=True)

def _get_feature_vector(df: pd.DataFrame) -> np.ndarray:
    return np.vstack([
        df['최소투자금액'].astype(float) / 1000.0,
        df['예상수익률'].astype(float) * 100.0,
        df['권장투자기간'].astype(float) / 12.0
    ]).T.astype('float32')

def _get_user_vector(user: dict) -> np.ndarray:
    return np.array([
        user['투자금액'] / 1000.0,
        user['목표월이자'],
        user['투자기간'] / 12.0
    ], dtype='float32').reshape(1, -1)

def _add_explain(df: pd.DataFrame, user: dict) -> pd.DataFrame:
    out = df.copy()
    out['월예상수익금(만원)'] = (out['예상수익률'].astype(float) * user['투자금액'] / 12.0).round(1)
    out['투자기간(개월)'] = out['권장투자기간'].astype(int)
    out['예상수익률'] = out['예상수익률'].astype(float)  # <- 숫자 보존 (0.05)
    out['예상수익률(연)'] = (out['예상수익률'] * 100).round(2).astype(str) + '%'

    # 상세 보기에 필요한 컬럼들을 반환 목록에 포함
    cols = [
        '구분','상품명','월예상수익금(만원)','예상수익률','예상수익률(연)',
        '리스크','투자기간(개월)','최소투자금액','투자성향'
    ]
    cols = [c for c in cols if c in out.columns]  # 누락 대비
    return out[cols]

def recommend_fallback_split(user: dict) -> pd.DataFrame:
    # 기본키 채워 넣기 (혹시 누락되면)
    user = {
        '투자금액': user.get('투자금액', 0) if isinstance(user, dict) else 0,
        '투자기간': user.get('투자기간', 0) if isinstance(user, dict) else 0,
        '투자성향': user.get('투자성향', '위험중립형') if isinstance(user, dict) else '위험중립형',
        '목표월이자': user.get('목표월이자', 0) if isinstance(user, dict) else 0,
    }

    try:
        dep_raw = load_deposit_csv()
        fun_raw = load_fund_csv()
    except FileNotFoundError:
        return pd.DataFrame({'메시지': ['상품 데이터 파일이 없어서 추천을 할 수 없습니다. 😢']})
    
    dep = preprocess_products(dep_raw, "예·적금")
    fun = preprocess_products(fun_raw, "펀드")

    dep_f = rule_based_filter(dep, user)
    fun_f = rule_based_filter(fun, user)

    if dep_f.empty and fun_f.empty:
        return pd.DataFrame({'메시지': ['조건에 맞는 상품이 없어요 😢']})

    # 예·적금 2
    if not dep_f.empty:
        Xd = _get_feature_vector(dep_f)
        idxd = build_index(Xd)
        _, idd = index_search(idxd, _get_user_vector(user), min(2, len(dep_f)))
        rec_dep = dep_f.iloc[idd[0]].copy().head(2)
    else:
        rec_dep = pd.DataFrame(columns=dep_f.columns)

    # 펀드 1
    if not fun_f.empty:
        Xf = _get_feature_vector(fun_f)
        idxf = build_index(Xf)
        _, idf = index_search(idxf, _get_user_vector(user), min(1, len(fun_f)))
        rec_fun = fun_f.iloc[idf[0]].copy().head(1)
    else:
        rec_fun = pd.DataFrame(columns=fun_f.columns)

    out = pd.concat([rec_dep, rec_fun], ignore_index=True)
    out = out.drop_duplicates(subset=['상품명']).reset_index(drop=True)
    return _add_explain(out, user)

# =================================
# [NEW] 노후 시뮬레이션 & 추천 근거 유틸
# =================================
def retirement_simulation(current_age, end_age, current_assets, monthly_income, monthly_expense,
                          inflation_rate=0.03, investment_return=0.02):
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

def simulate_with_financial_product(current_age, end_age, current_assets, monthly_income, monthly_expense,
                                    invest_return=0.05):
    return retirement_simulation(current_age, end_age, current_assets, monthly_income, monthly_expense,
                                 inflation_rate=0.03, investment_return=invest_return)

def get_invest_return_from_risk(risk_level: str) -> float:
    """예측/선택된 위험성향을 연 수익률 가정으로 변환"""
    if risk_level in ["안정형", "안정추구형"]:
        return 0.03
    if risk_level in ["위험중립형"]:
        return 0.05
    if risk_level in ["적극투자형", "공격투자형", "공격형"]:
        return 0.07
    return 0.05

def recommend_reason_from_simulation(depletion_age, current_age, current_assets,
                                     monthly_income, monthly_expense, risk_level: str):
    """시뮬레이션 결과 기반 간단 추천 근거 메시지"""
    surplus = monthly_income - monthly_expense
    if depletion_age:
        if surplus <= 0:
            return f"{depletion_age}세에 자산 고갈 예상 · 현금흐름 보강이 시급합니다."
        if current_assets < 10000:
            return f"{depletion_age}세 자산 고갈 위험 · 절세형/분산형 상품으로 수익률 제고가 필요합니다."
        return f"{depletion_age}세 자산 고갈 위험 · 위험도('{risk_level}')에 맞는 수익원 다변화가 필요합니다."
    # 고갈 없음
    if current_assets >= 20000 and surplus > 0:
        return f"자산/현금흐름이 양호합니다 · '{risk_level}'에 맞춘 분산투자로 실질가치(물가 3%) 방어를 권장합니다."
    return "지출 구조를 점검하고 비과세/저비용 상품으로 실질 수익률을 높이세요."

# =================================
# 결과 화면 (스케치 스타일)
# =================================
# 설명 사전들 (파일 상단에 선언)
RISK_STYLE_DESCRIPTIONS = {
    "안정형": "자산/연금 비율이 안정적이고 원금 보전을 선호해요. 예·적금과 초저위험 상품 위주가 좋아요.",
    "안정추구형": "수익과 안정의 균형을 중시해요. 예·적금 + 초저위험 펀드를 소폭 섞는 구성이 적합해요.",
    "위험중립형": "위험/수익을 균형 있게 받아들여요. 채권형·혼합형과 적금을 혼합하면 좋아요.",
    "적극투자형": "수익을 위해 변동성을 일정 수준 허용해요. 혼합형/주식형 비중을 조금 더 높여요.",
    "공격투자형": "높은 수익을 위해 변동성 감내도가 높아요. 주식형·테마형 등 성장지향 상품을 고려해요.",
    "위험취약형": "손실 회피 성향이 매우 큽니다. 원금 보전이 최우선이며 예·적금, MMF, 초저위험 채권형 위주가 적합합니다."
}
TABNET_TYPE_DESCRIPTIONS = {
    "자산운용형": "💼 투자 여력이 충분한 유형으로, 운용 전략 중심의 포트폴리오가 적합합니다.",
    "위험취약형": "⚠️ 재무 위험이 높은 유형입니다. 지출 관리와 복지 연계가 필요합니다.",
    "균형형": "⚖️ 자산과 연금이 안정적인 편으로, 보수적인 전략이 적합합니다.",
    "고소비형": "💳 소비가 많은 유형으로 절세 전략 및 예산 재조정이 필요합니다.",
    "자산의존형": "🏦 연금보다는 자산에 의존도가 높으며, 자산 관리 전략이 중요합니다.",
    "연금의존형": "🔥 자산보다 연금에 의존하는 경향이 강한 유형입니다.",
}
DEFAULT_DISPLAY_TYPE = "균형형"

def render_final_screen(display_type: str, rec_df: pd.DataFrame):
    # TabNet 유형 우선 → 없으면 투자성향 → 폴백
    if display_type in TABNET_TYPE_DESCRIPTIONS:
        title = display_type
        desc  = TABNET_TYPE_DESCRIPTIONS[display_type]
    elif display_type in RISK_STYLE_DESCRIPTIONS:
        title = display_type
        desc  = RISK_STYLE_DESCRIPTIONS[display_type]
    else:
        title = DEFAULT_DISPLAY_TYPE
        desc  = TABNET_TYPE_DESCRIPTIONS.get(DEFAULT_DISPLAY_TYPE, "")

    st.markdown("""
    <style>
      .hero { font-size: 38px; font-weight: 800; margin: 4px 0 8px 0; }
      .desc { font-size: 16px; opacity: 0.9; margin-bottom: 18px; }
      .cards { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
      .card {
        border: 2px solid #eaeaea; border-radius: 18px; padding: 16px 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.06); background: #fff;
      }
      .badge {
        display:inline-flex; align-items:center; justify-content:center;
        width:28px; height:28px; border-radius:50%; color:#fff; font-weight:700;
        margin-right:8px;
      }
      .b1{ background:#ff5a5a; } .b2{ background:#7c4dff; } .b3{ background:#10b981; }
      .pname{ font-size:17px; font-weight:700; margin:6px 0 10px 0; }
      .meta{ font-size:14px; line-height:1.5; }
      .k { font-weight:700; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="hero">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="desc">• {desc}</div>', unsafe_allow_html=True)

    colors = ["b1", "b2", "b3"]
    items = rec_df.head(3).to_dict(orient="records")

    cards = []
    for i, r in enumerate(items, start=1):
        cname = colors[i-1 if i-1 < len(colors) else -1]
        name = str(r.get("상품명", "-"))
        mret = r.get("월예상수익금(만원)", "-")
        risk = r.get("리스크", "-")
        card_html = (
            f'<div class="card">'
            f'<div><span class="badge {cname}">{i}</span><span class="pname">{name}</span></div>'
            f'<div class="meta"><span class="k">월 예상수익</span> {mret}만원</div>'
            f'<div class="meta"><span class="k">리스크</span> {risk}</div>'
            f'</div>'
        )
        cards.append(card_html)

    cards_html = '<div class="cards">' + ''.join(cards) + '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

# =================================
# CSS 스타일링 - KB 디자인
# =================================
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
    
    /* 메인 버튼 스타일링 */
    .stButton > button {
        width: 100% !important;
        border: none !important;
        border-radius: 20px !important;
        font-size: 20px !important;
        font-weight: bold !important;
        text-align: center !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1) !important;
        line-height: 1.4 !important;
        padding: 25px 20px !important;
        margin: 15px 0 !important;
        white-space: pre-line !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 15px rgba(0,0,0,0.15) !important;
    }
    
    /* 첫 번째 버튼 (미수령) - 노란색 */
    div[data-testid="stVerticalBlock"] > div:nth-child(2) .stButton > button {
        background: #FFE4B5 !important;
        color: #8B4513 !important;
        height: 80px !important;
    }
    
    /* 두 번째 버튼 (수령중) - 파란색 */
    div[data-testid="stVerticalBlock"] > div:nth-child(3) .stButton > button {
        background: #B8D4F0 !important;
        color: #2C5282 !important;
        height: 80px !important;
        margin-bottom: 25px !important;
    }
    
    /* 세 번째 버튼 (상품정보) - 초록색 */
    div[data-testid="stVerticalBlock"] > div:nth-child(4) div:nth-child(1) .stButton > button {
        background: #C6F6D5 !important;
        color: #22543D !important;
        height: 60px !important;
        font-size: 16px !important;
        padding: 20px 15px !important;
    }
    
    /* 네 번째 버튼 (전화상담) - 분홍색 */
    div[data-testid="stVerticalBlock"] > div:nth-child(4) div:nth-child(2) .stButton > button {
        background: #FED7E2 !important;
        color: #97266D !important;
        height: 60px !important;
        font-size: 16px !important;
        padding: 20px 15px !important;
    }
    
    /* 모바일 최적화 */
    @media (max-width: 400px) {
        .main-container {
            padding: 15px;
        }
        
        .stButton > button {
            font-size: 18px !important;
            padding: 20px 15px !important;
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

# =================================
# UI 흐름
# =================================
st.title("💬 KB 시니어 연금 계산기")

ss = st.session_state
ss.setdefault("flow", "main")          # ← 기존 'choose' 대신 'main'으로 시작
ss.setdefault("survey_type", None)     # 미수령/수령 구분 추가
ss.setdefault("pred_amount", None)
ss.setdefault("answers", {})
ss.setdefault("prefill_survey", {})
ss.setdefault("pred_label", None)
ss.setdefault("tabnet_label", None)    # ← TabNet 금융유형(표시용)

"""
app.py
Streamlit 기반 주식 매수·매도 타이밍 분류 웹 대시보드
"""

import streamlit as st
import pandas as pd
from datetime import date, timedelta

from data_loader import load_ohlcv, load_investor_net_buy, merge_data
from features   import build_features, create_labels, get_feature_columns
from model      import train_models, predict_full
from visualizer import plot_stock_signals, plot_model_comparison


# ──────────────────────────────────────────
# 페이지 설정
# ──────────────────────────────────────────
st.set_page_config(
    page_title="주식 매수·매도 타이밍 분류기",
    page_icon="📈",
    layout="wide",
)

st.title("📈 과거 주식 데이터 기반 매수·매도 타이밍 구분 시스템")
st.caption("파이썬 과학 기초 프로그래밍 프로젝트 | 소프트웨어학부")

# ──────────────────────────────────────────
# 사이드바 — 입력 파라미터
# ──────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")

    ticker = st.text_input(
        "종목 코드",
        value="005930",
        help="KRX 종목 코드 6자리  예) 삼성전자: 005930 / 카카오: 035720",
    )

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "시작일",
            value=date.today() - timedelta(days=365 * 2),
        )
    with col2:
        end_date = st.date_input("종료일", value=date.today())

    st.divider()
    st.subheader("레이블 설정")
    n_days        = st.slider("미래 N일 수익률 기준",  3, 20, 5)
    buy_thr       = st.slider("매수 임계값 (%)",       1, 10, 2) / 100
    sell_thr      = st.slider("매도 임계값 (%)",      -10, -1, -2) / 100

    st.divider()
    run_btn = st.button("🚀 분석 시작", use_container_width=True, type="primary")

# ──────────────────────────────────────────
# 분석 실행
# ──────────────────────────────────────────
if run_btn:
    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")

    with st.spinner("📥 데이터 수집 중..."):
        try:
            ohlcv    = load_ohlcv(ticker, start_str, end_str)
            investor = load_investor_net_buy(ticker, start_str, end_str)
            df_raw   = merge_data(ohlcv, investor)
        except Exception as e:
            st.error(f"데이터 수집 실패: {e}")
            st.stop()

    with st.spinner("🔧 지표 생성 및 레이블 생성 중..."):
        df_feat  = build_features(df_raw)
        df_label = create_labels(df_feat, n_days, buy_thr, sell_thr)

    # ── 데이터 요약 ──────────────────────────────────────
    st.subheader("📊 데이터 요약")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("수집 기간",  f"{start_str} ~ {end_str}")
    m2.metric("거래일 수",  f"{len(df_label):,}일")
    m3.metric("Buy 비율",   f"{(df_label['Label']==1).mean()*100:.1f}%")
    m4.metric("Sell 비율",  f"{(df_label['Label']==-1).mean()*100:.1f}%")

    with st.expander("원본 데이터 확인"):
        st.dataframe(df_label.tail(30), use_container_width=True)

    # ── 모델 학습 ────────────────────────────────────────
    with st.spinner("🤖 모델 학습 중..."):
        feature_cols = get_feature_columns()
        results      = train_models(df_label, feature_cols)

    # ── 성능 지표 ────────────────────────────────────────
    st.subheader("🏆 모델 성능 비교")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Logistic Regression")
        st.metric("Accuracy", f"{results['lr']['accuracy']*100:.1f}%")
        st.metric("F1-score (macro)", f"{results['lr']['f1']:.3f}")
        with st.expander("상세 리포트"):
            st.text(results["lr"]["report"])
    with c2:
        st.markdown("### Decision Tree")
        st.metric("Accuracy", f"{results['dt']['accuracy']*100:.1f}%")
        st.metric("F1-score (macro)", f"{results['dt']['f1']:.3f}")
        with st.expander("상세 리포트"):
            st.text(results["dt"]["report"])

    # ── 전체 데이터 예측 ─────────────────────────────────
    lr_pred_full = predict_full(
        results["lr"]["model"], results["lr"]["scaler"],
        df_label, feature_cols,
    )
    dt_pred_full = predict_full(
        results["dt"]["model"], results["dt"]["scaler"],
        df_label, feature_cols,
    )

    # ── 차트 탭 ──────────────────────────────────────────
    st.subheader("📉 매수·매도 타이밍 차트")
    tab1, tab2, tab3 = st.tabs([
        "Logistic Regression", "Decision Tree", "모델 비교"
    ])

    with tab1:
        fig_lr = plot_stock_signals(df_label, lr_pred_full, ticker, "Logistic Regression")
        st.pyplot(fig_lr)

    with tab2:
        fig_dt = plot_stock_signals(df_label, dt_pred_full, ticker, "Decision Tree")
        st.pyplot(fig_dt)

    with tab3:
        fig_cmp = plot_model_comparison(df_label, lr_pred_full, dt_pred_full, ticker)
        st.pyplot(fig_cmp)

else:
    # 실행 전 안내 화면
    st.info("👈 사이드바에서 종목 코드와 기간을 설정한 뒤 **분석 시작** 버튼을 눌러주세요.")
    st.markdown("""
    #### 사용 방법
    1. **종목 코드** 입력 (예: 삼성전자 `005930`, SK하이닉스 `000660`)
    2. **분석 기간** 설정
    3. **레이블 기준** 조정 (기본값 권장)
    4. **분석 시작** 클릭

    #### 분석 방법론
    | 단계 | 내용 |
    |------|------|
    | 데이터 수집 | FinanceDataReader (OHLCV) + pykrx (투자자 순매수) |
    | 지표 생성 | MA5/20/60, 거래량 MA, 골든·데드크로스 |
    | 레이블 | 미래 N일 수익률 기준 Buy/Sell/Hold |
    | 모델 | Logistic Regression + Decision Tree |
    | 시각화 | 주가 차트 + ▲▼ 타이밍 마커 |
    """)
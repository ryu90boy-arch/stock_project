"""
visualizer.py
주가 차트 위에 매수(▲) / 매도(▼) 타이밍을 표시하는 시각화 함수
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform


def _set_korean_font():
    """OS 별 한글 폰트 설정"""
    system = platform.system()
    if system == "Darwin":       # macOS
        plt.rcParams["font.family"] = "AppleGothic"
    elif system == "Windows":
        plt.rcParams["font.family"] = "Malgun Gothic"
    else:                        # Linux (Streamlit Cloud 등)
        # 나눔고딕이 설치된 경우
        try:
            fe = fm.FontEntry(
                fname="/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                name="NanumGothic",
            )
            fm.fontManager.ttflist.insert(0, fe)
            plt.rcParams["font.family"] = "NanumGothic"
        except Exception:
            pass  # 폰트 없으면 기본 폰트 사용

    plt.rcParams["axes.unicode_minus"] = False


def plot_stock_signals(
    df: pd.DataFrame,
    predictions: pd.Series,
    ticker: str,
    model_name: str = "Model",
) -> plt.Figure:
    """
    주가 차트 + 이동평균선 + 매수(▲)·매도(▼) 신호를 그립니다.

    Parameters
    ----------
    df          : OHLCV + 지표가 포함된 DataFrame
    predictions : 예측 레이블 Series  (1=Buy, -1=Sell, 0=Hold)
    ticker      : 종목 코드 (제목 표시용)
    model_name  : 모델 이름 (제목 표시용)
    """
    _set_korean_font()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor("#0E1117")
    for ax in [ax1, ax2]:
        ax.set_facecolor("#0E1117")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333333")

    # ── 종가 및 이동평균선 ──────────────────────────────
    ax1.plot(df.index, df["Close"], color="#FFFFFF", linewidth=1.2, label="종가", zorder=2)
    colors = {"MA5": "#F0E68C", "MA20": "#87CEEB", "MA60": "#FFA07A"}
    for col, color in colors.items():
        if col in df.columns:
            ax1.plot(df.index, df[col], color=color, linewidth=0.8,
                     linestyle="--", label=col, alpha=0.8)

    # ── 매수(▲) / 매도(▼) 마커 ──────────────────────────
    buy_idx  = predictions[predictions == 1].index
    sell_idx = predictions[predictions == -1].index

    if len(buy_idx):
        ax1.scatter(
            buy_idx, df.loc[buy_idx, "Close"] * 0.985,
            marker="^", color="#00FF7F", s=80, zorder=5, label="Buy ▲",
        )
    if len(sell_idx):
        ax1.scatter(
            sell_idx, df.loc[sell_idx, "Close"] * 1.015,
            marker="v", color="#FF4500", s=80, zorder=5, label="Sell ▼",
        )

    ax1.set_title(
        f"{ticker}  |  {model_name}  매수·매도 타이밍",
        color="white", fontsize=14, pad=10,
    )
    ax1.set_ylabel("가격 (원)", color="white")
    ax1.legend(loc="upper left", facecolor="#1E1E1E", labelcolor="white",
               fontsize=8, framealpha=0.7)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    # ── 거래량 바 차트 ───────────────────────────────────
    vol_colors = [
        "#00FF7F" if c >= o else "#FF4500"
        for c, o in zip(df["Close"], df["Open"])
    ]
    ax2.bar(df.index, df["Volume"], color=vol_colors, alpha=0.7, width=1)
    if "Vol_MA20" in df.columns:
        ax2.plot(df.index, df["Vol_MA20"], color="#87CEEB",
                 linewidth=0.8, label="Vol MA20")
    ax2.set_ylabel("거래량", color="white", fontsize=9)
    ax2.legend(loc="upper left", facecolor="#1E1E1E", labelcolor="white",
               fontsize=8, framealpha=0.7)

    plt.tight_layout()
    return fig


def plot_model_comparison(
    df: pd.DataFrame,
    lr_pred: pd.Series,
    dt_pred: pd.Series,
    ticker: str,
) -> plt.Figure:
    """로지스틱 회귀 vs 결정 트리 비교 차트 (2행)"""
    _set_korean_font()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.patch.set_facecolor("#0E1117")

    for ax, pred, name in zip(
        axes,
        [lr_pred, dt_pred],
        ["Logistic Regression", "Decision Tree"],
    ):
        ax.set_facecolor("#0E1117")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#333333")

        ax.plot(df.index, df["Close"], color="#FFFFFF", linewidth=1.0, label="종가")

        buy_idx  = pred[pred == 1].index
        sell_idx = pred[pred == -1].index

        if len(buy_idx):
            ax.scatter(buy_idx,  df.loc[buy_idx,  "Close"] * 0.985,
                       marker="^", color="#00FF7F", s=60, label="Buy ▲", zorder=5)
        if len(sell_idx):
            ax.scatter(sell_idx, df.loc[sell_idx, "Close"] * 1.015,
                       marker="v", color="#FF4500", s=60, label="Sell ▼", zorder=5)

        ax.set_title(f"{ticker}  |  {name}", color="white", fontsize=12)
        ax.set_ylabel("가격 (원)", color="white")
        ax.legend(loc="upper left", facecolor="#1E1E1E",
                  labelcolor="white", fontsize=8, framealpha=0.7)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x):,}")
        )

    plt.tight_layout()
    return fig
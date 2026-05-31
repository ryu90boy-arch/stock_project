"""
features.py
이동평균·거래량 지표 생성 및 매수/매도/관망 레이블 생성
"""

import pandas as pd
import numpy as np


# ──────────────────────────────────────────
# 1. 기술적 지표 생성
# ──────────────────────────────────────────

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """5일·20일·60일 이동평균선 추가"""
    df = df.copy()
    for window in [5, 20, 60]:
        df[f"MA{window}"] = df["Close"].rolling(window).mean()
    return df


def add_volume_ma(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """거래량 이동평균 추가"""
    df = df.copy()
    df[f"Vol_MA{window}"] = df["Volume"].rolling(window).mean()
    return df


def add_golden_dead_cross(df: pd.DataFrame) -> pd.DataFrame:
    """
    골든크로스 / 데드크로스 신호 컬럼 추가
    MA5 가 MA20 을 상향 돌파 → 1 (골든), 하향 돌파 → -1 (데드), 그 외 → 0
    """
    df = df.copy()
    df["Cross"] = 0
    cross = df["MA5"] - df["MA20"]
    df.loc[(cross > 0) & (cross.shift(1) <= 0), "Cross"] = 1   # 골든크로스
    df.loc[(cross < 0) & (cross.shift(1) >= 0), "Cross"] = -1  # 데드크로스
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """모든 파생 지표를 한 번에 추가하는 편의 함수"""
    df = add_moving_averages(df)
    df = add_volume_ma(df)
    df = add_golden_dead_cross(df)
    df.dropna(inplace=True)
    return df


# ──────────────────────────────────────────
# 2. 레이블 생성
# ──────────────────────────────────────────

def create_labels(
    df: pd.DataFrame,
    n_days: int = 5,
    buy_threshold: float = 0.02,
    sell_threshold: float = -0.02,
) -> pd.DataFrame:
    """
    미래 N일 후 수익률을 기준으로 레이블을 생성합니다.

    Parameters
    ----------
    n_days         : 몇 일 후 수익률 기준 (기본 5일)
    buy_threshold  : 이 수익률 이상이면 Buy=1  (기본 +2%)
    sell_threshold : 이 수익률 이하면 Sell=-1 (기본 -2%)
    나머지 → Hold=0

    Returns
    -------
    'Label' 컬럼이 추가된 DataFrame (미래 N일치 끝부분은 NaN → 제거)
    """
    df = df.copy()
    future_return = df["Close"].shift(-n_days) / df["Close"] - 1

    df["Label"] = 0  # Hold
    df.loc[future_return >= buy_threshold,  "Label"] = 1   # Buy
    df.loc[future_return <= sell_threshold, "Label"] = -1  # Sell

    df.dropna(subset=["Label"], inplace=True)
    return df


def get_feature_columns() -> list[str]:
    """모델 학습에 사용할 피처 컬럼 목록"""
    return [
        "Close", "Volume",
        "MA5", "MA20", "MA60",
        "Vol_MA20", "Cross",
        "기관", "외국인", "개인",
    ]
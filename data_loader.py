"""
data_loader.py
FinanceDataReader 를 이용한 주가 데이터 수집
(pykrx 투자자 데이터는 KRX 로그인 필요로 인해 생략)
"""

import pandas as pd
import FinanceDataReader as fdr


def load_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = fdr.DataReader(ticker, start, end)
    df.index = pd.to_datetime(df.index)
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    return df


def load_investor_net_buy(ticker: str, start: str, end: str) -> pd.DataFrame:
    """투자자 데이터 대신 빈 DataFrame 반환 (KRX 로그인 불필요)"""
    return pd.DataFrame()


def merge_data(ohlcv: pd.DataFrame, investor: pd.DataFrame) -> pd.DataFrame:
    if investor.empty:
        df = ohlcv.copy()
        df["기관"]   = 0.0
        df["외국인"] = 0.0
        df["개인"]   = 0.0
    else:
        df = ohlcv.join(investor, how="left")
        df.fillna(0, inplace=True)
    return df
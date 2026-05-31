"""
model.py
로지스틱 회귀 + 결정 트리 모델 학습·평가·예측
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report


def prepare_xy(df: pd.DataFrame, feature_cols: list[str]):
    """
    피처 컬럼과 레이블 컬럼을 분리합니다.
    피처 컬럼 중 DataFrame 에 없는 컬럼은 0으로 채웁니다.
    """
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]

    X = df[available].copy()
    for col in missing:
        X[col] = 0.0

    X = X[feature_cols]  # 순서 고정
    y = df["Label"].astype(int)
    return X, y


def train_models(df: pd.DataFrame, feature_cols: list[str], test_size: float = 0.2):
    """
    로지스틱 회귀 + 결정 트리를 학습하고 평가 결과를 반환합니다.

    Returns
    -------
    results : dict
        {
          "lr":  {"model": ..., "scaler": ..., "report": ..., "accuracy": ..., "f1": ...},
          "dt":  {"model": ..., "scaler": None, "report": ..., "accuracy": ..., "f1": ...},
          "X_test": ..., "y_test": ..., "test_index": ...
        }
    """
    X, y = prepare_xy(df, feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False  # 시계열 데이터이므로 순서 유지
    )
    test_index = X_test.index

    # ── 로지스틱 회귀 (스케일링 필요) ──────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    lr_pred = lr.predict(X_test_sc)

    # ── 결정 트리 ────────────────────────────────────────
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)

    results = {
        "lr": {
            "model":    lr,
            "scaler":   scaler,
            "pred":     lr_pred,
            "accuracy": accuracy_score(y_test, lr_pred),
            "f1":       f1_score(y_test, lr_pred, average="macro", zero_division=0),
            "report":   classification_report(
                            y_test, lr_pred,
                            target_names=["Sell(-1)", "Hold(0)", "Buy(1)"],
                            zero_division=0,
                        ),
        },
        "dt": {
            "model":    dt,
            "scaler":   None,
            "pred":     dt_pred,
            "accuracy": accuracy_score(y_test, dt_pred),
            "f1":       f1_score(y_test, dt_pred, average="macro", zero_division=0),
            "report":   classification_report(
                            y_test, dt_pred,
                            target_names=["Sell(-1)", "Hold(0)", "Buy(1)"],
                            zero_division=0,
                        ),
        },
        "X_test":     X_test,
        "y_test":     y_test,
        "test_index": test_index,
    }
    return results


def predict_full(
    model,
    scaler,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.Series:
    """
    전체 데이터에 대해 예측값을 반환합니다. (차트용)
    """
    X, _ = prepare_xy(df, feature_cols)
    if scaler is not None:
        X_sc = scaler.transform(X)
    else:
        X_sc = X
    preds = model.predict(X_sc)
    return pd.Series(preds, index=df.index, name="Prediction")
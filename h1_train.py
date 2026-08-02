from datetime import datetime, timezone
import os
import argparse
import joblib
from dados import obter_dados_mt5
import pandas as pd
from indicadores import calcular_rsi
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def build_h1(df_m5: pd.DataFrame) -> pd.DataFrame:
    dados = df_m5.copy().sort_values("time").drop_duplicates("time").reset_index(drop=True)
    dados["time"] = pd.to_datetime(dados["time"])
    df_h1 = dados.set_index("time").resample("1h", closed='left', label='left').agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna().reset_index()

    df_h1["ema_21_h1"] = df_h1["close"].ewm(span=21, adjust=False).mean()
    df_h1["ema_200_h1"] = df_h1["close"].ewm(span=200, adjust=False).mean()
    df_h1["rsi_14_h1"] = calcular_rsi(df_h1["close"])
    df_h1["feat_macro_cruzamento"] = (df_h1["ema_21_h1"] - df_h1["ema_200_h1"]) / df_h1["ema_200_h1"]
    df_h1["feat_macro_rsi"] = (df_h1["rsi_14_h1"] - 50.0) / 50.0
    return df_h1

def train_and_save(ativo="EURUSD", inicio=datetime(2024,1,1, tzinfo=timezone.utc), fim=None, out_path="models/h1_models.joblib", local_csv: str | None = None):
    print("Collecting M5 data from MT5 (or local fallback if provided)...")
    df_m5 = obter_dados_mt5(ativo=ativo, inicio=inicio, fim=fim, local_fallback=local_csv)
    df_h1 = build_h1(df_m5)

    # build targets
    df_h1["alvo_direcao"] = (df_h1["close"].shift(-1) > df_h1["open"].shift(-1)).astype(int)
    df_h1["alvo_tamanho"] = (df_h1["close"].shift(-1) - df_h1["open"].shift(-1)).abs()
    media_diaria = df_h1["close"].rolling(24, min_periods=1).mean()
    df_h1["alvo_dist_dia"] = (df_h1["close"].shift(-1) - media_diaria.shift(-1)) / media_diaria.shift(-1)

    df_h1 = df_h1.dropna(subset=["alvo_direcao", "alvo_tamanho", "alvo_dist_dia"]).reset_index(drop=True)
    if len(df_h1) < 10:
        raise RuntimeError("Não há dados H1 suficientes para treinar os modelos H1.")

    cols = ["feat_macro_cruzamento", "feat_macro_rsi", "open", "high", "low", "close"]
    X = df_h1[cols]

    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    clf.fit(X, df_h1["alvo_direcao"])

    reg_tam = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    reg_tam.fit(X, df_h1["alvo_tamanho"])

    reg_dist = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    reg_dist.fit(X, df_h1["alvo_dist_dia"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump((clf, reg_tam, reg_dist), out_path)
    print(f"Saved H1 models to {out_path}")

def _parse_args():
    p = argparse.ArgumentParser(description="Train and save H1 models (with optional local CSV fallback)")
    p.add_argument("--ativo", default="EURUSD")
    p.add_argument("--inicio", default=None, help="ISO date start (YYYY-MM-DD)")
    p.add_argument("--fim", default=None, help="ISO date end (YYYY-MM-DD)")
    p.add_argument("--local-csv", dest="local_csv", default=None, help="Path to local CSV fallback with M5 data")
    p.add_argument("--out", dest="out_path", default="models/h1_models.joblib")
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    inicio = datetime.fromisoformat(args.inicio) if args.inicio else datetime(2024, 1, 1, tzinfo=timezone.utc)
    fim = datetime.fromisoformat(args.fim) if args.fim else None
    train_and_save(ativo=args.ativo, inicio=inicio, fim=fim, out_path=args.out_path, local_csv=args.local_csv)

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def prever_proximo_h1(df_h1: pd.DataFrame) -> pd.DataFrame:
    """Treina uma IA rápida para prever o próximo candle H1 e gera as features preditivas."""
    df = df_h1.copy()
    
    # =========================================================================
    # 1. CRIAR OS ALVOS (TARGETS) DO FUTURO
    # Usamos shift(-1) para trazer a resposta do próximo candle para a linha atual.
    # Esses dados serão usados APENAS para treinar a IA, nunca para o robô operar!
    # =========================================================================
    
    # Target 1: Direção (1 para Alta, 0 para Baixa)
    df["alvo_direcao"] = np.where(df["close"].shift(-1) > df["open"].shift(-1), 1, 0)
    
    # Target 2: Distância/Tamanho do corpo do próximo candle
    df["alvo_tamanho"] = (df["close"].shift(-1) - df["open"].shift(-1)).abs()
    
    # Target 3: Distância do próximo fechamento em relação à média das últimas 24h
    media_diaria = df["close"].rolling(24, min_periods=1).mean()
    df["alvo_dist_dia"] = (df["close"].shift(-1) - media_diaria.shift(-1)) / media_diaria.shift(-1)

    # Removemos a última linha que ficou com alvos NaN por causa do shift(-1)
    df = df.dropna(subset=["alvo_direcao", "alvo_tamanho", "alvo_dist_dia"])

    # =========================================================================
    # 2. DEFINIR AS FEATURES DE ENTRADA (O que a IA vai usar para adivinhar)
    # =========================================================================
    colunas_features = ["feat_macro_cruzamento", "feat_macro_rsi", "close", "open", "high", "low"]
    
    # =========================================================================
    # 3. TREINAR A IA E GERAR PREVISÕES (Evitando Look-ahead Bias)
    # Para ser rápido no processamento e seguro no backtest, prevemos a linha atual 
    # treinando o modelo APENAS com os dados que vieram antes dela.
    # =========================================================================
    
    # Inicializamos as colunas de previsão com zero
    df["feat_ia_prev_direcao"] = 0.0
    df["feat_ia_prev_tamanho"] = 0.0
    df["feat_ia_prev_dist_dia"] = 0.0

    df_treino = df.dropna(subset=["alvo_direcao", "alvo_tamanho", "alvo_dist_dia"])
    models_path = os.path.join("models", "h1_models.joblib")

    if len(df_treino) < 10:
        # fallback simple heuristic when not enough H1 history
        df["feat_ia_prev_direcao"] = (df["close"] > df["open"]).astype(float)
        df["feat_ia_prev_tamanho"] = (df["close"] - df["open"]).abs()
        media_diaria = df["close"].rolling(24, min_periods=1).mean()
        df["feat_ia_prev_dist_dia"] = (df["close"] - media_diaria) / media_diaria.replace(0, 1)
    else:
        X_train = df_treino[colunas_features]

        # If pre-trained models exist on disk, load and use them (faster for live)
        if os.path.exists(models_path):
            try:
                clf_direcao, reg_tamanho, reg_dist_dia = joblib.load(models_path)
                df.loc[df.index, "feat_ia_prev_direcao"] = clf_direcao.predict(df[colunas_features])
                df.loc[df.index, "feat_ia_prev_tamanho"] = reg_tamanho.predict(df[colunas_features])
                df.loc[df.index, "feat_ia_prev_dist_dia"] = reg_dist_dia.predict(df[colunas_features])
            except Exception:
                # if load fails, fallback to training once below
                pass

        # If we didn't fill predictions from disk, train once on available H1 and predict
        if df["feat_ia_prev_direcao"].isnull().any() or (df["feat_ia_prev_direcao"] == 0).all():
            y_dir = df_treino["alvo_direcao"]
            clf_direcao = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=42)
            clf_direcao.fit(X_train, y_dir)
            df.loc[df.index, "feat_ia_prev_direcao"] = clf_direcao.predict(df[colunas_features])

            y_tam = df_treino["alvo_tamanho"]
            reg_tamanho = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42)
            reg_tamanho.fit(X_train, y_tam)
            df.loc[df.index, "feat_ia_prev_tamanho"] = reg_tamanho.predict(df[colunas_features])

            y_dist = df_treino["alvo_dist_dia"]
            reg_dist_dia = RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42)
            reg_dist_dia.fit(X_train, y_dist)
            df.loc[df.index, "feat_ia_prev_dist_dia"] = reg_dist_dia.predict(df[colunas_features])

            # attempt to save the trained models for future use
            try:
                os.makedirs(os.path.dirname(models_path), exist_ok=True)
                joblib.dump((clf_direcao, reg_tamanho, reg_dist_dia), models_path)
            except Exception:
                pass

    # Removemos as colunas de "gabarito/alvo" para o PPO não trapacear lendo o futuro exato
    df = df.drop(columns=["alvo_direcao", "alvo_tamanho", "alvo_dist_dia"])
    
    return df

def calcular_rsi(close: pd.Series, periodo: int = 14) -> pd.Series:
    """RSI de Wilder calculado somente com barras já encerradas."""
    variacao = close.diff()
    ganhos = variacao.clip(lower=0)
    perdas = -variacao.clip(upper=0)
    
    ganho_medio = ganhos.ewm(alpha=1 / periodo, adjust=False, min_periods=periodo).mean()
    perda_media = perdas.ewm(alpha=1 / periodo, adjust=False, min_periods=periodo).mean()
    
    rs = ganho_medio / perda_media.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    rsi = rsi.mask((perda_media == 0) & (ganho_medio > 0), 100.0)
    rsi = rsi.mask((ganho_medio == 0) & (perda_media > 0), 0.0)
    return rsi.fillna(50.0)

def preparar_dados_mercado(df: pd.DataFrame, janela_sup_res: int = 50) -> pd.DataFrame:
    """Calcula M5 e injeta tendências macro (H1 e D1) usando Shift para evitar Lookahead."""
    obrigatorias = {"time", "open", "high", "low", "close", "tick_volume"}
    faltantes = obrigatorias.difference(df.columns)
    if faltantes:
        raise ValueError(f"Dados sem colunas obrigatórias: {sorted(faltantes)}")
        
    dados = df.copy().sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if not pd.api.types.is_datetime64_any_dtype(dados['time']):
        dados['time'] = pd.to_datetime(dados['time'])

    # =========================================================================
    # 🕒 CÁLCULO MULTITIMEFRAME (H1 a partir do M5)
    # =========================================================================
    df_h1 = dados.set_index("time").resample("1h", closed='left', label='left').agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna().reset_index()
    
    # Indicadores básicos do H1
    df_h1["ema_21_h1"] = df_h1["close"].ewm(span=21, adjust=False).mean()
    df_h1["ema_200_h1"] = df_h1["close"].ewm(span=200, adjust=False).mean()
    df_h1["rsi_14_h1"] = calcular_rsi(df_h1["close"])
    
    df_h1["feat_macro_cruzamento"] = (df_h1["ema_21_h1"] - df_h1["ema_200_h1"]) / df_h1["ema_200_h1"]
    df_h1["feat_macro_rsi"] = (df_h1["rsi_14_h1"] - 50.0) / 50.0

    # =========================================================================
    # 🧠 AQUI ENTRA A CHAMADA DA IA PREDITIVA!
    # Passamos o df_h1 para a IA estudar e devolver com as colunas de previsão
    # =========================================================================
    df_h1 = prever_proximo_h1(df_h1)
    
    # Agora separamos apenas as features que o robô vai ler (incluindo as novas da IA)
    df_h1_features = df_h1[[
        "time", 
        "feat_macro_cruzamento", 
        "feat_macro_rsi",
        "feat_ia_prev_direcao",   # <-- Nova coluna da IA
        "feat_ia_prev_tamanho",   # <-- Nova coluna da IA
        "feat_ia_prev_dist_dia"   # <-- Nova coluna da IA
    ]].copy()
    
    # Deslocamos o tempo H1 em 1 hora para frente (como você já fazia)
    df_h1_features["hora_chave"] = df_h1_features["time"] + pd.Timedelta(hours=1)
    df_h1_features = df_h1_features.drop(columns=["time"])
    
    """ # =========================================================================
    # 🕒 1. CÁLCULO MACRO H1
    # =========================================================================
    df_h1 = dados.set_index("time").resample("1h", closed='left', label='left').agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna().reset_index()
    
    df_h1["ema_21_h1"] = df_h1["close"].ewm(span=21, adjust=False).mean()
    df_h1["ema_200_h1"] = df_h1["close"].ewm(span=200, adjust=False).mean()
    df_h1["rsi_14_h1"] = calcular_rsi(df_h1["close"])
    
    df_h1["feat_macro_cruzamento"] = (df_h1["ema_21_h1"] - df_h1["ema_200_h1"]) / df_h1["ema_200_h1"]
    df_h1["feat_macro_rsi"] = (df_h1["rsi_14_h1"] - 50.0) / 50.0
    
    df_h1_features = df_h1[["time", "feat_macro_cruzamento", "feat_macro_rsi"]].copy()
    # SHIFT(1): Desloca o H1 para a próxima hora. O M5 de 10h vai ler os dados que fecharam 09h.
    df_h1_features.iloc[:, 1:] = df_h1_features.iloc[:, 1:].shift(1)
    df_h1_features = df_h1_features.dropna().rename(columns={"time": "hora_chave"}) """

    # =========================================================================
    # 📅 2. CÁLCULO MACRO DIÁRIO (D1)
    # =========================================================================
    df_d1 = dados.set_index("time").resample("1D", closed='left', label='left').agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna().reset_index()
    
    df_d1["ema_21_d1"] = df_d1["close"].ewm(span=21, adjust=False).mean()
    df_d1["ema_200_d1"] = df_d1["close"].ewm(span=200, adjust=False).mean()
    df_d1["rsi_14_d1"] = calcular_rsi(df_d1["close"])
    
    df_d1["feat_diario_cruzamento"] = (df_d1["ema_21_d1"] - df_d1["ema_200_d1"]) / df_d1["ema_200_d1"]
    df_d1["feat_diario_rsi"] = (df_d1["rsi_14_d1"] - 50.0) / 50.0
    
    df_d1_features = df_d1[["time", "feat_diario_cruzamento", "feat_diario_rsi"]].copy()
    # SHIFT(1): Desloca o D1 para o próximo dia útil. O M5 de hoje só vê como o mercado fechou ontem.
    df_d1_features.iloc[:, 1:] = df_d1_features.iloc[:, 1:].shift(1)
    df_d1_features = df_d1_features.dropna().rename(columns={"time": "dia_chave"})

    # =========================================================================
    # ⚡ 3. CÁLCULOS DO CALOR DO MOMENTO (M5)
    # =========================================================================
    dados["atr_14"] = (dados["high"] - dados["low"]).rolling(14, min_periods=14).mean()
    dados["media_atr"] = dados["atr_14"].rolling(20, min_periods=20).mean()
    dados["ema_21"] = dados["close"].ewm(span=21, adjust=False, min_periods=21).mean()
    dados["ema_200"] = dados["close"].ewm(span=200, adjust=False, min_periods=200).mean()
    dados["rsi_14"] = calcular_rsi(dados["close"])
    dados["suporte"] = dados["low"].rolling(janela_sup_res, min_periods=janela_sup_res).min()
    dados["resistencia"] = dados["high"].rolling(janela_sup_res, min_periods=janela_sup_res).max()
    dados["media_volume_20"] = dados["tick_volume"].rolling(20, min_periods=20).mean()
    dados["retorno_5"] = dados["close"].pct_change(periods=5)
    
    dados = dados.dropna().reset_index(drop=True)
    
    # Engenharia de Features M5
    dados["feat_dist_ema21"] = (dados["close"] - dados["ema_21"]) / dados["ema_21"]
    dados["feat_dist_ema200"] = (dados["close"] - dados["ema_200"]) / dados["ema_200"]
    dados["feat_dist_suporte"] = (dados["close"] - dados["suporte"]) / dados["suporte"]
    dados["feat_dist_resistencia"] = (dados["resistencia"] - dados["close"]) / dados["close"]
    dados["feat_rsi_normalizado"] = (dados["rsi_14"] - 50.0) / 50.0
    dados["feat_volume_relativo"] = dados["tick_volume"] / dados["media_volume_20"].replace(0, 1)
    dados["feat_volatilidade_relativa"] = dados["atr_14"] / dados["media_atr"].replace(0, 1)
    dados["feat_cruzamento_medias"] = (dados["ema_21"] - dados["ema_200"]) / dados["ema_200"]
    dados["feat_momento_5"] = dados["retorno_5"] * 100.0
    
    # =========================================================================
    # 🧩 4. SINCRONISMO DOS TEMPOS (MERGE H1 e D1 NO M5)
    # =========================================================================
    dados["hora_chave"] = dados["time"].dt.floor("1h")
    dados["dia_chave"] = dados["time"].dt.floor("1D")
    
    # Anexando H1
    dados = pd.merge(dados, df_h1_features, on="hora_chave", how="left")
    # Anexando D1
    dados = pd.merge(dados, df_d1_features, on="dia_chave", how="left")
    
    # Forward Fill nas novas colunas para garantir cobertura em caso de feriados/buracos
    colunas_macro = ["feat_macro_cruzamento", "feat_macro_rsi", "feat_diario_cruzamento", "feat_diario_rsi"]
    dados[colunas_macro] = dados[colunas_macro].ffill().fillna(0.0)
    
    # Limpeza final
    dados = dados.drop(columns=["hora_chave", "dia_chave"])
    
    return dados.dropna().reset_index(drop=True)
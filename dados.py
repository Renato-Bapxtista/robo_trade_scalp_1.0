from datetime import datetime, timezone
import os
import MetaTrader5 as mt5
import pandas as pd

def _utc(data):
    if data.tzinfo is None:
        return data.replace(tzinfo=timezone.utc)
    return data.astimezone(timezone.utc)

def obter_dados_mt5(ativo="EURUSD", timeframe=mt5.TIMEFRAME_M5, inicio=datetime(2024, 1, 1, tzinfo=timezone.utc), fim=None, local_fallback: str | None = None) -> pd.DataFrame:
    # If a local CSV fallback path is provided and exists, load it instead of querying MT5.
    if local_fallback is not None and os.path.exists(local_fallback):
        print(f"Loading M5 data from local CSV fallback: {local_fallback}")
        df = pd.read_csv(local_fallback, parse_dates=["time"])  # expects a 'time' column
        # ensure times are timezone-aware UTC
        if hasattr(df["time"].dt, "tz") and df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize(timezone.utc)
        elif hasattr(df["time"].dt, "tz"):
            df["time"] = df["time"].dt.tz_convert(timezone.utc)

        # Compute derived columns if missing so downstream code can rely on them
        if "price_range" not in df.columns and {"high", "low"}.issubset(df.columns):
            df["price_range"] = df["high"] - df["low"]
        if "price_volume" not in df.columns:
            df["price_volume"] = df.get("price_range", 0) * df.get("tick_volume", 0)
        if "real_volume" not in df.columns:
            df["real_volume"] = df.get("real_volume", 0)
        return df

    if not mt5.initialize():
        raise RuntimeError(f"Não foi possível inicializar MT5: {mt5.last_error()}")
    
    try:
        if not mt5.symbol_select(ativo, True):
            raise RuntimeError(f"Ativo indisponível no MT5: {ativo}")
        
        # --- FORÇA O MT5 A SINCRONIZAR O HISTÓRICO COM O SERVIDOR ---
        # Abre o ativo no Market Watch para disparar o download em segundo plano
        mt5.market_book_add(ativo) 
        
        inicio_utc = _utc(inicio)
        fim_utc = _utc(fim or datetime.now(timezone.utc))
        
        if inicio_utc >= fim_utc:
            raise ValueError("A data inicial deve ser anterior à data final.")
            
        print(f"Sincronizando histórico de {ativo} desde {inicio_utc.year}... Aguarde.")
        dados = mt5.copy_rates_range(ativo, timeframe, inicio_utc, fim_utc)
        
        # Se falhar na primeira tentativa (dados ainda baixando), espera 3 segundos e tenta de novo
        if dados is None or len(dados) == 0:
            import time
            print("Dados antigos ainda estão sendo baixados pelo MT5... Tentando novamente em 3 segundos.")
            time.sleep(3)
            dados = mt5.copy_rates_range(ativo, timeframe, inicio_utc, fim_utc)
            
        if dados is None or len(dados) == 0:
            raise RuntimeError(f"MT5 não retornou candles. Verifique as configurações de 'Max Bars' no terminal.")
            
        df = pd.DataFrame(dados)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        
        df["price_range"] = df["high"] - df["low"] 
        df["price_volume"] = df["price_range"] * df["tick_volume"] 
        df["real_volume"] = df["real_volume"] 
        
        return df
    finally:
        mt5.market_book_release(ativo)
        mt5.shutdown()


def separar_dados_temporal(dados: pd.DataFrame, proporcao_treino: float = 0.70, proporcao_validacao: float = 0.15, minimo_por_conjunto: int = 250) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < proporcao_treino < 1 or not 0 < proporcao_validacao < 1:
        raise ValueError("Proporções inválidas.")
    if proporcao_treino + proporcao_validacao >= 1:
        raise ValueError("Treino e validação devem deixar uma parcela para teste.")
    if "time" not in dados.columns:
        raise ValueError("Dados sem a coluna obrigatória 'time'.")
        
    dados_ordenados = dados.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    
    fim_treino = int(len(dados_ordenados) * proporcao_treino)
    fim_validacao = fim_treino + int(len(dados_ordenados) * proporcao_validacao)
    
    treino = dados_ordenados.iloc[:fim_treino].copy()
    validacao = dados_ordenados.iloc[fim_treino:fim_validacao].copy()
    teste = dados_ordenados.iloc[fim_validacao:].copy()
    
    if min(len(treino), len(validacao), len(teste)) < minimo_por_conjunto:
        raise ValueError("Volume de dados insuficiente. Puxe um período maior no MT5.")
        
    return treino, validacao, teste

# --- BLOCO DE TESTE ---
# Este bloco roda apenas quando você executa este arquivo direto
if __name__ == "__main__":
    print("Testando a nossa Peça 1...")
    # Buscando dados de teste (usando EURUSD como padrão do seu código)
    df_completo = obter_dados_mt5(ativo="EURUSD", inicio=datetime(2024, 1, 1, tzinfo=timezone.utc))
    print(f"Total de linhas baixadas: {len(df_completo)}")
    
    df_treino, df_val, df_teste = separar_dados_temporal(df_completo)
    print(f"Linhas para Treino: {len(df_treino)}")
    print(f"Linhas para Validação: {len(df_val)}")
    print(f"Linhas para Teste: {len(df_teste)}")

import time
from datetime import datetime, timezone
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Importamos as regras de indicadores que criamos juntos
from indicadores import preparar_dados_mercado

def enviar_ordem_mercado(ativo, tipo_ordem, volume=1.0, sl_pips=0.0003, tp_pips=0.0005):
    """Envia uma ordem real de Compra ou Venda com SL e TP automáticos para o MT5."""
    tick = mt5.symbol_info_tick(ativo)
    if tick is None:
        print(f"❌ Erro ao obter cotação atual de {ativo}")
        return False
        
    preco_mercado = tick.ask if tipo_ordem == "COMPRA" else tick.bid
    tipo_operacao = mt5.ORDER_TYPE_BUY if tipo_ordem == "COMPRA" else mt5.ORDER_TYPE_SELL
    
    # Calcula os alvos físicos no preço de execução
    if tipo_ordem == "COMPRA":
        sl = preco_mercado - sl_pips
        tp = preco_mercado + tp_pips
    else:
        sl = preco_mercado + sl_pips
        tp = preco_mercado - tp_pips
        
    requisicao = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": ativo,
        "volume": volume,
        "type": tipo_operacao,
        "price": preco_mercado,
        "sl": sl,
        "tp": tp,
        "deviation": 10,
        "magic": 2026,  # Identificador único das ordens do robô
        "comment": "Robo PPO Live",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    resultado = mt5.order_send(requisicao)
    if resultado.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Falha ao enviar ordem: {resultado.comment} (Código: {resultado.retcode})")
        return False
        
    print(f"🎯 Ordem de {tipo_ordem} executada com sucesso! Preço: {preco_mercado} | SL: {sl:.5f} | TP: {tp:.5f}")
    return True

def verificar_posicoes_abertas(ativo):
    """Verifica se o robô já possui um contrato rodando neste ativo."""
    posicoes = mt5.positions_get(symbol=ativo)
    if posicoes is None:
        return False
    # Filtra posições abertas pelo número mágico do robô
    posicoes_robo = [p for p in posicoes if p.magic == 2026]
    return len(posicoes_robo) > 0

def rodar_operador_live(ativo="EURUSD"):
    print("================ INITIALIZING LIVE OPERATOR ================")
    # 1. Conecta ao terminal do MT5
    if not mt5.initialize():
        print("❌ Falha ao inicializar o MetaTrader 5")
        return
        
    # 2. Carrega o cérebro treinado da IA
    print("🧠 Carregando modelo 'robo_financeiro_ppo.zip'...")
    model = PPO.load("robo_financeiro_ppo")
    
    colunas_features = None
    print("🚀 Robô em execução... Monitorando o mercado real a cada 10s.")
    
    try:
        while True:
            # Verifica se já estamos posicionados para cumprir a regra de 1 contrato por vez
            if verificar_posicoes_abertas(ativo):
                time.sleep(10)
                continue
                
            # 3. Puxa histórico recente corrigido em UTC
            agora_utc = datetime.now(timezone.utc)
            dados_brutos = mt5.copy_rates_from(ativo, mt5.TIMEFRAME_M5, agora_utc, 300)
            
            if dados_brutos is None or len(dados_brutos) == 0:
                print("⚠️ Falha ao ler candles do MT5. Tentando novamente...")
                time.sleep(5)
                continue
                
            # Transforma em DataFrame estruturado
            df = pd.DataFrame(dados_brutos)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            
            # 4. Processa e calcula os indicadores M5 + H1 idênticos aos do treino
            df_features = preparar_dados_mercado(df)
            
            if colunas_features is None:
                colunas_features = [col for col in df_features.columns if col.startswith("feat_")]
                
            # Pega o último candle fechado (a linha final da tabela)
            ultima_linha = df_features.iloc[-1]
            vetor_estado = np.array(ultima_linha[colunas_features].values, dtype=np.float32)
            
            # 5. A IA analisa o vetor e toma a decisão determinística
            action, _ = model.predict(vetor_estado, deterministic=True)
            
            horario_atual = datetime.now().strftime("%H:%M:%S")
            
            # 6. Executa a ação decidida
            if action == 2:    # COMPRA
                print(f"[{horario_atual}] 🟢 IA detectou sinal de COMPRA!")
                enviar_ordem_mercado(ativo, "COMPRA")
            elif action == 0:  # VENDA
                print(f"[{horario_atual}] 🔴 IA detectou sinal de VENDA!")
                enviar_ordem_mercado(ativo, "VENDA")
            else:              # HOLD (Ação 1)
                # Avisa na tela que o robô analisou e preferiu ficar de fora no candle atual
                print(f"[{horario_atual}] ⏸️ Filtro de Análise: Mantendo posição em HOLD.")
                
            # Aguarda 10 segundos para reavaliar a barra (ou esperar nova oscilação)
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n⏹️ Operador Live desligado pelo usuário.")
    finally:
        mt5.shutdown()

if __name__ == "__main__":
    rodar_operador_live()

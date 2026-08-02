import gymnasium as gymnasium
from gymnasium import spaces
import numpy as np
import pandas as pd

class AmbienteTrading(gymnasium.Env):
    """Simulador Day Trade/Scalper: Com alvos curtos, time-out e custos de transação reais."""
    
    def __init__(self, df: pd.DataFrame, sl_pips: float = 0.0003, tp_pips: float = 0.0005):
        super(AmbienteTrading, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.colunas_features = [col for col in df.columns if col.startswith("feat_")]
        self.num_features = len(self.colunas_features)
        
        self.sl_pips = sl_pips
        self.tp_pips = tp_pips
        
        # [NOVO] Custo de transação real por trade (Spread + Taxas = 0.5 pip no EURUSD)
        self.custo_transacao = 0.00005
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(self.num_features,), dtype=np.float32
        )
        
        self.passo_atual = 0

    def _pegar_observacao(self):
        return np.array(self.df.iloc[self.passo_atual][self.colunas_features].values, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.passo_atual = 0
        return self._pegar_observacao(), {}

    def step(self, action):
        # --- CASO AÇÃO SEJA 1: HOLD (ASSISTIR) ---
        if action == 1:
            self.passo_atual += 1
            finalizado = self.passo_atual >= len(self.df) - 1
            nova_observacao = self._pegar_observacao() if not finalizado else np.zeros((self.num_features,), dtype=np.float32)
            recompensa_bruta = 0.0
            recompensa_bruta -= 0.0003  # Penalidade por não agir (em percentual do preço)
            return nova_observacao, recompensa_bruta, finalizado, False, {}
            
        # --- CASO AÇÃO SEJA 0 (VENDA) OU 2 (COMPRA) ---
        preco_entrada = self.df.iloc[self.passo_atual]["close"]
        operacao_encerrada = False
        duracao_trade = 0 
        max_duracao = 6 
        
        if action == 2:  # COMPRA
            alvo_sl = preco_entrada - self.sl_pips
            alvo_tp = preco_entrada + self.tp_pips
            tipo_ordem = "COMPRA"
        else:            # VENDA
            alvo_sl = preco_entrada + self.sl_pips
            alvo_tp = preco_entrada - self.tp_pips
            tipo_ordem = "VENDA"
            
        while not operacao_encerrada:
            self.passo_atual += 1
            duracao_trade += 1
            
            if self.passo_atual >= len(self.df) - 1:
                preco_final = self.df.iloc[-1]["close"]
                recompensa_bruta = (preco_final - preco_entrada) / preco_entrada if tipo_ordem == "COMPRA" else (preco_entrada - preco_final) / preco_entrada
                break
                
            preco_high = self.df.iloc[self.passo_atual]["high"]
            preco_low = self.df.iloc[self.passo_atual]["low"]
            preco_close = self.df.iloc[self.passo_atual]["close"]
            
            if tipo_ordem == "COMPRA":
                if preco_low <= alvo_sl:
                    recompensa_bruta = -self.sl_pips / preco_entrada
                    operacao_encerrada = True
                elif preco_high >= alvo_tp:
                    recompensa_bruta = self.tp_pips / preco_entrada
                    operacao_encerrada = True
                elif duracao_trade >= max_duracao: 
                    recompensa_bruta = (preco_close - preco_entrada) / preco_entrada
                    operacao_encerrada = True
                    
            elif tipo_ordem == "VENDA":
                if preco_high >= alvo_sl:
                    recompensa_bruta = -self.sl_pips / preco_entrada
                    operacao_encerrada = True
                elif preco_low <= alvo_tp:
                    recompensa_bruta = self.tp_pips / preco_entrada
                    operacao_encerrada = True
                elif duracao_trade >= max_duracao: 
                    recompensa_bruta = (preco_entrada - preco_close) / preco_entrada
                    operacao_encerrada = True

        # [NOVO] Deduz o custo de transação do resultado final do trade (em percentual do preço)
        custo_percentual = self.custo_transacao / preco_entrada
        recompensa_final = recompensa_bruta - custo_percentual

        finalizado = self.passo_atual >= len(self.df) - 1
        nova_observacao = self._pegar_observacao() if not finalizado else np.zeros((self.num_features,), dtype=np.float32)
        
        return nova_observacao, recompensa_final, finalizado, False, {}

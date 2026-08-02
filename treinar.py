from datetime import datetime, timezone
from stable_baselines3 import PPO
from dados import obter_dados_mt5, separar_dados_temporal
from indicadores import preparar_dados_mercado
from ambiente import AmbienteTrading

def executar_treinamento():
    print("--- PASSO 1: Coletando Dados do MetaTrader 5 ---")
    # Puxamos a base completa do EURUSD
    df_bruto = obter_dados_mt5(ativo="EURUSD", inicio=datetime(2024, 1, 1, tzinfo=timezone.utc))
    
    print("\n--- PASSO 2: Calculando e Normalizando Indicadores ---")
    df_com_features = preparar_dados_mercado(df_bruto)
    
    print("\n--- PASSO 3: Separando Dados em Treino e Teste ---")
    # Usamos a sua função técnica de divisão temporal
    df_treino, df_val, df_teste = separar_dados_temporal(df_com_features)
    print(f"Linhas dedicadas ao aprendizado (Treino): {len(df_treino)}")
    
    print("\n--- PASSO 4: Inicializando o Ambiente do Simulador ---")
    # Criamos o ambiente do jogo passando apenas os dados de treino
    env_treino = AmbienteTrading(df_treino)
    
    print("\n--- PASSO 5: Configurando o Cérebro da IA (PPO) ---")
    # Criamos o modelo de Inteligência Artificial
    # 'MlpPolicy' significa que a IA usará uma rede neural padrão para ler as features
    model = PPO(
        policy="MlpPolicy", 
        env=env_treino, 
        verbose=1,
        learning_rate=0.0003,
        ent_coef=0.02, # <--  Força a IA a explorar vendas e compras (padrão é 0.0)
        batch_size=256 # <--  Ajuda a IA a aprender com lotes maiores de dados
    )
    
    print("\n--- PASSO 6: Iniciando Aprendizado por Reforço ---")
    # Mandamos o robô rodar 50.000 passos (candles) dentro do mercado para praticar
    model.learn(total_timesteps=50000, progress_bar=True)
    print()
    # Salva o cérebro treinado em um arquivo
    model.save("robo_financeiro_ppo")
    print("\nTreinamento concluído! Modelo salvo como 'robo_financeiro_ppo.zip'")

if __name__ == "__main__":
    executar_treinamento()

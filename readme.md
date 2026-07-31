# 🤖 RL FX Trader - MetaTrader 5 PPO Trading Bot

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-green.svg)
![MetaTrader5](https://img.shields.io/badge/MetaTrader5-Integration-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Um ecossistema completo de **Algorithmic Trading** utilizando **Reinforcement Learning (RL)** com o algoritmo **PPO (Proximal Policy Optimization)**.

O projeto foi desenvolvido para operar no mercado **Forex**, inicialmente focado no par **EURUSD**, utilizando estratégias de **Scalping** e **Day Trade** no timeframe **M5**, mantendo consciência das tendências de médio e longo prazo através dos gráficos **H1** e **D1**.

---

## 📖 Visão Geral

O objetivo do projeto é construir um agente de Inteligência Artificial capaz de aprender sozinho quando:

- Comprar (Long)
- Vender (Short)
- Não operar (Hold)

O aprendizado ocorre através de milhões de simulações utilizando **Aprendizado por Reforço**, onde o agente recebe recompensas e penalizações conforme seus resultados.

Todo o ambiente foi desenvolvido para reduzir vieses comuns em backtests e aproximar o treinamento das condições reais do mercado.

---

## ✨ Principais Funcionalidades

## 📈 Integração com MetaTrader 5

- Download automático de histórico
- Operação em tempo real
- Envio de ordens diretamente pelo MT5
- Utilização da biblioteca oficial MetaTrader5

---

## 🧠 Inteligência Artificial

- Reinforcement Learning
- PPO (Proximal Policy Optimization)
- Stable-Baselines3
- Política MLP (Rede Neural)

---

## ⏳ Consciência Multi-Timeframe

O agente opera no gráfico M5 mas recebe informações dos timeframes:

- M5
- H1
- D1

As tendências superiores são sincronizadas utilizando **shift(1)** para evitar **Lookahead Bias**.

---

## 🚫 Prevenção de Lookahead Bias

Toda feature utilizada pelo modelo é baseada exclusivamente em candles já fechados.

Isso garante que o treinamento não utilize informações futuras.

---

## 🎯 Ambiente Customizado (Gymnasium)

O ambiente de treinamento foi criado do zero utilizando `gymnasium.Env`.

### Ações possíveis

| Ação | Descrição |

| 0 | Hold |
| 1 | Buy |
| 2 | Sell |

---

### Regras Operacionais

- Stop Loss: **3 pips**
- Take Profit: **5 pips**
- Time-Out: **6 candles (30 minutos)**
- Custos de Spread: **0.5 pip**

---

## 📊 Indicadores utilizados

Exemplo de features utilizadas:

- RSI
- Média Móvel Curta
- Média Móvel Longa
- Distância do preço para as médias
- Tendência H1
- Tendência D1
- Retornos relativos
- Features normalizadas

---

## 🗂 Estrutura do Projeto

```text

rl-fx-trader/
│
├── dados.py
├── indicadores.py
├── ambiente.py
├── treinar.py
├── validar.py
├── live.py
│
├── modelos/
│     └── robo_financeiro_ppo.zip
│
├── logs/
│
├── requirements.txt
│
└── README.md
```

---

## 📄 Descrição dos Arquivos

## dados.py

Responsável por:

- conectar ao MetaTrader 5
- baixar candles históricos
- sincronizar UTC
- separar treino, validação e teste

---

## indicadores.py

Calcula todas as features do modelo:

- RSI
- Médias móveis
- Tendências H1
- Tendências D1
- Features relativas

---

## ambiente.py

Implementa o ambiente Gymnasium.

Controla:

- entradas
- saídas
- stop
- take
- timeout
- recompensa

---

## treinar.py

Pipeline completo de treinamento.

Responsável por:

- criar ambiente
- iniciar PPO
- treinar
- salvar o modelo

Saída:

```text
robo_financeiro_ppo.zip
```

---

## validar.py

Executa um backtest utilizando dados nunca vistos pelo modelo.

Métricas:

- Win Rate
- Número de Trades
- Quantidade de Holds
- Profit/Loss
- Retorno acumulado

---

## live.py

Modo de operação ao vivo.

Fluxo:

1. Atualiza candles
2. Calcula indicadores
3. Consulta IA
4. Decide ação
5. Envia ordem para o MT5

---

## 🚀 Instalação

## Pré-requisitos

- MetaTrader 5 instalado
- Conta Demo ou Real
- Python 3.9+

---

## Clone o projeto

```bash
git clone https://github.com/SEU_USUARIO/rl-fx-trader.git

cd rl-fx-trader
```

---

## Instale as dependências

```bash
pip install -r requirements.txt
```

ou

```bash
pip install MetaTrader5 pandas numpy gymnasium stable-baselines3
```

---

## ▶ Fluxo de Utilização

## 1 — Treinar a IA

```bash
python treinar.py
```

Resultado:

```text
robo_financeiro_ppo.zip
```

---

## 2 — Validar

```bash
python validar.py
```

Saída esperada:

```text
Win Rate

Total Trades

Total Holds

Profit

Drawdown

PNL
```

---

## 3 — Operar ao Vivo

Com o MetaTrader aberto:

```bash
python live.py
```

O robô:

- atualiza o mercado
- calcula indicadores
- consulta a IA
- envia ordens automaticamente

---

## 🧠 Reward Design

A função de recompensa foi construída para incentivar operações de alta qualidade.

## Take Profit

```python
+5 pips
-
custos
=
recompensa positiva
```

---

## Stop Loss

```python
-3 pips
-
custos
=
penalização
```

---

## Time-Out

Caso a operação permaneça aberta por 6 candles:

- encerra automaticamente
- recompensa baseada no lucro/prejuízo atual

---

## Hold

Não entrar no mercado possui recompensa neutra.

```python
Reward = 0
```

Isso incentiva a IA a evitar operações ruins.

---

## 📊 Pipeline do Projeto

```text
                Histórico MT5
                      │
                      ▼
              Download de Dados
                      │
                      ▼
            Cálculo de Indicadores
                      │
                      ▼
              Ambiente Gymnasium
                      │
                      ▼
             PPO (Stable Baselines3)
                      │
                      ▼
            Modelo Treinado (.zip)
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
      Backtest                Operação Live
```

---

## 📈 Tecnologias Utilizadas

- Python
- MetaTrader5
- Pandas
- NumPy
- Gymnasium
- Stable-Baselines3
- PPO
- Reinforcement Learning

---

## 🔮 Melhorias Futuras

- Suporte para múltiplos ativos
- Multi-Asset Training
- LSTM Policy
- Transformer Policy
- SAC
- TD3
- A2C
- Optuna para otimização de hiperparâmetros
- Dashboard Web
- TensorBoard
- Banco de Dados para histórico
- Gestão dinâmica de risco
- Position Sizing Inteligente
- Stop Loss adaptativo
- Trailing Stop baseado em IA

---

## ⚠️ Aviso Legal

Este projeto possui finalidade exclusivamente **educacional e de pesquisa** em Inteligência Artificial aplicada ao mercado financeiro.

Não existe qualquer garantia de lucro.

Nunca opere em conta real antes de realizar testes extensivos em contas de demonstração.

O mercado financeiro apresenta riscos elevados e qualquer decisão de investimento é de responsabilidade exclusiva do operador.

---

## 📜 Licença

Este projeto é distribuído sob a licença **MIT**.

Sinta-se livre para estudar, modificar e contribuir.

---

## ⭐ Contribuições

Contribuições são muito bem-vindas!

Caso tenha sugestões, melhorias ou encontre algum problema, abra uma **Issue** ou envie um **Pull Request**.

---

## Desenvolvido com ❤️ utilizando Reinforcement Learning + MetaTrader 5

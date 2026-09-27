"""
Configurações do projeto.

Quase tudo que muda o comportamento da simulação e do treino está aqui — é o
melhor lugar para experimentar. Atenção: mudar os sensores ou as camadas da
rede deixa os cérebros já salvos incompatíveis (será preciso treinar de novo).
"""

from pathlib import Path

# --- Pastas ------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
PASTA_PISTAS = RAIZ / "pistas"
PASTA_CEREBROS = RAIZ / "cerebros"

# --- Mundo -------------------------------------------------------------------
# Todas as pistas ficam num "mundo" deste tamanho (unidade: pixels do mundo).
# A janela só escala o mundo para caber na tela.
LARGURA_MUNDO = 1600
ALTURA_MUNDO = 1000

# --- Simulação ---------------------------------------------------------------
DT = 1 / 60                   # passo de tempo da física (s); a 1× roda em tempo real
TEMPO_MAX_GERACAO = 45.0      # duração máxima de uma geração (s simulados)
TEMPO_SEM_PROGRESSO = 3.0     # quem passa esse tempo sem avançar é eliminado

# --- Carro -------------------------------------------------------------------
VELOCIDADE_MAX = 400.0        # px/s
ACELERACAO = 250.0            # px/s²
FRENAGEM = 600.0              # px/s²
ARRASTO = 0.3                 # perda de velocidade proporcional à velocidade (1/s)
GIRO_MAX = 3.0                # velocidade máxima de giro do carro (rad/s)
ADERENCIA = 800.0             # aceleração lateral máxima (px/s²): em alta velocidade o
                              # carro não consegue fazer curvas fechadas — precisa frear!
VELOCIDADE_GIRO_PLENO = 60.0  # abaixo desta velocidade o giro diminui (parado não gira)
COMPRIMENTO_CARRO = 24.0
LARGURA_CARRO = 12.0

# --- Sensores ----------------------------------------------------------------
# Cada sensor é um raio que mede a distância até a borda da pista.
ANGULOS_SENSORES = (-90, -60, -35, -15, 0, 15, 35, 60, 90)   # graus; 0 = para frente
ALCANCE_SENSOR = 300.0        # px
PASSO_SENSOR = 4.0            # resolução da busca ao longo do raio (px)

# --- Rede neural -------------------------------------------------------------
# Entradas: um valor por sensor + a velocidade. Saídas: direção e acelerador.
CAMADAS_OCULTAS = (12,)

# --- Algoritmo genético ------------------------------------------------------
TAMANHO_POPULACAO = 100
ELITE = 5                     # os melhores passam intactos para a geração seguinte
TAMANHO_TORNEIO = 3           # seleção: sorteia N indivíduos e escolhe o melhor deles
PROB_CRUZAMENTO = 0.5         # chance de um filho misturar os neurônios de dois pais
TAXA_MUTACAO = 0.1            # fração dos pesos alterados em cada filho
FORCA_MUTACAO = 0.3           # desvio padrão dessa alteração
FRACAO_NOVOS = 0.05           # indivíduos totalmente aleatórios em cada geração
ESCALA_INICIAL = 1.0          # desvio padrão dos pesos da primeira geração
SEMENTE = None                # um número fixo aqui torna o treino reproduzível

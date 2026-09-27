# Corrida IA 🏁

Redes neurais aprendendo a pilotar numa pista de corrida por **neuroevolução**: uma
população de 100 carros, cada um com a sua própria rede neural, dirige ao mesmo tempo.
Os que vão mais longe passam os "genes" (os pesos da rede) para a geração seguinte,
com pequenas mutações. Em poucos minutos eles aprendem a fazer as curvas — e a frear
antes delas.

![Tela de treino](docs/treino.jpg)

## Como rodar

**Jeito fácil (Windows):** dê dois cliques em `iniciar.bat`. Na primeira vez ele cria o
ambiente Python e instala as dependências.

**Pelo terminal:**

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py
```

Precisa de Python 3.10 ou mais novo. Dependências: `numpy` e `pygame-ce`.

## O que dá para fazer

| No menu | O que acontece |
|---|---|
| **Treinar do zero** | Solta 100 carros com redes aleatórias e assiste à evolução. O melhor cérebro é salvo sozinho sempre que bate o recorde. |
| **Continuar treino** | Retoma a população de onde parou (o treino é salvo a cada geração). |
| **Assistir a IA** | O melhor cérebro dirigindo sozinho. Troque de pista com ← → para ver se ele se vira numa pista que nunca viu. |
| **Correr contra a IA** | Você (setas ou WASD) contra os 3 melhores cérebros, em 3 voltas. |
| **Editor de pistas** | Clique para criar pontos, arraste para ajustar; a pista se desenha sozinha. Teste dirigindo (T) ou veja a IA nela (A). |

Dica para o treino: aperte **7** (velocidade máxima) e acompanhe o gráfico. Na pista
*Clássica* a primeira volta completa costuma sair perto da geração 10, e o tempo de
volta segue caindo por dezenas de gerações.

As pistas vêm em ordem de dificuldade: Oval → Feijão → Serpentina → Clássica → Desafio.

## Como funciona

**O carro** tem física simples: acelera, freia e vira — mas os pneus têm aderência
limitada, então em alta velocidade ele não consegue fazer curvas fechadas. Para ir
rápido, a rede precisa aprender a *frear antes das curvas*.

**Os sensores** são 9 raios (de −90° a +90°) que medem a distância até a borda da pista.
Junto com a velocidade, formam as 10 entradas da rede.

**A rede neural** é um perceptron com uma camada oculta de 12 neurônios (tanh) e 2 saídas:
direção (−1 esquerda … +1 direita) e acelerador (−1 freia … +1 acelera). Ela é
escrita só com NumPy, e a população inteira é avaliada de uma vez.

**O fitness** (a nota de cada carro) é a distância que ele percorreu no sentido da pista
antes de bater, parar de avançar por 3 s ou acabar o tempo da geração (45 s).
Quem é mais rápido anda mais no mesmo tempo, então a velocidade também é premiada.

**O algoritmo genético**, a cada geração:
1. mantém os 5 melhores intactos (elite — os carros laranja na tela);
2. escolhe pais por torneio (sorteia 3, fica com o melhor);
3. cruza os pais neurônio a neurônio;
4. muta ~10% dos pesos de cada filho com ruído gaussiano;
5. adiciona 5% de indivíduos totalmente novos, para manter a diversidade.

Não há backpropagation: a rede melhora só por seleção e mutação, como na evolução natural.

## Linha de comando

```bash
python main.py                                    # menu
python main.py treinar --pista classica           # treino com janela
python main.py treinar --pista desafio --sem-janela --geracoes 200   # bem mais rápido
python main.py treinar --pista desafio --continuar
python main.py assistir --pista serpentina
python main.py correr --pista feijao
python main.py editor --pista oval
```

O nome da pista é o do arquivo em `pistas/`, sem o `.json`.

## Experimente

Tudo que dá para ajustar está em [`corrida/config.py`](corrida/config.py):

- **Física:** `ADERENCIA` menor deixa as curvas mais traiçoeiras; `VELOCIDADE_MAX` maior exige mais freio.
- **Sensores:** menos raios, ou alcance menor, e veja se a IA ainda consegue.
- **Rede:** `CAMADAS_OCULTAS = (8,)` ou `(16, 8)`.
- **Evolução:** `TAXA_MUTACAO`, `FORCA_MUTACAO`, `ELITE`, `TAMANHO_POPULACAO`.

Mudar sensores ou camadas deixa os cérebros salvos incompatíveis (é só treinar de novo).

## Estrutura

```
main.py                 ponto de entrada (menu e linha de comando)
corrida/
  config.py             parâmetros da simulação, da rede e da evolução
  pista.py              spline da pista, máscara de asfalto, sensores e progresso
  carros.py             física, colisão e voltas (toda a frota de uma vez, com NumPy)
  rede_neural.py        a rede neural (MLP) em NumPy
  evolucao.py           o algoritmo genético
  treino.py             o laço de gerações (também roda sem janela)
  arquivos.py           salvar/carregar cérebros e o estado do treino
  desenho.py            desenho da pista, carros, sensores, gráfico e rede
  interface.py, app.py  janela, fontes, botões e painel
  telas/                menu, treino, assistir, corrida e editor
pistas/                 as pistas (JSON: nome, largura e pontos de controle)
cerebros/               criada no primeiro treino: <pista>.npz e <pista>.treino.npz
tests/                  testes (python -m unittest discover tests)
```

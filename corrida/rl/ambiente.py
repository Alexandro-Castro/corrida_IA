"""
O ambiente do Aprendizado por Reforço.

Na neuroevolução, cada carro dirige uma vez por geração e só no fim recebe uma nota.
No Aprendizado por Reforço o carro recebe uma *recompensa a cada passo* — e é com
esse sinal, passo a passo, que o PPO ajusta a rede.

- Observação: as mesmas 10 entradas da neuroevolução (9 sensores + velocidade).
- Ação: direção e acelerador, de -1 a 1 (as mesmas 2 saídas).
- Recompensa: quanto o carro avançou no sentido da pista naquele passo
  (andar para trás dá recompensa negativa). Bater custa uma penalidade.
- Episódio: termina quando o carro bate ou fica parado; é cortado no tempo máximo.

Vários carros rodam em paralelo (um "ambiente vetorizado") e cada um recomeça sozinho,
de preferência num ponto aleatório da pista — assim todos os trechos da pista são
treinados desde o começo, e não só a primeira curva.
"""

from collections import deque

import numpy as np

from .. import config
from ..carros import Frota

ESCALA_RECOMPENSA = 0.1      # avançar à velocidade máxima rende 0,1 por passo
PENALIDADE_BATIDA = 1.0


class AmbienteCorrida:
    def __init__(self, pistas, carros_por_pista=64, largada_aleatoria=True,
                 tempo_max=config.TEMPO_MAX_GERACAO, semente=None):
        self.pistas = [p.preparar() for p in pistas]
        self.m = carros_por_pista
        self.n = len(self.pistas) * carros_por_pista
        self.largada_aleatoria = largada_aleatoria
        self.max_passos = int(round(tempo_max / config.DT))
        self.rng = np.random.default_rng(semente)
        self.frotas = [Frota(p, carros_por_pista) for p in self.pistas]
        self.n_observacoes = len(config.ANGULOS_SENSORES) + 1
        self.passos = np.zeros(self.n, dtype=int)
        self.retorno = np.zeros(self.n)
        self.episodios = deque(maxlen=200)       # (retorno, voltas, segundos) dos últimos episódios
        for k, frota in enumerate(self.frotas):
            self._recomecar(k, np.arange(frota.n))

    def _recomecar(self, k, idx):
        frota = self.frotas[k]
        n = len(idx)
        if self.largada_aleatoria:
            folga = max(frota.pista.meia_largura - 14.0, 0.0) * 0.5
            frota.recolocar(idx,
                            s=self.rng.uniform(0, frota.pista.comprimento, n),
                            lateral=self.rng.uniform(-folga, folga, n),
                            desvio=self.rng.uniform(-0.25, 0.25, n),
                            velocidade=self.rng.uniform(0, 0.6, n) * config.VELOCIDADE_MAX)
        else:
            frota.recolocar(idx, s=np.zeros(n))
        global_ = k * self.m + idx
        self.passos[global_] = 0
        self.retorno[global_] = 0.0

    def observar(self):
        return np.concatenate([f.entradas(np.arange(f.n)) for f in self.frotas]).astype(np.float32)

    def passo(self, acoes):
        """Aplica as ações (n, 2) e devolve (observação, recompensa, terminou, cortado, observação_final).
        Os carros cujo episódio acabou já voltam recomeçados; `observação_final` é o que eles
        viam no instante em que acabou (o PPO precisa disso para os episódios cortados pelo tempo)."""
        acoes = np.clip(acoes, -1.0, 1.0)
        recompensa = np.zeros(self.n)
        terminou = np.zeros(self.n, dtype=bool)
        for k, frota in enumerate(self.frotas):
            fatia = slice(k * self.m, (k + 1) * self.m)
            antes = frota.progresso.copy()
            frota.passo(acoes[fatia, 0], acoes[fatia, 1], np.arange(frota.n))
            avanco = frota.progresso - antes
            recompensa[fatia] = avanco / (config.VELOCIDADE_MAX * config.DT) * ESCALA_RECOMPENSA
            terminou[fatia] = ~frota.vivo
        recompensa[terminou] -= PENALIDADE_BATIDA
        self.passos += 1
        self.retorno += recompensa
        cortado = (self.passos >= self.max_passos) & ~terminou
        observacao_final = self.observar()

        acabou = terminou | cortado
        if acabou.any():
            for k, frota in enumerate(self.frotas):
                idx = np.flatnonzero(acabou[k * self.m:(k + 1) * self.m])
                if idx.size == 0:
                    continue
                for i in idx:
                    self.episodios.append((self.retorno[k * self.m + i],
                                           frota.melhor_progresso[i] / frota.pista.comprimento,
                                           self.passos[k * self.m + i] * config.DT))
                self._recomecar(k, idx)
            observacao = self.observar()
        else:
            observacao = observacao_final
        return observacao, recompensa, terminou, cortado, observacao_final

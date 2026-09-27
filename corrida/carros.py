"""
Os carros: física simples, colisão com a borda da pista, sensores e medição do
progresso. Tudo vetorizado com NumPy — a frota inteira anda de uma vez.
"""

import numpy as np

from . import config

_CANTOS = np.array([(1, 1), (1, -1), (-1, 1), (-1, -1)], dtype=float) * (
    config.COMPRIMENTO_CARRO / 2, config.LARGURA_CARRO / 2)


class Frota:
    """Estado de N carros numa pista.

    modo "treino":  bateu ou ficou sem avançar → eliminado.
    modo "corrida": bateu → volta para o meio da pista, parado (e perde tempo).
                    `reposicionar_parados` (um valor ou um por carro) faz o mesmo com
                    quem fica sem avançar — útil para a IA, não para o jogador.
    """

    def __init__(self, pista, n, modo="treino", grade=False, reposicionar_parados=False):
        self.pista = pista.preparar()
        self.n = n
        self.modo = modo
        self.reposicionar_parados = np.broadcast_to(np.asarray(reposicionar_parados, dtype=bool), (n,)).copy()
        self.angulos_sensores = np.radians(config.ANGULOS_SENSORES)
        self._largar(grade)

    def _largar(self, grade):
        n = self.n
        indices = np.arange(n)
        if grade:
            # grid de largada: fila dupla, atrás da linha de chegada
            recuo = 18.0 + 34.0 * (indices // 2) + 14.0 * (indices % 2)
            lateral = np.where(indices % 2 == 0, -1.0, 1.0) * self.pista.meia_largura * 0.4
        else:
            recuo = np.zeros(n)
            lateral = np.zeros(n)
        x, y, ang = self.pista.ponto_em(-recuo)
        self.x = x - np.sin(ang) * lateral
        self.y = y + np.cos(ang) * lateral
        self.angulo = np.array(ang, dtype=float)
        self.velocidade = np.zeros(n)
        self.direcao = np.zeros(n)          # últimos comandos (-1..1), para desenhar
        self.acelerador = np.zeros(n)
        self.vivo = np.ones(n, dtype=bool)
        self.s = self.pista.projetar(self.x, self.y)
        self.progresso = -recuo             # distância percorrida no sentido da pista
        self.melhor_progresso = self.progresso.copy()
        self.sem_avanco = np.zeros(n)       # tempo desde o último avanço
        self.tempo = 0.0
        self.batidas = np.zeros(n, dtype=int)
        self.voltas = np.zeros(n, dtype=int)
        self.inicio_volta = np.zeros(n)
        self.tempos_volta = [[] for _ in range(n)]
        self.distancias = np.full((n, len(self.angulos_sensores)), config.ALCANCE_SENSOR)
        self._ler_sensores(indices)

    # ------------------------------------------------------------------ passo
    def passo(self, direcao, acelerador, idx=None):
        """Avança a física um passo de tempo para os carros `idx` (padrão: todos os vivos),
        com os comandos dados: direção (-1 esquerda .. 1 direita) e acelerador (-1 freia .. 1 acelera)."""
        dt = config.DT
        self.tempo += dt
        if idx is None:
            idx = np.flatnonzero(self.vivo)
        if idx.size == 0:
            return
        d = np.clip(direcao, -1.0, 1.0)
        a = np.clip(acelerador, -1.0, 1.0)
        self.direcao[idx] = d
        self.acelerador[idx] = a

        v = self.velocidade[idx]
        forca = np.where(a >= 0, a * config.ACELERACAO, a * config.FRENAGEM)
        v = np.clip(v + (forca - config.ARRASTO * v) * dt, 0.0, config.VELOCIDADE_MAX)
        # o volante gira o carro, mas a aderência dos pneus limita a curva em alta velocidade
        giro = d * config.GIRO_MAX * np.minimum(1.0, v / config.VELOCIDADE_GIRO_PLENO)
        limite = config.ADERENCIA / np.maximum(v, 1.0)
        giro = np.clip(giro, -limite, limite)
        ang = self.angulo[idx] + giro * dt
        x0 = self.x[idx]
        y0 = self.y[idx]
        x = x0 + np.cos(ang) * v * dt
        y = y0 + np.sin(ang) * v * dt
        self.x[idx] = x
        self.y[idx] = y
        self.angulo[idx] = ang
        self.velocidade[idx] = v

        # colisão: os quatro cantos (e o meio do deslocamento) precisam estar no asfalto
        ok = self._no_asfalto(x, y, ang) & self.pista.livre((x + x0) / 2, (y + y0) / 2)
        bateram = idx[~ok]
        seguem = idx[ok]
        self._medir_progresso(seguem)

        parados = seguem[self.sem_avanco[seguem] > config.TEMPO_SEM_PROGRESSO]
        if self.modo == "treino":
            self.vivo[bateram] = False
            self.vivo[parados] = False
        else:
            self.batidas[bateram] += 1
            self._reposicionar(bateram)
            self._reposicionar(parados[self.reposicionar_parados[parados]])
        self._ler_sensores(idx[self.vivo[idx]])

    def _medir_progresso(self, idx):
        if idx.size == 0:
            return
        L = self.pista.comprimento
        s = self.pista.projetar(self.x[idx], self.y[idx])
        delta = (s - self.s[idx] + L / 2) % L - L / 2     # atravessar a largada não dá salto
        self.s[idx] = s
        self.progresso[idx] += delta
        avancou = self.progresso[idx] > self.melhor_progresso[idx]
        self.melhor_progresso[idx] = np.maximum(self.melhor_progresso[idx], self.progresso[idx])
        self.sem_avanco[idx] = np.where(avancou, 0.0, self.sem_avanco[idx] + config.DT)

        voltas = np.maximum(np.floor(self.melhor_progresso[idx] / L), 0).astype(int)
        for i in idx[voltas > self.voltas[idx]]:
            self.tempos_volta[i].append(self.tempo - self.inicio_volta[i])
            self.inicio_volta[i] = self.tempo
        self.voltas[idx] = voltas

    def _reposicionar(self, idx):
        """Coloca os carros de volta no meio da pista, apontando para a frente e parados."""
        if idx.size == 0:
            return
        x, y, ang = self.pista.ponto_em(self.s[idx])
        self.x[idx] = x
        self.y[idx] = y
        self.angulo[idx] = ang
        self.velocidade[idx] = 0.0
        self.sem_avanco[idx] = 0.0

    def _no_asfalto(self, x, y, ang):
        c = np.cos(ang)[:, None]
        s = np.sin(ang)[:, None]
        cx = x[:, None] + _CANTOS[:, 0] * c - _CANTOS[:, 1] * s
        cy = y[:, None] + _CANTOS[:, 0] * s + _CANTOS[:, 1] * c
        return self.pista.livre(cx, cy).all(axis=1)

    def _ler_sensores(self, idx):
        if idx.size == 0:
            return
        angulos = self.angulo[idx, None] + self.angulos_sensores
        self.distancias[idx] = self.pista.medir(self.x[idx], self.y[idx], angulos)

    # ------------------------------------------------------------------ consultas
    def entradas(self, idx):
        """O que a rede neural 'vê': sensores (0 = parede colada, 1 = livre) e velocidade (0..1)."""
        return np.column_stack([self.distancias[idx] / config.ALCANCE_SENSOR,
                                self.velocidade[idx] / config.VELOCIDADE_MAX])

    def lider(self):
        """Índice do carro vivo mais adiantado (ou do que foi mais longe, se todos pararam)."""
        if self.vivo.any():
            vivos = np.flatnonzero(self.vivo)
            return int(vivos[np.argmax(self.progresso[vivos])])
        return int(np.argmax(self.melhor_progresso))

    def melhor_volta(self, i=None):
        tempos = self.tempos_volta[i] if i is not None else [t for ts in self.tempos_volta for t in ts]
        return min(tempos) if tempos else float("nan")

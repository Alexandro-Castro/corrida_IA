"""
Pista de corrida.

Uma pista é definida por alguns pontos de controle (os que você posiciona no
editor) e por uma largura. A partir deles calculamos:

- a linha central: uma curva suave (spline Catmull-Rom) que passa pelos pontos;
- para cada pixel do mundo, a distância até a linha central (é asfalto quando
  for menor que meia largura) e qual trecho da linha central está mais perto.

Esse "mapa de trechos" diz instantaneamente quanto cada carro já andou ao
longo da pista — é a base do fitness (a nota de cada rede neural).
"""

import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np

from . import config

ESPACAMENTO = 8.0      # distância entre pontos consecutivos da linha central (px)
LARGURA_MIN = 50.0
LARGURA_MAX = 160.0


def spline_fechada(pontos, amostras=24, alpha=0.5):
    """Curva Catmull-Rom centrípeta, fechada, que passa por todos os pontos."""
    p = np.asarray(pontos, dtype=float)
    p0, p1, p2, p3 = (np.roll(p, k, axis=0) for k in (1, 0, -1, -2))

    def intervalo(a, b):
        return np.maximum(np.hypot(*(b - a).T) ** alpha, 1e-3)

    t0 = np.zeros(len(p))
    t1 = t0 + intervalo(p0, p1)
    t2 = t1 + intervalo(p1, p2)
    t3 = t2 + intervalo(p2, p3)
    t = t1[:, None] + (t2 - t1)[:, None] * (np.arange(amostras) / amostras)

    def mistura(a, b, ta, tb):
        w = ((t - ta[:, None]) / (tb - ta)[:, None])[..., None]
        return a + (b - a) * w

    q0, q1, q2, q3 = (q[:, None, :] for q in (p0, p1, p2, p3))
    a1 = mistura(q0, q1, t0, t1)
    a2 = mistura(q1, q2, t1, t2)
    a3 = mistura(q2, q3, t2, t3)
    b1 = mistura(a1, a2, t0, t2)
    b2 = mistura(a2, a3, t1, t3)
    return mistura(b1, b2, t1, t2).reshape(-1, 2)


def reamostrar_fechada(linha, espacamento):
    """Redistribui os pontos de um laço fechado com espaçamento uniforme."""
    fechada = np.vstack([linha, linha[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(fechada, axis=0).T))])
    n = max(int(round(s[-1] / espacamento)), 12)
    alvo = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(alvo, s, fechada[:, 0]), np.interp(alvo, s, fechada[:, 1])])


def nome_para_arquivo(nome):
    """'Minha Pista 2' → 'minha_pista_2'."""
    s = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower() or "pista"


class Pista:
    def __init__(self, pontos, largura=80.0, nome="Nova pista", arquivo=None, ordem=None):
        if len(pontos) < 3:
            raise ValueError("Uma pista precisa de pelo menos 3 pontos.")
        self.pontos = np.asarray(pontos, dtype=float)
        self.largura = float(largura)
        self.meia_largura = self.largura / 2
        self.nome = nome
        self.arquivo = Path(arquivo) if arquivo else None
        self.ordem = ordem

        # Linha central: trecho i vai de centro[i] até centro[i + 1] (laço fechado).
        self.centro = reamostrar_fechada(spline_fechada(self.pontos), ESPACAMENTO)
        self.vetor_seg = np.roll(self.centro, -1, axis=0) - self.centro
        self.comp_seg = np.hypot(*self.vetor_seg.T)
        self.s_inicio = np.concatenate([[0.0], np.cumsum(self.comp_seg)[:-1]])
        self.comprimento = float(self.comp_seg.sum())
        self._inv_comp2 = 1.0 / np.maximum(self.comp_seg ** 2, 1e-9)
        self.angulo_largada = math.atan2(self.vetor_seg[0, 1], self.vetor_seg[0, 0])

        self.mascara = None       # (H, W) bool: True = asfalto
        self.mapa_seg = None      # (H, W) int: trecho da linha central mais próximo
        self._campos = None
        passos = int(config.ALCANCE_SENSOR / config.PASSO_SENSOR)
        self._passos = np.arange(1, passos + 1) * config.PASSO_SENSOR

    # ------------------------------------------------------------------ arquivos
    @property
    def id(self):
        return self.arquivo.stem if self.arquivo else nome_para_arquivo(self.nome)

    @classmethod
    def carregar(cls, caminho):
        caminho = Path(caminho)
        with open(caminho, encoding="utf-8") as f:
            dados = json.load(f)
        return cls(dados["pontos"], dados.get("largura", 80), dados.get("nome", caminho.stem),
                   caminho, dados.get("ordem"))

    def salvar(self, caminho):
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        campos = [f'  "nome": {json.dumps(self.nome, ensure_ascii=False)}',
                  f'  "largura": {round(self.largura)}']
        if self.ordem is not None:
            campos.append(f'  "ordem": {self.ordem}')
        pontos = ",\n    ".join(f"[{round(x)}, {round(y)}]" for x, y in self.pontos)
        campos.append(f'  "pontos": [\n    {pontos}\n  ]')
        caminho.write_text("{\n" + ",\n".join(campos) + "\n}\n", encoding="utf-8")
        self.arquivo = caminho

    @staticmethod
    def listar():
        """Arquivos de pista: primeiro as que vêm com o projeto (por dificuldade), depois as suas."""
        encontrados = []
        for caminho in config.PASTA_PISTAS.glob("*.json"):
            try:
                with open(caminho, encoding="utf-8") as f:
                    ordem = json.load(f).get("ordem")
            except (OSError, ValueError):
                continue
            encontrados.append((ordem if ordem is not None else 999, caminho.stem, caminho))
        return [caminho for *_, caminho in sorted(encontrados)]

    # ------------------------------------------------------------------ raster
    def campos(self, escala=1.0, margem=14.0):
        """Para cada pixel de uma grade que cobre o mundo (na escala dada): a distância até a
        linha central, o trecho mais próximo, a posição ao longo da pista (s) e o lado."""
        if escala == 1.0 and self._campos is not None:
            return self._campos
        W = round(config.LARGURA_MUNDO * escala)
        H = round(config.ALTURA_MUNDO * escala)
        dist = np.full((H, W), np.inf, np.float32)
        seg = np.full((H, W), -1, np.int32)
        t_seg = np.zeros((H, W), np.float32)
        lado = np.zeros((H, W), np.float32)
        alcance = self.meia_largura + margem
        dados = np.column_stack([self.centro, self.vetor_seg, self.comp_seg]).tolist()
        for i, (ax, ay, vx, vy, comp) in enumerate(dados):
            # só os pixels na vizinhança deste trecho
            j0 = max(int((min(ax, ax + vx) - alcance) * escala), 0)
            j1 = min(int((max(ax, ax + vx) + alcance) * escala) + 2, W)
            i0 = max(int((min(ay, ay + vy) - alcance) * escala), 0)
            i1 = min(int((max(ay, ay + vy) + alcance) * escala) + 2, H)
            if j0 >= j1 or i0 >= i1:
                continue
            dx = (np.arange(j0, j1, dtype=np.float32) + 0.5) / escala - ax
            dy = (np.arange(i0, i1, dtype=np.float32)[:, None] + 0.5) / escala - ay
            t = np.clip((dx * vx + dy * vy) / (comp * comp), 0.0, 1.0)
            ex = dx - t * vx
            ey = dy - t * vy
            d = np.sqrt(ex * ex + ey * ey)
            bloco = dist[i0:i1, j0:j1]
            perto = d < bloco
            bloco[perto] = d[perto]
            seg[i0:i1, j0:j1][perto] = i
            t_seg[i0:i1, j0:j1][perto] = t[perto]
            lado[i0:i1, j0:j1][perto] = np.copysign(d, vx * dy - vy * dx)[perto]
        valido = np.maximum(seg, 0)
        s = (self.s_inicio[valido] + t_seg * self.comp_seg[valido]).astype(np.float32)
        campos = {"dist": dist, "seg": seg, "s": s, "lado": lado, "escala": escala}
        if escala == 1.0:
            self._campos = campos
        return campos

    def preparar(self):
        """Calcula a máscara de asfalto e o mapa de trechos (necessários para simular)."""
        if self.mascara is None:
            c = self.campos(1.0)
            self.mascara = c["dist"] <= self.meia_largura
            self.mapa_seg = c["seg"]
        return self

    def liberar_campos(self):
        """Descarta os campos usados só para desenhar (a simulação continua funcionando)."""
        self._campos = None

    # ------------------------------------------------------------------ consultas
    def livre(self, x, y):
        """True onde (x, y) está sobre o asfalto."""
        x = np.asarray(x)
        y = np.asarray(y)
        H, W = self.mascara.shape
        dentro = (x >= 0) & (y >= 0) & (x < W) & (y < H)
        ix = np.clip(x.astype(np.intp), 0, W - 1)
        iy = np.clip(y.astype(np.intp), 0, H - 1)
        return self.mascara[iy, ix] & dentro

    def projetar(self, x, y):
        """Posição ao longo da linha central (0 ≤ s < comprimento) mais próxima de (x, y)."""
        H, W = self.mapa_seg.shape
        ix = np.clip(np.asarray(x).astype(np.intp), 0, W - 1)
        iy = np.clip(np.asarray(y).astype(np.intp), 0, H - 1)
        i = np.maximum(self.mapa_seg[iy, ix], 0)
        t = ((x - self.centro[i, 0]) * self.vetor_seg[i, 0]
             + (y - self.centro[i, 1]) * self.vetor_seg[i, 1]) * self._inv_comp2[i]
        return self.s_inicio[i] + np.clip(t, 0.0, 1.0) * self.comp_seg[i]

    def ponto_em(self, s):
        """Ponto da linha central e direção da pista na posição s (aceita arrays)."""
        s = np.asarray(s, dtype=float) % self.comprimento
        i = np.clip(np.searchsorted(self.s_inicio, s, side="right") - 1, 0, len(self.centro) - 1)
        t = (s - self.s_inicio[i]) / self.comp_seg[i]
        x = self.centro[i, 0] + t * self.vetor_seg[i, 0]
        y = self.centro[i, 1] + t * self.vetor_seg[i, 1]
        return x, y, np.arctan2(self.vetor_seg[i, 1], self.vetor_seg[i, 0])

    def medir(self, x, y, angulos):
        """Distância até a borda da pista, partindo de (x, y), em cada direção.
        x, y: (N,)   angulos: (N, R) em radianos   →   (N, R)"""
        c = np.cos(angulos)
        s = np.sin(angulos)
        passos = self._passos
        # amostra pontos ao longo de cada raio e procura o primeiro fora do asfalto
        fora = ~self.livre(x[:, None, None] + c[..., None] * passos,
                           y[:, None, None] + s[..., None] * passos)
        primeiro = fora.argmax(axis=2)
        bateu = np.take_along_axis(fora, primeiro[..., None], axis=2)[..., 0]
        # refina por bisseção entre a última amostra livre e a primeira fora da pista
        alto = passos[primeiro]
        baixo = alto - config.PASSO_SENSOR
        x = x[:, None]
        y = y[:, None]
        for _ in range(3):
            meio = 0.5 * (baixo + alto)
            livre = self.livre(x + c * meio, y + s * meio)
            baixo = np.where(livre, meio, baixo)
            alto = np.where(livre, alto, meio)
        return np.where(bateu, 0.5 * (baixo + alto), config.ALCANCE_SENSOR)

    def trechos_sobrepostos(self, folga=12.0):
        """Pontos da linha central perto demais de outra parte da pista: as faixas de
        asfalto se encostariam e o carro poderia 'pular' de uma para a outra."""
        c = self.centro
        d = np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1])
        ds = np.abs(self.s_inicio[:, None] - self.s_inicio[None, :])
        ds = np.minimum(ds, self.comprimento - ds)
        limite = self.largura + folga
        return ((d < limite) & (ds > 1.6 * limite)).any(axis=1)

    def fora_do_mapa(self):
        """Pontos da linha central cujo asfalto seria cortado pela borda do mundo."""
        x, y = self.centro.T
        w = self.meia_largura
        return (x < w) | (y < w) | (x > config.LARGURA_MUNDO - w) | (y > config.ALTURA_MUNDO - w)

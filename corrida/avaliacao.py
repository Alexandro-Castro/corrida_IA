"""
Avaliação justa: o mesmo teste para qualquer técnica de treino.

O carro larga parado da linha de chegada e dirige até bater, parar de avançar ou
acabar o tempo — exatamente como um carro numa geração da neuroevolução. A nota é
a mesma do fitness: quantas voltas ele andou.
"""

import math
from dataclasses import dataclass

import numpy as np

from . import config
from .carros import Frota


@dataclass
class Resultado:
    voltas: float              # distância percorrida, em voltas
    melhor_volta: float        # segundos (nan se não completou nenhuma)
    bateu: bool                # parou antes do tempo (bateu ou ficou sem avançar)
    tempo: float               # quanto tempo dirigiu


def avaliar(pista, agir, tempo_max=config.TEMPO_MAX_GERACAO):
    """`agir` recebe as entradas (1, n) e devolve os comandos (1, 2) — por exemplo, Piloto.agir."""
    frota = Frota(pista, 1)
    while frota.vivo[0] and frota.tempo < tempo_max - 1e-9:
        comandos = agir(frota.entradas(np.array([0])))
        frota.passo(comandos[:, 0], comandos[:, 1])
    return Resultado(voltas=float(frota.melhor_progresso[0]) / pista.comprimento,
                     melhor_volta=frota.melhor_volta(0),
                     bateu=not frota.vivo[0] and frota.tempo < tempo_max - 1e-9,
                     tempo=frota.tempo)


def resumo(resultados):
    """Média de voltas e melhor volta de uma lista de resultados (várias pistas)."""
    voltas = [r.melhor_volta for r in resultados if not math.isnan(r.melhor_volta)]
    return float(np.mean([r.voltas for r in resultados])), (min(voltas) if voltas else math.nan)

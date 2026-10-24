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


def avaliar_todos(pista, agires, tempo_max=config.TEMPO_MAX_GERACAO):
    """Avalia vários pilotos ao mesmo tempo, cada um no seu carro (os carros não interagem).
    Cada `agir` recebe as entradas (1, n) e devolve os comandos (1, 2) — por exemplo, Piloto.agir."""
    frota = Frota(pista, len(agires))
    fim = np.full(len(agires), tempo_max)
    while frota.vivo.any() and frota.tempo < tempo_max - 1e-9:
        vivos = np.flatnonzero(frota.vivo)
        entradas = frota.entradas(vivos)
        comandos = np.array([agires[i](entradas[k:k + 1])[0] for k, i in enumerate(vivos)])
        frota.passo(comandos[:, 0], comandos[:, 1], vivos)
        fim[vivos[~frota.vivo[vivos]]] = frota.tempo
    return [Resultado(voltas=float(frota.melhor_progresso[i]) / pista.comprimento,
                      melhor_volta=frota.melhor_volta(i),
                      bateu=bool(fim[i] < tempo_max - 1e-9),
                      tempo=float(fim[i]))
            for i in range(len(agires))]


def avaliar(pista, agir, tempo_max=config.TEMPO_MAX_GERACAO):
    return avaliar_todos(pista, [agir], tempo_max)[0]


def resumo(resultados):
    """Média de voltas e melhor volta de uma lista de resultados (várias pistas)."""
    voltas = [r.melhor_volta for r in resultados if not math.isnan(r.melhor_volta)]
    return float(np.mean([r.voltas for r in resultados])), (min(voltas) if voltas else math.nan)

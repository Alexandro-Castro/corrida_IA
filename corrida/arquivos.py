"""
Salvar e carregar cérebros (as melhores redes treinadas) e o estado do treino.

  cerebros/<pista>.npz          neuroevolução: os 5 melhores genomas da geração recorde
  cerebros/<pista>.treino.npz   neuroevolução: a população inteira, para continuar depois
  cerebros/<pista>.ppo.npz      PPO: a rede do ator (a parte que dirige), no mesmo formato
  cerebros/<pista>.ppo.pt       PPO: ator, crítico e otimizador, para continuar depois

Cada cérebro guarda a própria arquitetura: redes de tamanhos diferentes (e de
técnicas diferentes) convivem — só os sensores precisam ser os mesmos do config.
"""

import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config
from .rede_neural import Rede

METODOS = {"neuroevolucao": "Neuroevolução", "ppo": "PPO"}


class CerebroIncompativel(Exception):
    pass


@dataclass
class Cerebro:
    arquivo: Path
    genomas: np.ndarray        # (k, G), do melhor para o pior
    fitness: np.ndarray        # progresso (px) de cada genoma na avaliação em que foi salvo
    tamanhos: tuple            # arquitetura da rede: (entradas, ocultas..., saídas)
    metodo: str                # "neuroevolucao" ou "ppo"
    pista_id: str
    pista_nome: str
    comprimento_pista: float
    geracao: int               # geração (neuroevolução) ou atualização (PPO) em que foi salvo
    melhor_volta: float        # segundos (nan se não completou nenhuma volta)
    experiencia: float         # segundos de direção simulados até aqui, somando todos os carros

    @property
    def voltas(self):
        return float(self.fitness[0]) / self.comprimento_pista

    @property
    def rede(self):
        return Rede(self.tamanhos)

    @property
    def nome_metodo(self):
        return METODOS.get(self.metodo, self.metodo)

    @property
    def descricao(self):
        return f"{self.nome_metodo} · {self.pista_nome}"


def caminho_cerebro(pista_id, metodo="neuroevolucao"):
    sufixo = "" if metodo == "neuroevolucao" else f".{metodo}"
    return config.PASTA_CEREBROS / f"{pista_id}{sufixo}.npz"


def caminho_treino(pista_id):
    return config.PASTA_CEREBROS / f"{pista_id}.treino.npz"


def caminho_treino_ppo(pista_id):
    return config.PASTA_CEREBROS / f"{pista_id}.ppo.pt"


def _gravar(caminho, **dados):
    """Grava num arquivo temporário e troca no fim: um arquivo nunca fica pela metade."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(caminho.name + ".tmp")
    with open(temporario, "wb") as f:
        np.savez(f, **dados)
    os.replace(temporario, caminho)


def _angulos():
    return np.array(config.ANGULOS_SENSORES, dtype=float)


def _assinatura_treino():
    return {"tamanhos": np.array(Rede.padrao().tamanhos), "angulos": _angulos()}


def _conferir_treino(dados, caminho):
    """O treino salvo só continua com exatamente a mesma rede e os mesmos sensores."""
    for chave, valor in _assinatura_treino().items():
        if chave not in dados or dados[chave].shape != valor.shape or not np.allclose(dados[chave], valor):
            raise CerebroIncompativel(f"{Path(caminho).name} foi treinado com outra configuração de sensores ou de rede.")


def _conferir_cerebro(dados, caminho):
    """Um cérebro pronto funciona com qualquer arquitetura, desde que veja os mesmos sensores."""
    angulos = _angulos()
    tamanhos = dados["tamanhos"]
    if (dados["angulos"].shape != angulos.shape or not np.allclose(dados["angulos"], angulos)
            or tamanhos[0] != len(angulos) + 1 or tamanhos[-1] != 2):
        raise CerebroIncompativel(f"{Path(caminho).name} foi treinado com outros sensores.")


def salvar_cerebro(pista, genomas, fitness, geracao, melhor_volta, tamanhos, metodo="neuroevolucao",
                   experiencia=math.nan):
    _gravar(caminho_cerebro(pista.id, metodo), genomas=genomas, fitness=fitness, geracao=geracao,
            tamanhos=np.array(tamanhos), angulos=_angulos(), metodo=np.array(metodo),
            pista_id=np.array(pista.id), pista_nome=np.array(pista.nome), comprimento=pista.comprimento,
            melhor_volta=melhor_volta, experiencia=experiencia,
            data=np.array(datetime.now().isoformat(timespec="seconds")))


def carregar_cerebro(caminho):
    with np.load(caminho) as d:
        _conferir_cerebro(d, caminho)
        return Cerebro(arquivo=Path(caminho), genomas=d["genomas"], fitness=d["fitness"],
                       tamanhos=tuple(int(t) for t in d["tamanhos"]),
                       metodo=str(d["metodo"]) if "metodo" in d.files else "neuroevolucao",
                       pista_id=str(d["pista_id"]), pista_nome=str(d["pista_nome"]),
                       comprimento_pista=float(d["comprimento"]), geracao=int(d["geracao"]),
                       melhor_volta=float(d["melhor_volta"]),
                       experiencia=float(d["experiencia"]) if "experiencia" in d.files else math.nan)


def listar_cerebros():
    if not config.PASTA_CEREBROS.exists():
        return []
    return sorted(p for p in config.PASTA_CEREBROS.glob("*.npz") if not p.name.endswith(".treino.npz"))


def cerebros_disponiveis():
    """Todos os cérebros salvos que funcionam com os sensores atuais."""
    cerebros = []
    for caminho in listar_cerebros():
        try:
            cerebros.append(carregar_cerebro(caminho))
        except (CerebroIncompativel, OSError, KeyError, ValueError):
            pass
    return cerebros


def cerebro_da_pista(pista, metodo="neuroevolucao"):
    """O cérebro treinado nesta pista com o método dado, ou None se não houver."""
    caminho = caminho_cerebro(pista.id, metodo)
    if not caminho.exists():
        return None
    try:
        return carregar_cerebro(caminho)
    except (CerebroIncompativel, OSError, KeyError, ValueError):
        return None


def recorde_salvo(pista, metodo="neuroevolucao"):
    """Fitness do cérebro salvo — um treino novo só o substitui se fizer melhor.
    Se a pista foi editada desde então, o recorde antigo não vale mais."""
    cerebro = cerebro_da_pista(pista, metodo)
    if cerebro is None or abs(cerebro.comprimento_pista - pista.comprimento) > 1.0:
        return -math.inf
    return float(cerebro.fitness[0])


def salvar_treino(pista, ag, historico_volta, historico_experiencia):
    _gravar(caminho_treino(pista.id), historico_volta=np.array(historico_volta, dtype=float),
            historico_experiencia=np.array(historico_experiencia, dtype=float),
            comprimento=pista.comprimento, **ag.estado(), **_assinatura_treino())


def geracao_do_treino(pista):
    """Geração em que o treino salvo desta pista parou (None se não houver ou se for de outra rede)."""
    caminho = caminho_treino(pista.id)
    if not caminho.exists():
        return None
    try:
        with np.load(caminho) as d:
            _conferir_treino(d, caminho)
            return int(d["geracao"])
    except (CerebroIncompativel, OSError, KeyError, ValueError):
        return None


def carregar_treino(pista):
    """Estado do treino salvo, ou None se não houver (ou se for de outra configuração)."""
    caminho = caminho_treino(pista.id)
    if not caminho.exists():
        return None
    try:
        with np.load(caminho) as d:
            _conferir_treino(d, caminho)
            return {chave: d[chave] for chave in d.files}
    except (CerebroIncompativel, OSError, KeyError, ValueError):
        return None

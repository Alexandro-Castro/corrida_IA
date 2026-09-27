"""
Salvar e carregar cérebros (as melhores redes treinadas) e o estado do treino.

  cerebros/<pista>.npz          os 5 melhores genomas da geração recorde
  cerebros/<pista>.treino.npz   a população inteira, para continuar o treino depois
"""

import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config
from .rede_neural import Rede


class CerebroIncompativel(Exception):
    pass


@dataclass
class Cerebro:
    arquivo: Path
    genomas: np.ndarray        # (k, G), do melhor para o pior
    fitness: np.ndarray        # progresso (px) de cada genoma na geração em que foi salvo
    pista_id: str
    pista_nome: str
    comprimento_pista: float
    geracao: int
    melhor_volta: float        # segundos (nan se não completou nenhuma volta)

    @property
    def voltas(self):
        return float(self.fitness[0]) / self.comprimento_pista


def caminho_cerebro(pista_id):
    return config.PASTA_CEREBROS / f"{pista_id}.npz"


def caminho_treino(pista_id):
    return config.PASTA_CEREBROS / f"{pista_id}.treino.npz"


def _gravar(caminho, **dados):
    """Grava num arquivo temporário e troca no fim: um arquivo nunca fica pela metade."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    temporario = caminho.with_name(caminho.name + ".tmp")
    with open(temporario, "wb") as f:
        np.savez(f, **dados)
    os.replace(temporario, caminho)


def _assinatura():
    return {"tamanhos": np.array(Rede.padrao().tamanhos),
            "angulos": np.array(config.ANGULOS_SENSORES, dtype=float)}


def _conferir(dados, caminho):
    esperado = _assinatura()
    for chave, valor in esperado.items():
        if chave not in dados or dados[chave].shape != valor.shape or not np.allclose(dados[chave], valor):
            raise CerebroIncompativel(
                f"{Path(caminho).name} foi treinado com outra configuração de sensores ou de rede.")


def salvar_cerebro(pista, genomas, fitness, geracao, melhor_volta):
    _gravar(caminho_cerebro(pista.id), genomas=genomas, fitness=fitness, geracao=geracao,
            pista_id=np.array(pista.id), pista_nome=np.array(pista.nome),
            comprimento=pista.comprimento, melhor_volta=melhor_volta,
            data=np.array(datetime.now().isoformat(timespec="seconds")), **_assinatura())


def carregar_cerebro(caminho):
    with np.load(caminho) as d:
        _conferir(d, caminho)
        return Cerebro(arquivo=Path(caminho), genomas=d["genomas"], fitness=d["fitness"],
                       pista_id=str(d["pista_id"]), pista_nome=str(d["pista_nome"]),
                       comprimento_pista=float(d["comprimento"]), geracao=int(d["geracao"]),
                       melhor_volta=float(d["melhor_volta"]))


def listar_cerebros():
    if not config.PASTA_CEREBROS.exists():
        return []
    return sorted(p for p in config.PASTA_CEREBROS.glob("*.npz") if not p.name.endswith(".treino.npz"))


def cerebros_disponiveis():
    """Todos os cérebros salvos que funcionam com a configuração atual."""
    cerebros = []
    for caminho in listar_cerebros():
        try:
            cerebros.append(carregar_cerebro(caminho))
        except (CerebroIncompativel, OSError, KeyError, ValueError):
            pass
    return cerebros


def cerebro_da_pista(pista):
    """O cérebro treinado nesta pista, ou None se não houver (ou se for incompatível)."""
    caminho = caminho_cerebro(pista.id)
    if not caminho.exists():
        return None
    try:
        return carregar_cerebro(caminho)
    except (CerebroIncompativel, OSError, KeyError, ValueError):
        return None


def recorde_salvo(pista):
    """Fitness do cérebro salvo — um treino novo só o substitui se fizer melhor.
    Se a pista foi editada desde então, o recorde antigo não vale mais."""
    cerebro = cerebro_da_pista(pista)
    if cerebro is None or abs(cerebro.comprimento_pista - pista.comprimento) > 1.0:
        return -math.inf
    return float(cerebro.fitness[0])


def salvar_treino(pista, ag, historico_volta):
    _gravar(caminho_treino(pista.id), historico_volta=np.array(historico_volta, dtype=float),
            comprimento=pista.comprimento, **ag.estado(), **_assinatura())


def geracao_do_treino(pista):
    """Geração em que o treino salvo desta pista parou (None se não houver)."""
    caminho = caminho_treino(pista.id)
    if not caminho.exists():
        return None
    try:
        with np.load(caminho) as d:
            _conferir(d, caminho)
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
            _conferir(d, caminho)
            return {chave: d[chave] for chave in d.files}
    except (CerebroIncompativel, OSError, KeyError, ValueError):
        return None

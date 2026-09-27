"""
Rede neural (perceptron multicamadas) escrita só com NumPy.

Na neuroevolução não há backpropagation: os pesos de cada rede ficam num único
vetor (o "genoma") e quem os ajusta é o algoritmo genético. Como a população
inteira tem a mesma arquitetura, empilhamos os pesos de todas as redes e
avaliamos todas de uma vez.
"""

import numpy as np

from . import config


class Rede:
    def __init__(self, tamanhos):
        self.tamanhos = tuple(int(t) for t in tamanhos)
        self.formas = list(zip(self.tamanhos[:-1], self.tamanhos[1:]))
        self.n_parametros = sum(e * s + s for e, s in self.formas)

    @classmethod
    def padrao(cls):
        """Arquitetura do config: sensores + velocidade → camadas ocultas → direção e acelerador."""
        return cls((len(config.ANGULOS_SENSORES) + 1, *config.CAMADAS_OCULTAS, 2))

    def desempacotar(self, genomas):
        """Genomas (P, G) → lista de camadas (W: (P, entradas, saídas), b: (P, saídas))."""
        genomas = np.atleast_2d(genomas)
        camadas, i = [], 0
        for e, s in self.formas:
            W = genomas[:, i:i + e * s].reshape(-1, e, s)
            i += e * s
            b = genomas[:, i:i + s]
            i += s
            camadas.append((W, b))
        return camadas

    @staticmethod
    def selecionar(camadas, indices):
        """Só as redes de alguns indivíduos (por exemplo, dos carros ainda vivos)."""
        return [(W[indices], b[indices]) for W, b in camadas]

    @staticmethod
    def avaliar(camadas, entradas, com_ativacoes=False):
        """Propaga as entradas (P, n_entradas) pelas P redes. Ativação: tangente hiperbólica."""
        a = entradas
        ativacoes = [a]
        for W, b in camadas:
            a = np.tanh(np.matmul(a[:, None, :], W)[:, 0, :] + b)
            ativacoes.append(a)
        return ativacoes if com_ativacoes else a

    def grupos_por_neuronio(self):
        """Para cada gene, o neurônio que ele alimenta (seus pesos de entrada e o bias).
        O cruzamento troca neurônios inteiros entre os pais, e não pesos soltos."""
        grupos, base = [], 0
        for e, s in self.formas:
            grupos.append(np.tile(np.arange(s), e) + base)   # W[e, s]: o neurônio é a coluna
            grupos.append(np.arange(s) + base)               # b[s]
            base += s
        return np.concatenate(grupos)

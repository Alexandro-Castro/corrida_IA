"""
Algoritmo genético: cria a próxima geração a partir do desempenho (fitness) da atual.

  1. Elitismo: os melhores são copiados sem nenhuma alteração.
  2. Seleção por torneio: sorteia alguns indivíduos e o melhor deles vira pai (ou mãe).
  3. Cruzamento: o filho herda cada neurônio de um dos dois pais.
  4. Mutação: pequenas alterações aleatórias em alguns pesos.
  5. Alguns indivíduos totalmente novos, para manter a diversidade.
"""

import numpy as np

from . import config


class AlgoritmoGenetico:
    def __init__(self, rede, tamanho=config.TAMANHO_POPULACAO, semente=config.SEMENTE):
        self.rede = rede
        self.rng = np.random.default_rng(semente)
        self.grupos = rede.grupos_por_neuronio()
        self.n_grupos = int(self.grupos.max()) + 1
        self.populacao = self.aleatorios(tamanho)
        self.geracao = 1
        self.historico_melhor = []    # fitness do melhor indivíduo de cada geração
        self.historico_media = []

    def aleatorios(self, n):
        return self.rng.normal(0.0, config.ESCALA_INICIAL, (n, self.rede.n_parametros))

    def _torneio(self, fitness, n):
        candidatos = self.rng.integers(0, len(fitness), (n, config.TAMANHO_TORNEIO))
        vencedores = np.argmax(fitness[candidatos], axis=1)
        return candidatos[np.arange(n), vencedores]

    def evoluir(self, fitness):
        """Recebe o fitness de cada indivíduo e substitui a população pela próxima geração.
        Os elites ficam nas primeiras posições da nova população."""
        fitness = np.asarray(fitness, dtype=float)
        tamanho = len(self.populacao)
        ordem = np.argsort(-fitness, kind="stable")
        self.historico_melhor.append(float(fitness[ordem[0]]))
        self.historico_media.append(float(fitness.mean()))

        n_elite = min(config.ELITE, tamanho)
        n_novos = min(int(round(tamanho * config.FRACAO_NOVOS)), tamanho - n_elite)
        n_filhos = tamanho - n_elite - n_novos

        pais = self._torneio(fitness, n_filhos)
        maes = self._torneio(fitness, n_filhos)
        filhos = self.populacao[pais].copy()
        cruza = self.rng.random(n_filhos) < config.PROB_CRUZAMENTO
        da_mae = (self.rng.random((n_filhos, self.n_grupos)) < 0.5)[:, self.grupos] & cruza[:, None]
        filhos[da_mae] = self.populacao[maes][da_mae]
        muta = self.rng.random(filhos.shape) < config.TAXA_MUTACAO
        filhos += muta * self.rng.normal(0.0, config.FORCA_MUTACAO, filhos.shape)

        self.populacao = np.vstack([self.populacao[ordem[:n_elite]], filhos, self.aleatorios(n_novos)])
        self.geracao += 1
        return ordem

    def estado(self):
        return {
            "populacao": self.populacao,
            "geracao": self.geracao,
            "historico_melhor": np.array(self.historico_melhor),
            "historico_media": np.array(self.historico_media),
        }

    def restaurar(self, estado):
        self.populacao = np.array(estado["populacao"], dtype=float)
        self.geracao = int(estado["geracao"])
        self.historico_melhor = [float(v) for v in estado["historico_melhor"]]
        self.historico_media = [float(v) for v in estado["historico_media"]]

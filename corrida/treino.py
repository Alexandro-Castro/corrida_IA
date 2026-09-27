"""Treino por neuroevolução: simula gerações de carros e evolui as redes neurais."""

import math
import time

import numpy as np

from . import arquivos, config
from .carros import Frota
from .evolucao import AlgoritmoGenetico
from .formato import numero, tempo
from .rede_neural import Rede


class Treinador:
    """Cada indivíduo da população é um carro com sua própria rede neural.
    Uma geração termina quando todos os carros são eliminados (bateram ou pararam
    de avançar) ou quando o tempo acaba. O fitness é a distância percorrida."""

    def __init__(self, pista, continuar=False, salvar=True):
        self.pista = pista.preparar()
        self.rede = Rede.padrao()
        self.salvar = salvar
        self.ag = AlgoritmoGenetico(self.rede)
        self.historico_volta = []     # melhor tempo de volta de cada geração (nan = ninguém completou)
        if continuar:
            estado = arquivos.carregar_treino(pista)
            if estado is not None:
                self.ag.restaurar(estado)
                self.historico_volta = [float(v) for v in estado["historico_volta"]]
        # o cérebro salvo só é substituído por um melhor do que ele
        self.recorde = arquivos.recorde_salvo(pista) if salvar else -math.inf
        self.resumo = None            # dados da última geração concluída
        self._nova_geracao()

    @property
    def geracao(self):
        return self.ag.geracao

    def _nova_geracao(self):
        self.frota = Frota(self.pista, len(self.ag.populacao))
        self.camadas = self.rede.desempacotar(self.ag.populacao)
        self.n_elite = min(config.ELITE, len(self.ag.populacao)) if self.ag.geracao > 1 else 0

    def passo(self):
        """Avança a simulação um passo. Devolve True quando uma geração acabou
        (a próxima já estará pronta)."""
        f = self.frota
        vivos = np.flatnonzero(f.vivo)
        if vivos.size == 0 or f.tempo >= config.TEMPO_MAX_GERACAO:
            self._encerrar_geracao()
            return True
        saidas = self.rede.avaliar(self.rede.selecionar(self.camadas, vivos), f.entradas(vivos))
        f.passo(saidas[:, 0], saidas[:, 1], vivos)
        return False

    def encerrar_agora(self):
        self._encerrar_geracao()

    def _encerrar_geracao(self):
        f = self.frota
        fitness = f.melhor_progresso.copy()
        ordem = np.argsort(-fitness, kind="stable")
        melhor = float(fitness[ordem[0]])
        L = self.pista.comprimento
        novo_recorde = melhor > self.recorde + 1e-6
        if novo_recorde:
            self.recorde = melhor
            if self.salvar:
                top = ordem[:config.ELITE]
                arquivos.salvar_cerebro(self.pista, self.ag.populacao[top], fitness[top],
                                        self.ag.geracao, f.melhor_volta(int(ordem[0])))
        self.resumo = {
            "geracao": self.ag.geracao,
            "melhor": melhor / L,
            "media": float(fitness.mean()) / L,
            "completaram": int((f.voltas >= 1).sum()),
            "melhor_volta": f.melhor_volta(),
            "novo_recorde": novo_recorde,
        }
        self.historico_volta.append(self.resumo["melhor_volta"])
        self.ag.evoluir(fitness)
        if self.salvar:
            arquivos.salvar_treino(self.pista, self.ag, self.historico_volta)
        self._nova_geracao()


def treinar_sem_janela(pista, geracoes, continuar=False):
    """Treina sem desenhar nada (bem mais rápido) e mostra o progresso no terminal."""
    treinador = Treinador(pista, continuar=continuar)
    print(f"Treinando na pista '{pista.nome}' ({pista.comprimento:.0f} px, "
          f"{len(treinador.ag.populacao)} carros por geração). Ctrl+C para parar.")
    try:
        for _ in range(geracoes):
            inicio = time.perf_counter()
            while not treinador.passo():
                pass
            r = treinador.resumo
            print(f"Geração {r['geracao']:4d} | melhor {numero(r['melhor']):>5} voltas | média {numero(r['media']):>5}"
                  f" | completaram 1 volta: {r['completaram']:3d} | melhor volta {tempo(r['melhor_volta']):>8}"
                  f" | {numero(time.perf_counter() - inicio, 1):>4} s{'  << recorde salvo' if r['novo_recorde'] else ''}")
    except KeyboardInterrupt:
        print("\nTreino interrompido. O progresso de cada geração concluída já está salvo.")
    return treinador

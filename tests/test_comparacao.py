"""Testes da comparação entre técnicas (sem treinar nada: dados sintéticos)."""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from corrida import arquivos, comparacao, config
from corrida.avaliacao import avaliar, avaliar_todos
from corrida.piloto import Piloto
from corrida.rede_neural import Rede
from test_nucleo import pista_oval


class TestComparacao(unittest.TestCase):
    def setUp(self):
        self.pasta = tempfile.TemporaryDirectory()
        self.originais = config.PASTA_CEREBROS, comparacao.PASTA_RESULTADOS
        config.PASTA_CEREBROS = Path(self.pasta.name) / "cerebros"
        comparacao.PASTA_RESULTADOS = Path(self.pasta.name) / "resultados"

    def tearDown(self):
        config.PASTA_CEREBROS, comparacao.PASTA_RESULTADOS = self.originais
        self.pasta.cleanup()

    def test_avaliar_varios_de_uma_vez_e_igual_a_um_por_vez(self):
        pista = pista_oval()
        agires = [lambda e: np.array([[0.0, 1.0]]), lambda e: np.array([[0.05, 0.6]]),
                  lambda e: np.array([[0.0, 0.0]])]
        juntos = avaliar_todos(pista, agires)
        for agir, r in zip(agires, juntos):
            sozinho = avaliar(pista, agir)
            self.assertAlmostEqual(r.voltas, sozinho.voltas, places=9)
            self.assertEqual((r.bateu, r.tempo), (sozinho.bateu, sozinho.tempo))
        self.assertTrue(juntos[2].bateu)          # parado: eliminado por não avançar

    def test_matriz_e_tabela(self):
        pista = pista_oval()
        pista.arquivo = Path("oval_teste.json")
        pista.salvar(Path(self.pasta.name) / "oval_teste.json")
        rede = Rede.padrao()
        arquivos.salvar_cerebro(pista, np.zeros((1, rede.n_parametros)), np.ones(1), 1, math.nan, rede.tamanhos)
        cerebros, pistas, resultados = comparacao.matriz_generalizacao(
            arquivos_pista=[Path(self.pasta.name) / "oval_teste.json"])
        self.assertEqual((len(cerebros), len(pistas), len(resultados[0])), (1, 1, 1))
        self.assertTrue(comparacao.treinou_em(cerebros[0], pistas[0]))
        tabela = comparacao.tabela_generalizacao(cerebros, pistas, resultados)
        self.assertIn("Neuroevolução · Teste", tabela)
        self.assertIn("**", tabela)               # a pista de treino aparece em negrito
        relatorio = comparacao.escrever_relatorio(cerebros, pistas, resultados)
        self.assertTrue(relatorio.exists())

    def test_curvas_marcos_e_ida_e_volta(self):
        dados = {"pista_id": "x", "pista_nome": "X", "horas": 2, "dispositivo": "cpu", "rede_ppo": [10, 8, 2],
                 "data": "", "neuroevolucao": {"experiencia": [0.5, 1.0, 1.5], "relogio": [1, 2, 3],
                                               "voltas": [0.2, 1.4, 2.0], "melhor_volta": [math.nan, 20.0, 18.0]},
                 "ppo": {"experiencia": [0.7, 1.4], "relogio": [5, 10], "voltas": [0.3, 0.9],
                         "melhor_volta": [math.nan, math.nan]}}
        m = comparacao.marcos(dados["neuroevolucao"])
        self.assertEqual((m["primeira_volta_h"], m["primeira_volta_s"], m["voltas_final"], m["melhor_volta"]),
                         (1.0, 2, 2.0, 18.0))
        self.assertTrue(math.isnan(comparacao.marcos(dados["ppo"])["primeira_volta_h"]))
        comparacao.salvar_curvas(dados)
        lido = comparacao.carregar_curvas("x")
        self.assertTrue(math.isnan(lido["ppo"]["melhor_volta"][0]))
        self.assertIn("não chegou", comparacao.tabela_curvas(lido))

    def test_cerebro_de_varias_pistas_conta_todas_como_vistas(self):
        pista = pista_oval()
        pista.nome = "Oval"
        rede = Rede.padrao()
        destino = type("D", (), {"id": "varias", "nome": "Várias (Oval, Clássica)", "comprimento": 1.0})()
        arquivos.salvar_cerebro(destino, np.zeros((1, rede.n_parametros)), np.ones(1), 1, math.nan, rede.tamanhos,
                                metodo="ppo")
        cerebro = arquivos.cerebros_disponiveis()[0]
        self.assertTrue(comparacao.treinou_em(cerebro, pista))
        self.assertEqual(Piloto(cerebro).nome, "PPO")


if __name__ == "__main__":
    unittest.main()

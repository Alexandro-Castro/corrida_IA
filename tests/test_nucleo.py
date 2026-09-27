"""Testes da parte sem janela: pista, carros, rede neural, algoritmo genético e arquivos.

    python -m unittest discover tests
"""

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from corrida import arquivos, config
from corrida.carros import Frota
from corrida.evolucao import AlgoritmoGenetico
from corrida.pista import Pista, nome_para_arquivo, spline_fechada
from corrida.rede_neural import Rede


def pista_oval(largura=100):
    """Estádio: retas em y=200 e y=800, curvas de raio 300 centradas em x=500 e x=1100."""
    pontos = [(500, 200), (800, 200), (1100, 200), (1250, 240), (1360, 350), (1400, 500), (1360, 650),
              (1250, 760), (1100, 800), (800, 800), (500, 800), (350, 760), (240, 650), (200, 500),
              (240, 350), (350, 240)]
    return Pista(pontos, largura, "Teste").preparar()


class TestPista(unittest.TestCase):
    def test_spline_passa_pelos_pontos(self):
        pontos = np.array([(0, 0), (100, 0), (100, 100), (0, 100)], dtype=float)
        curva = spline_fechada(pontos, amostras=10)
        np.testing.assert_allclose(curva[::10], pontos, atol=1e-9)

    def test_linha_central_uniforme_e_fechada(self):
        p = pista_oval()
        self.assertLess(np.ptp(p.comp_seg), 0.5)
        self.assertAlmostEqual(p.comprimento, p.comp_seg.sum())
        self.assertLess(math.dist(p.centro[-1], p.centro[0]), 10)

    def test_mascara_de_asfalto(self):
        p = pista_oval()
        self.assertTrue(p.livre(np.array([800.0]), np.array([200.0]))[0])       # meio da reta
        self.assertTrue(p.livre(np.array([800.0]), np.array([245.0]))[0])       # perto da borda
        self.assertFalse(p.livre(np.array([800.0]), np.array([260.0]))[0])      # grama
        self.assertFalse(p.livre(np.array([800.0]), np.array([500.0]))[0])      # centro do oval
        self.assertFalse(p.livre(np.array([-5.0]), np.array([200.0]))[0])       # fora do mundo

    def test_projetar_e_ponto_em_sao_inversos(self):
        p = pista_oval()
        s = np.linspace(0, p.comprimento, 50, endpoint=False)
        x, y, _ = p.ponto_em(s)
        np.testing.assert_allclose(p.projetar(x, y), s, atol=1.0)

    def test_sensores_numa_reta(self):
        p = pista_oval(largura=100)
        # carro no meio da reta de cima, apontando para a direita (+x)
        angulos = np.radians([[-90.0, 0.0, 90.0]])
        d = p.medir(np.array([800.0]), np.array([200.0]), angulos)[0]
        self.assertAlmostEqual(d[0], 50, delta=1.5)      # parede de cima
        self.assertAlmostEqual(d[2], 50, delta=1.5)      # parede de baixo
        self.assertEqual(d[1], config.ALCANCE_SENSOR)    # nada à frente dentro do alcance

    def test_deteccao_de_trechos_sobrepostos(self):
        p = pista_oval(largura=100)
        self.assertFalse(p.trechos_sobrepostos().any())
        # duas retas paralelas a 100 px: com 60 de largura sobra grama entre elas; com 95, não
        retas = [(200, 300), (600, 300), (1000, 300), (1400, 300), (1400, 400), (1000, 400), (600, 400), (200, 400)]
        self.assertFalse(Pista(retas, 60).trechos_sobrepostos().any())
        self.assertTrue(Pista(retas, 95).trechos_sobrepostos().any())

    def test_salvar_e_carregar(self):
        p = Pista([(100, 100), (500, 120), (300, 400)], 90, "Pista Ação")
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "x.json"
            p.salvar(caminho)
            q = Pista.carregar(caminho)
        self.assertEqual(q.nome, "Pista Ação")
        self.assertEqual(q.largura, 90)
        np.testing.assert_allclose(q.pontos, p.pontos)

    def test_nome_para_arquivo(self):
        self.assertEqual(nome_para_arquivo("Minha Pista Ação 2!"), "minha_pista_acao_2")
        self.assertEqual(nome_para_arquivo("???"), "pista")


class TestCarros(unittest.TestCase):
    def test_acelerar_em_linha_reta_faz_progresso(self):
        p = pista_oval()
        f = Frota(p, 1)
        for _ in range(60):
            f.passo(np.zeros(1), np.ones(1))
        self.assertTrue(f.vivo[0])
        self.assertGreater(f.velocidade[0], 100)
        self.assertAlmostEqual(f.progresso[0], f.x[0] - 500, delta=2)

    def test_bater_na_parede_elimina_no_treino(self):
        f = Frota(pista_oval(), 1)
        for _ in range(300):
            f.passo(np.ones(1), np.ones(1))      # vira sempre para a direita
        self.assertFalse(f.vivo[0])

    def test_bater_na_corrida_reposiciona(self):
        f = Frota(pista_oval(), 1, modo="corrida")
        for _ in range(300):
            f.passo(np.ones(1), np.ones(1))
        self.assertTrue(f.vivo[0])
        self.assertGreater(f.batidas[0], 0)
        self.assertTrue(f.pista.livre(f.x, f.y)[0])

    def test_parado_e_eliminado(self):
        f = Frota(pista_oval(), 1)
        for _ in range(int((config.TEMPO_SEM_PROGRESSO + 0.5) / config.DT)):
            f.passo(np.zeros(1), np.zeros(1))
        self.assertFalse(f.vivo[0])

    def test_volta_completa_e_contada(self):
        p = pista_oval()
        f = Frota(p, 1)
        # "teletransporta" o carro ao longo da linha central e confere a contagem de voltas
        for s in np.arange(10, p.comprimento * 1.2, 10):
            x, y, a = p.ponto_em(s)
            f.x[:], f.y[:], f.angulo[:] = x, y, a
            f.tempo += config.DT
            f._medir_progresso(np.array([0]))
        self.assertEqual(f.voltas[0], 1)
        self.assertEqual(len(f.tempos_volta[0]), 1)
        self.assertGreater(f.progresso[0], p.comprimento)


class TestRedeEGenetica(unittest.TestCase):
    def test_formas_da_rede(self):
        rede = Rede((10, 12, 2))
        self.assertEqual(rede.n_parametros, 10 * 12 + 12 + 12 * 2 + 2)
        genomas = np.random.default_rng(0).normal(size=(7, rede.n_parametros))
        saida = rede.avaliar(rede.desempacotar(genomas), np.zeros((7, 10)))
        self.assertEqual(saida.shape, (7, 2))
        self.assertTrue(np.all(np.abs(saida) <= 1))

    def test_avaliacao_em_lote_igual_a_individual(self):
        rede = Rede((4, 5, 2))
        rng = np.random.default_rng(1)
        genomas = rng.normal(size=(3, rede.n_parametros))
        entradas = rng.random((3, 4))
        lote = rede.avaliar(rede.desempacotar(genomas), entradas)
        for i in range(3):
            sozinho = rede.avaliar(rede.desempacotar(genomas[i]), entradas[i:i + 1])
            np.testing.assert_allclose(lote[i], sozinho[0])

    def test_grupos_por_neuronio(self):
        rede = Rede((3, 2, 1))
        grupos = rede.grupos_por_neuronio()
        self.assertEqual(len(grupos), rede.n_parametros)
        self.assertEqual(sorted(set(grupos.tolist())), [0, 1, 2])

    def test_elites_sobrevivem_intactos(self):
        rede = Rede((4, 3, 2))
        ag = AlgoritmoGenetico(rede, tamanho=20, semente=0)
        fitness = np.arange(20.0)
        melhores = ag.populacao[np.argsort(-fitness)[:config.ELITE]].copy()
        ag.evoluir(fitness)
        self.assertEqual(ag.populacao.shape, (20, rede.n_parametros))
        np.testing.assert_array_equal(ag.populacao[:config.ELITE], melhores)
        self.assertEqual(ag.geracao, 2)
        self.assertEqual(ag.historico_melhor, [19.0])


class TestArquivos(unittest.TestCase):
    def setUp(self):
        self.pasta = tempfile.TemporaryDirectory()
        self.original = config.PASTA_CEREBROS
        config.PASTA_CEREBROS = Path(self.pasta.name)

    def tearDown(self):
        config.PASTA_CEREBROS = self.original
        self.pasta.cleanup()

    def test_cerebro_ida_e_volta(self):
        p = pista_oval()
        p.arquivo = Path("oval_teste.json")
        rede = Rede.padrao()
        genomas = np.random.default_rng(0).normal(size=(5, rede.n_parametros))
        arquivos.salvar_cerebro(p, genomas, np.arange(5.0)[::-1] * 1000, 7, 12.5)
        c = arquivos.cerebro_da_pista(p)
        np.testing.assert_array_equal(c.genomas, genomas)
        self.assertEqual((c.pista_id, c.geracao, c.melhor_volta), ("oval_teste", 7, 12.5))
        self.assertEqual(arquivos.recorde_salvo(p), 4000.0)
        self.assertEqual([x.pista_id for x in arquivos.cerebros_disponiveis()], ["oval_teste"])

    def test_treino_ida_e_volta(self):
        p = pista_oval()
        p.arquivo = Path("oval_teste.json")
        ag = AlgoritmoGenetico(Rede.padrao(), tamanho=10, semente=0)
        ag.evoluir(np.arange(10.0))
        arquivos.salvar_treino(p, ag, [float("nan")])
        outro = AlgoritmoGenetico(Rede.padrao(), tamanho=10, semente=1)
        outro.restaurar(arquivos.carregar_treino(p))
        np.testing.assert_array_equal(outro.populacao, ag.populacao)
        self.assertEqual(outro.geracao, 2)
        self.assertEqual(arquivos.geracao_do_treino(p), 2)


if __name__ == "__main__":
    unittest.main()

"""Testes do Aprendizado por Reforço: ambiente, avaliação e PPO (este só se o PyTorch estiver instalado)."""

import importlib.util
import unittest

import numpy as np

from corrida import config
from corrida.avaliacao import avaliar
from corrida.rl.ambiente import PENALIDADE_BATIDA, AmbienteCorrida
from test_nucleo import pista_oval

TEM_TORCH = importlib.util.find_spec("torch") is not None


class TestAmbiente(unittest.TestCase):
    def test_observacoes_e_largada_aleatoria(self):
        amb = AmbienteCorrida([pista_oval()], carros_por_pista=32, semente=0)
        obs = amb.observar()
        self.assertEqual(obs.shape, (32, len(config.ANGULOS_SENSORES) + 1))
        self.assertEqual(obs.dtype, np.float32)
        frota = amb.frotas[0]
        self.assertTrue(frota.pista.livre(frota.x, frota.y).all())
        self.assertGreater(np.ptp(frota.s), 1000)          # espalhados pela pista

    def test_avancar_da_recompensa_positiva(self):
        amb = AmbienteCorrida([pista_oval()], carros_por_pista=2, largada_aleatoria=False)
        total = 0.0
        for _ in range(60):
            _, r, terminou, cortado, _ = amb.passo(np.array([[0.0, 1.0], [0.0, 1.0]]))
            self.assertFalse(terminou.any() or cortado.any())
            total += r[0]
        self.assertGreater(total, 0)
        esperado = amb.frotas[0].progresso[0] / (config.VELOCIDADE_MAX * config.DT) * 0.1
        self.assertAlmostEqual(total, esperado, places=6)

    def test_bater_termina_e_recomeca(self):
        amb = AmbienteCorrida([pista_oval()], carros_por_pista=1, largada_aleatoria=False)
        for _ in range(300):
            _, r, terminou, _, _ = amb.passo(np.array([[1.0, 1.0]]))     # sempre para a direita
            if terminou[0]:
                break
        self.assertTrue(terminou[0])
        self.assertLessEqual(r[0], -PENALIDADE_BATIDA + 1e-9)        # no passo da batida não há avanço
        self.assertTrue(amb.frotas[0].vivo[0])
        self.assertEqual(amb.passos[0], 0)
        self.assertEqual(len(amb.episodios), 1)

    def test_episodio_cortado_pelo_tempo(self):
        amb = AmbienteCorrida([pista_oval()], carros_por_pista=1, largada_aleatoria=False, tempo_max=0.5)
        cortes = 0
        for _ in range(31):
            _, _, terminou, cortado, _ = amb.passo(np.array([[0.0, 0.5]]))
            self.assertFalse(terminou[0])
            cortes += int(cortado[0])
        self.assertEqual(cortes, 1)

    def test_varias_pistas(self):
        amb = AmbienteCorrida([pista_oval(), pista_oval(120)], carros_por_pista=8, semente=1)
        obs, r, *_ = amb.passo(np.zeros((16, 2)))
        self.assertEqual(obs.shape[0], 16)
        self.assertEqual(r.shape, (16,))


@unittest.skipUnless(TEM_TORCH, "PyTorch não instalado (pip install -r requirements-rl.txt)")
class TestPPO(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path
        self.pasta = tempfile.TemporaryDirectory()
        self.original = config.PASTA_CEREBROS
        config.PASTA_CEREBROS = Path(self.pasta.name)

    def tearDown(self):
        config.PASTA_CEREBROS = self.original
        self.pasta.cleanup()

    def pequeno(self):
        from corrida.rl.ppo import ParametrosPPO
        return ParametrosPPO(camadas=(16, 16), carros=8, passos_por_coleta=32, epocas=2, minilotes=2,
                             avaliar_a_cada=2)

    def test_cerebro_exportado_dirige_igual_ao_ator(self):
        import torch
        from corrida.rede_neural import Rede
        from corrida.rl.ppo import AtorCritico
        modelo = AtorCritico(10, 2, (32, 16), -0.5)
        for parametro in modelo.parameters():          # pesos aleatórios "de verdade", não os iniciais
            torch.nn.init.normal_(parametro, 0, 0.5)
        obs = np.random.default_rng(0).random((20, 10)).astype(np.float32)
        with torch.no_grad():
            esperado = modelo.distribuicao(torch.as_tensor(obs)).mean.numpy()
        rede = Rede(modelo.tamanhos)
        np.testing.assert_allclose(rede.avaliar(rede.desempacotar(modelo.genoma()), obs), esperado, atol=1e-5)

    def test_treina_avalia_salva_e_continua(self):
        from pathlib import Path
        from corrida import arquivos
        from corrida.rl.ppo import TreinadorPPO
        pista = pista_oval()
        pista.arquivo = Path("oval_teste.json")
        t = TreinadorPPO([pista], self.pequeno(), "cpu", semente=0)
        self.assertIsNone(t.iteracao())
        registro = t.iteracao()                    # 2ª atualização: avaliação justa
        self.assertEqual(t.passos, 2 * 32 * 8)
        self.assertTrue(registro["novo_recorde"])
        cerebro = arquivos.cerebro_da_pista(pista, "ppo")
        self.assertEqual((cerebro.metodo, cerebro.tamanhos), ("ppo", (10, 16, 16, 2)))
        self.assertAlmostEqual(cerebro.experiencia, t.experiencia)
        outro = TreinadorPPO([pista], None, "cpu", continuar=True)
        self.assertEqual((outro.passos, outro.atualizacoes, outro.p.camadas), (t.passos, 2, (16, 16)))
        self.assertEqual(len(outro.historico), 1)

    def test_varias_pistas_salva_como_varias(self):
        from pathlib import Path
        from corrida import arquivos
        from corrida.rl.ppo import TreinadorPPO
        a, b = pista_oval(), pista_oval(120)
        a.arquivo, b.arquivo = Path("a.json"), Path("b.json")
        t = TreinadorPPO([a, b], self.pequeno(), "cpu", semente=0)
        t.iteracao()
        t.iteracao()
        nomes = [c.arquivo.name for c in arquivos.cerebros_disponiveis()]
        self.assertEqual(nomes, ["varias.ppo.npz"])


class TestAvaliacao(unittest.TestCase):
    def test_acelerar_reto_bate_na_primeira_curva(self):
        r = avaliar(pista_oval(), lambda e: np.array([[0.0, 1.0]]))
        self.assertTrue(r.bateu)
        self.assertGreater(r.voltas, 0.1)
        self.assertLess(r.voltas, 0.5)


if __name__ == "__main__":
    unittest.main()

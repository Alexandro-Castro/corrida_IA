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

    def test_bater_termina_penaliza_e_recomeca(self):
        amb = AmbienteCorrida([pista_oval()], carros_por_pista=1, largada_aleatoria=False)
        for _ in range(300):
            _, r, terminou, _, _ = amb.passo(np.array([[1.0, 1.0]]))     # sempre para a direita
            if terminou[0]:
                break
        self.assertTrue(terminou[0])
        self.assertLess(r[0], -PENALIDADE_BATIDA + 0.2)
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


class TestAvaliacao(unittest.TestCase):
    def test_acelerar_reto_bate_na_primeira_curva(self):
        r = avaliar(pista_oval(), lambda e: np.array([[0.0, 1.0]]))
        self.assertTrue(r.bateu)
        self.assertGreater(r.voltas, 0.1)
        self.assertLess(r.voltas, 0.5)


if __name__ == "__main__":
    unittest.main()

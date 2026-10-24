"""Comparar técnicas: cada cérebro salvo em cada pista, e as curvas de aprendizado medidas."""

import math
import threading

import pygame

from .. import comparacao, desenho
from ..app import Tela
from ..formato import numero, tempo
from ..interface import (DESTAQUE, FUNDO, LINHA, PAINEL, PAINEL_CLARO, TEXTO, TEXTO_FRACO, VERDE, VERMELHO, Painel,
                         dividir_tela, misturar, texto)

CORES_METODO = {"neuroevolucao": VERDE, "ppo": DESTAQUE}


class TelaComparacao(Tela):
    def __init__(self, app, pista_id=None):
        super().__init__(app)
        self.pista_id = pista_id
        self.i_curva = 0
        self._calcular()

    def _calcular(self):
        self.matriz = None
        self.progresso = (0, 1)
        self.curvas = []

        def trabalho():
            cerebros, pistas, resultados = comparacao.matriz_generalizacao(
                ao_avancar=lambda k, n: setattr(self, "progresso", (k, n)))
            self.curvas = [c for c in (comparacao.carregar_curvas(p.id) for p in pistas) if c is not None]
            ids = [c["pista_id"] for c in self.curvas]
            self.i_curva = ids.index(self.pista_id) if self.pista_id in ids else 0
            self.matriz = (cerebros, pistas, resultados)

        threading.Thread(target=trabalho, daemon=True).start()

    def ao_redimensionar(self):
        self.area, self.rect_painel = dividir_tela(self.app.janela.get_size(), self.app.escala_ui, 420)

    def evento(self, ev):
        if ev.type != pygame.KEYDOWN:
            return
        if ev.key == pygame.K_ESCAPE:
            from .menu import TelaMenu
            self.app.ir_para(TelaMenu(self.app, self.pista_id))
        elif ev.key == pygame.K_r and self.matriz is not None:
            self._calcular()
        elif ev.key in (pygame.K_LEFT, pygame.K_RIGHT) and self.curvas:
            self.i_curva = (self.i_curva + (1 if ev.key == pygame.K_RIGHT else -1)) % len(self.curvas)

    # ------------------------------------------------------------------ desenho
    def desenhar(self, sup):
        e = self.app.escala_ui
        f = self.app.fontes
        sup.fill(FUNDO)
        m = round(28 * e)
        x, y = self.area.x + m, self.area.y + m
        r = texto(sup, "Neuroevolução × PPO", f.titulo, DESTAQUE, (x, y))
        r = texto(sup, "Voltas em 45 s de cada cérebro em cada pista — a mesma avaliação justa para todos: "
                       "larga parado e dirige até bater ou acabar o tempo.", f.normal, TEXTO_FRACO, (x, r.bottom + 4 * e))
        topo = r.bottom + round(18 * e)
        if self.matriz is None:
            k, n = self.progresso
            texto(sup, f"Avaliando os cérebros... pista {k} de {n}", f.subtitulo, TEXTO, (x, topo))
        elif not self.matriz[0]:
            texto(sup, "Ainda não há cérebros treinados para comparar.", f.subtitulo, TEXTO, (x, topo))
        else:
            self._desenhar_matriz(sup, pygame.Rect(x, topo, self.area.w - 2 * m, self.area.bottom - topo - m))
        self._desenhar_painel(sup)

    def _desenhar_matriz(self, sup, area):
        e = self.app.escala_ui
        f = self.app.fontes
        cerebros, pistas, resultados = self.matriz
        largura_nome = max(f.negrito.size(c.descricao)[0] for c in cerebros) + round(20 * e)
        colunas = len(pistas) + 2
        largura = max(round(64 * e), (area.w - largura_nome) // colunas)
        altura = min(round(44 * e), max(round(26 * e), (area.h - round(110 * e)) // (len(cerebros) + 1)))
        y = area.y
        nomes = [p.nome for p in pistas] + ["vistas", "inéditas"]
        for k, nome in enumerate(nomes):
            cx = area.x + largura_nome + k * largura + largura / 2
            cor = TEXTO_FRACO if k >= len(pistas) else TEXTO
            texto(sup, nome if f.pequena.size(nome)[0] < largura - 6 else nome[:7] + "…", f.pequena, cor,
                  (cx, y + altura / 2), "center")
        texto(sup, "média nas pistas", f.pequena, TEXTO_FRACO,
              (area.x + largura_nome + (len(pistas) + 1) * largura, y - 2 * e), "midbottom")
        for j, cerebro in enumerate(cerebros):
            y += altura
            texto(sup, cerebro.descricao, f.negrito, CORES_METODO.get(cerebro.metodo, TEXTO), (area.x, y + altura / 2),
                  "midleft")
            vistas, ineditas = [], []
            for i, pista in enumerate(pistas):
                res = resultados[i][j]
                melhor = max(r.voltas for r in resultados[i]) or 1.0
                casa = comparacao.treinou_em(cerebro, pista)
                (vistas if casa else ineditas).append(res.voltas)
                celula = pygame.Rect(area.x + largura_nome + i * largura + 2, y + 2, largura - 4, altura - 4)
                if res.bateu:
                    fundo = misturar(PAINEL, VERMELHO, 0.25 + 0.2 * min(res.voltas / melhor, 1))
                else:
                    fundo = misturar(PAINEL, VERDE, 0.2 + 0.5 * res.voltas / melhor)
                pygame.draw.rect(sup, fundo, celula, border_radius=round(5 * e))
                if casa:
                    pygame.draw.rect(sup, TEXTO, celula, max(1, round(2 * e)), border_radius=round(5 * e))
                texto(sup, numero(res.voltas) + (" x" if res.bateu else ""), f.negrito if casa else f.normal, TEXTO,
                      celula.center, "center")
            for k, valores in enumerate((vistas, ineditas)):
                celula = pygame.Rect(area.x + largura_nome + (len(pistas) + k) * largura + 2, y + 2, largura - 4, altura - 4)
                pygame.draw.rect(sup, PAINEL_CLARO, celula, border_radius=round(5 * e))
                media = numero(sum(valores) / len(valores)) if valores else "—"
                texto(sup, media, f.negrito, TEXTO, celula.center, "center")
        y += altura + round(16 * e)
        for linha in ("Borda branca: pista em que o cérebro treinou.   x: bateu antes de acabar o tempo.",
                      "Cor: verde mais forte = mais perto do melhor resultado naquela pista; vermelho = bateu.",
                      "\"Inéditas\" mede a generalização: dirigir bem numa pista que a rede nunca viu."):
            r = texto(sup, linha, f.normal, TEXTO_FRACO, (area.x, y))
            y = r.bottom + round(2 * e)

    def _desenhar_painel(self, sup):
        e = self.app.escala_ui
        fontes = self.app.fontes
        p = Painel(sup, self.rect_painel, fontes)
        fim = p.rodape([("← →", "curvas de outra pista"), ("R", "avaliar de novo"), ("Esc", "voltar ao menu")])
        p.titulo("Aprendizado", "do zero, com o mesmo orçamento")
        if not self.curvas:
            p.paragrafo("Nenhuma curva medida ainda. No terminal, rode:", TEXTO)
            p.texto("python main.py comparar --curvas --pista classica", DESTAQUE, fontes.mono)
            p.espaco(6)
            p.paragrafo("As duas técnicas treinam do zero com as mesmas horas de direção simulada (sem mexer nos "
                        "seus cérebros) e o resultado aparece aqui e em resultados/comparacao.md.")
            return
        dados = self.curvas[self.i_curva]
        p.linha("Pista", dados["pista_nome"])
        p.linha("Orçamento", f"{numero(dados['horas'], 0)} h de direção cada")
        ne, ppo = dados["neuroevolucao"], dados["ppo"]
        altura = max(90 * e, (fim - p.y - 250 * e) / 2)
        for eixo, titulo, rotulo, escala in (("experiencia", "Voltas × experiência", "h de direção", 1),
                                             ("relogio", "Voltas × tempo de relógio", "min de relógio", 60)):
            p.secao(titulo)
            desenho.desenhar_curvas(sup, p.area(altura), fontes, [
                ([x / escala for x in ne[eixo]], ne["voltas"], VERDE, "neuroevolução"),
                ([x / escala for x in ppo[eixo]], ppo["voltas"], DESTAQUE, "PPO"),
            ], rotulo, "voltas em 45 s")
        m_ne, m_ppo = comparacao.marcos(ne), comparacao.marcos(ppo)
        p.secao("1ª volta completa")
        fmt = lambda v, f: "não chegou" if math.isnan(v) else f(v)
        p.linha("Neuroevolução", fmt(m_ne["primeira_volta_h"], lambda v: f"{numero(v, 1)} h · "
                                                                   f"{numero(m_ne['primeira_volta_s'], 0)} s"), VERDE)
        p.linha("PPO", fmt(m_ppo["primeira_volta_h"], lambda v: f"{numero(v, 1)} h · "
                                                            f"{numero(m_ppo['primeira_volta_s'], 0)} s"), DESTAQUE)
        p.secao("Melhor volta")
        p.linha("Neuroevolução", tempo(m_ne["melhor_volta"]), VERDE)
        p.linha("PPO", tempo(m_ppo["melhor_volta"]), DESTAQUE)

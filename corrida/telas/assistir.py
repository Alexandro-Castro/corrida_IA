"""Assistir: o melhor cérebro salvo dirigindo sozinho — em qualquer pista, até nas que ele nunca viu."""

import numpy as np
import pygame

from .. import arquivos, desenho
from ..app import Tela
from ..carros import Frota
from ..interface import (DESTAQUE, FUNDO, LARANJA, TEXTO, VERDE, Aviso, Painel, Vista, dividir_tela, kmh, numero,
                         tempo, texto)
from ..pista import Pista
from ..rede_neural import Rede

VELOCIDADES = (1, 2, 4, 8)


class TelaAssistir(Tela):
    def __init__(self, app, pista, ao_voltar=None):
        super().__init__(app)
        self.ao_voltar = ao_voltar
        self.rede = Rede.padrao()
        self.cerebros = arquivos.cerebros_disponiveis()
        ids = [c.pista_id for c in self.cerebros]
        self.arquivos_pista = Pista.listar()
        caminhos = [c.resolve() for c in self.arquivos_pista]
        atual = pista.arquivo.resolve() if pista.arquivo else None
        self.i_pista = caminhos.index(atual) if atual in caminhos else -1
        self.i_vel = 0
        self.pausado = False
        self.sensores = True
        self.aviso = Aviso()
        self.vista = None
        self._usar_pista(pista)
        self._usar_cerebro(ids.index(pista.id) if pista.id in ids else 0)

    def _usar_pista(self, pista):
        self.pista = pista.preparar()
        self.imagem = desenho.imagem_pista(pista).convert()
        pista.liberar_campos()
        if self.vista is not None:
            self.fundo = self.vista.ajustar(self.imagem)
        self._reiniciar()

    def _usar_cerebro(self, indice):
        self.i_cerebro = indice % len(self.cerebros)
        self.cerebro = self.cerebros[self.i_cerebro]
        self.camadas = self.rede.desempacotar(self.cerebro.genomas[:1])
        self._reiniciar()

    def _reiniciar(self):
        self.frota = Frota(self.pista, 1, modo="corrida", reposicionar_parados=True)
        self.rastro = []
        self.batidas_vistas = 0

    def _trocar_pista(self, passo):
        if not self.arquivos_pista:
            return
        self.i_pista = (self.i_pista + passo) % len(self.arquivos_pista)
        self._usar_pista(Pista.carregar(self.arquivos_pista[self.i_pista]))
        if self.pista.id != self.cerebro.pista_id:
            self.aviso.mostrar("Esta pista é inédita para este cérebro: ele nunca a viu no treino!", 3, LARANJA)

    def voltar(self):
        if self.ao_voltar:
            self.ao_voltar()
            return
        from .menu import TelaMenu
        self.app.ir_para(TelaMenu(self.app, self.pista.id))

    # ------------------------------------------------------------------ laço
    def ao_redimensionar(self):
        area, self.rect_painel = dividir_tela(self.app.janela.get_size(), self.app.escala_ui)
        self.vista = Vista(area, margem=round(10 * self.app.escala_ui))
        self.fundo = self.vista.ajustar(self.imagem)

    def evento(self, ev):
        if ev.type != pygame.KEYDOWN:
            return
        k = ev.key
        if k == pygame.K_ESCAPE:
            self.voltar()
        elif k == pygame.K_SPACE:
            self.pausado = not self.pausado
        elif k == pygame.K_LEFT:
            self._trocar_pista(-1)
        elif k == pygame.K_RIGHT:
            self._trocar_pista(1)
        elif k == pygame.K_b:
            self._usar_cerebro(self.i_cerebro + 1)
            self.aviso.mostrar(f"Cérebro treinado na pista {self.cerebro.pista_nome}", 2)
        elif k in (pygame.K_UP, pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.i_vel = min(self.i_vel + 1, len(VELOCIDADES) - 1)
        elif k in (pygame.K_DOWN, pygame.K_MINUS, pygame.K_KP_MINUS):
            self.i_vel = max(self.i_vel - 1, 0)
        elif k == pygame.K_s:
            self.sensores = not self.sensores
        elif k == pygame.K_r:
            self._reiniciar()

    def atualizar(self, dt):
        if self.pausado:
            return
        f = self.frota
        for _ in range(VELOCIDADES[self.i_vel]):
            saida = self.rede.avaliar(self.camadas, f.entradas(np.array([0])))
            f.passo(saida[:, 0], saida[:, 1])
            self.rastro.append((f.x[0], f.y[0]))
        del self.rastro[:-240]
        if f.batidas[0] > self.batidas_vistas:
            self.batidas_vistas = f.batidas[0]
            self.rastro = []
            self.aviso.mostrar("Bateu! O carro volta para o meio da pista.", 1.5, LARANJA)

    # ------------------------------------------------------------------ desenho
    def desenhar(self, sup):
        e = self.app.escala_ui
        v = self.vista
        f = self.frota
        sup.fill(FUNDO)
        sup.blit(self.fundo, v.rect_mundo)
        if len(self.rastro) > 1:
            pygame.draw.lines(sup, (255, 226, 120), False, [v.tela(x, y) for x, y in self.rastro], max(1, round(2 * e)))
        if self.sensores:
            desenho.desenhar_sensores(sup, v, f.x[0], f.y[0], f.angulo[0], f.angulos_sensores, f.distancias[0])
        desenho.desenhar_carros(sup, v, f.x, f.y, f.angulo, DESTAQUE, contorno=(255, 255, 255))
        self._desenhar_painel(sup)
        if self.pausado:
            texto(sup, "PAUSADO", self.app.fontes.titulo, TEXTO, (v.rect.centerx, v.rect_mundo.y + 16 * e), "midtop")
        self.aviso.desenhar(sup, self.app.fontes, v.rect.centerx, v.rect_mundo.bottom - 64 * e)

    def _desenhar_painel(self, sup):
        f = self.frota
        e = self.app.escala_ui
        fontes = self.app.fontes
        p = Painel(sup, self.rect_painel, fontes)
        fim = p.rodape([("← →", "trocar de pista"), ("B", "trocar de cérebro"), ("Espaço", "pausar"),
                        ("↑ ↓", "velocidade"), ("S", "mostrar sensores"), ("R", "recomeçar"),
                        ("Esc", "voltar")])
        p.titulo("Assistir a IA", self.pista.nome)
        p.linha("Cérebro de", self.cerebro.pista_nome)
        p.linha("Treinado até", f"geração {self.cerebro.geracao}")
        if self.pista.id != self.cerebro.pista_id:
            p.paragrafo("Pista inédita para este cérebro: é um teste de generalização!", LARANJA)
        p.secao("Volta")
        p.linha("Volta", str(f.voltas[0] + 1))
        p.linha("Tempo da volta", tempo(f.tempo - f.inicio_volta[0]))
        p.linha("Última volta", tempo(f.tempos_volta[0][-1]) if f.tempos_volta[0] else "—")
        p.linha("Melhor volta", tempo(f.melhor_volta(0)), VERDE)
        p.linha("Batidas", str(f.batidas[0]), LARANJA if f.batidas[0] else TEXTO)
        p.linha("Velocidade", f"{numero(kmh(f.velocidade[0]), 0)} km/h", DESTAQUE)
        p.linha("Simulação", "pausada" if self.pausado else f"{VELOCIDADES[self.i_vel]}×")
        p.secao("Rede neural")
        area = p.area(fim - p.y)
        if area.h > 70 * e:
            desenho.desenhar_rede(sup, area, fontes, self.rede, self.cerebro.genomas[0], f.entradas(np.array([0]))[0])

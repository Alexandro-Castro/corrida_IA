"""Tela de treino: a população inteira dirigindo, a evolução geração a geração e a rede neural do líder."""

import math
import time

import numpy as np
import pygame

from .. import config, desenho
from ..app import Tela
from ..interface import (AZUL, DESTAQUE, FUNDO, LARANJA, TEXTO, VERDE, Aviso, Painel, Vista, dividir_tela, numero,
                         tempo, texto)
from ..treino import Treinador

VELOCIDADES = (1, 2, 4, 8, 16, 32, 0)     # passos de simulação por quadro; 0 = o máximo possível
COR_CARRO = (86, 170, 255)
COR_ELIMINADO = (105, 108, 116)


class TelaTreino(Tela):
    def __init__(self, app, pista, continuar=False):
        super().__init__(app)
        self.pista = pista
        self.treinador = Treinador(pista, continuar=continuar)
        self.imagem = desenho.imagem_pista(pista).convert()
        pista.liberar_campos()
        self.i_vel = 0
        self.pausado = False
        self.todos_sensores = False
        self.mostrar_eliminados = True
        self.ritmo = 1.0          # segundos simulados por segundo real (média móvel)
        self.aviso = Aviso()
        if self.treinador.geracao > 1:
            self.aviso.mostrar(f"Continuando o treino a partir da geração {self.treinador.geracao}", 3)

    def ao_redimensionar(self):
        area, self.rect_painel = dividir_tela(self.app.janela.get_size(), self.app.escala_ui)
        self.vista = Vista(area, margem=round(10 * self.app.escala_ui))
        self.fundo = self.vista.ajustar(self.imagem)

    def voltar(self):
        from .menu import TelaMenu
        self.app.ir_para(TelaMenu(self.app, self.pista.id))

    # ------------------------------------------------------------------ laço
    def evento(self, ev):
        if ev.type != pygame.KEYDOWN:
            return
        k = ev.key
        if k == pygame.K_ESCAPE:
            self.voltar()
        elif k == pygame.K_SPACE:
            self.pausado = not self.pausado
        elif k in (pygame.K_UP, pygame.K_RIGHT, pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.i_vel = min(self.i_vel + 1, len(VELOCIDADES) - 1)
        elif k in (pygame.K_DOWN, pygame.K_LEFT, pygame.K_MINUS, pygame.K_KP_MINUS):
            self.i_vel = max(self.i_vel - 1, 0)
        elif pygame.K_1 <= k <= pygame.K_7:
            self.i_vel = k - pygame.K_1
        elif k == pygame.K_s:
            self.todos_sensores = not self.todos_sensores
        elif k == pygame.K_m:
            self.mostrar_eliminados = not self.mostrar_eliminados
        elif k == pygame.K_k:
            self.treinador.encerrar_agora()
            self._fim_de_geracao()

    def atualizar(self, dt):
        if self.pausado:
            return
        vel = VELOCIDADES[self.i_vel]
        inicio = time.perf_counter()
        limite = 0.045 if vel == 0 else 0.03      # nunca trava a janela por muito tempo
        passos = 0
        while (vel == 0 or passos < vel) and time.perf_counter() - inicio < limite:
            if self.treinador.passo():
                self._fim_de_geracao()
            passos += 1
        self.ritmo += (passos * config.DT / max(dt, 1e-3) - self.ritmo) * 0.1

    def _fim_de_geracao(self):
        r = self.treinador.resumo
        mensagem = f"Geração {r['geracao']}: o melhor andou {numero(r['melhor'])} voltas"
        if r["novo_recorde"]:
            self.aviso.mostrar(mensagem + " — novo recorde, cérebro salvo!", 3, DESTAQUE)
        else:
            self.aviso.mostrar(mensagem, 2)

    # ------------------------------------------------------------------ desenho
    def desenhar(self, sup):
        e = self.app.escala_ui
        sup.fill(FUNDO)
        sup.blit(self.fundo, self.vista.rect_mundo)
        t = self.treinador
        f = t.frota
        v = self.vista
        lider = f.lider()
        vivos = f.vivo

        if self.mostrar_eliminados and not vivos.all():
            m = ~vivos
            desenho.desenhar_carros(sup, v, f.x[m], f.y[m], f.angulo[m], COR_ELIMINADO, alfa=110, parabrisa=False)
        if self.todos_sensores:
            for i in np.flatnonzero(vivos):
                desenho.desenhar_sensores(sup, v, f.x[i], f.y[i], f.angulo[i], f.angulos_sensores, f.distancias[i],
                                          destaque=False)
        elite = np.arange(f.n) < t.n_elite
        for mascara, cor in ((vivos & ~elite, COR_CARRO), (vivos & elite, LARANJA)):
            desenho.desenhar_carros(sup, v, f.x[mascara], f.y[mascara], f.angulo[mascara], cor)
        desenho.desenhar_sensores(sup, v, f.x[lider], f.y[lider], f.angulo[lider], f.angulos_sensores, f.distancias[lider])
        desenho.desenhar_carros(sup, v, f.x[lider], f.y[lider], f.angulo[lider], DESTAQUE, contorno=(255, 255, 255))

        self._desenhar_legenda(sup)
        self._desenhar_painel(sup, lider)
        if self.pausado:
            texto(sup, "PAUSADO", self.app.fontes.titulo, TEXTO, (v.rect.centerx, v.rect_mundo.y + 16 * e), "midtop")
        self.aviso.desenhar(sup, self.app.fontes, v.rect.centerx, v.rect_mundo.bottom - 64 * e)

    def _desenhar_legenda(self, sup):
        f = self.app.fontes
        e = self.app.escala_ui
        itens = [(DESTAQUE, "líder (com sensores)")]
        if self.treinador.n_elite:
            itens.append((LARANJA, f"elite: os {self.treinador.n_elite} melhores da geração anterior"))
        itens += [(COR_CARRO, "população"), (COR_ELIMINADO, "eliminados")]
        x = self.vista.rect_mundo.x + round(12 * e)
        y = self.vista.rect_mundo.bottom - round(12 * e) - f.pequena.get_linesize()
        for cor, rotulo in itens:
            pygame.draw.circle(sup, cor, (x + 5 * e, y + f.pequena.get_linesize() / 2), 5 * e)
            r = texto(sup, rotulo, f.pequena, (235, 238, 240), (x + 14 * e, y))
            x = r.right + round(16 * e)

    def _desenhar_painel(self, sup, lider):
        t = self.treinador
        f = t.frota
        L = self.pista.comprimento
        fontes = self.app.fontes
        e = self.app.escala_ui
        p = Painel(sup, self.rect_painel, fontes)
        fim = p.rodape([("Espaço", "pausar / continuar"), ("↑ ↓  1-7", "velocidade da simulação"),
                        ("S", "sensores de todos os carros"), ("M", "mostrar / esconder eliminados"),
                        ("K", "encerrar a geração agora"), ("Esc", "voltar ao menu (o treino fica salvo)")])
        p.titulo("Treino", self.pista.nome)
        p.linha("Geração", str(t.geracao))
        p.linha("Carros na pista", f"{int(f.vivo.sum())} de {f.n}")
        p.linha("Tempo da geração", f"{numero(f.tempo, 1)} / {numero(config.TEMPO_MAX_GERACAO, 0)} s")
        vel = VELOCIDADES[self.i_vel]
        if self.pausado:
            ritmo = "pausado"
        elif vel:
            ritmo = f"{vel}×"
        else:
            ritmo = f"máxima (~{numero(self.ritmo, 0)}×)"
        p.linha("Velocidade", ritmo)
        p.linha("Líder agora", f"{numero(max(f.progresso[lider], 0) / L)} voltas", DESTAQUE)
        recorde = t.recorde / L if math.isfinite(t.recorde) else None
        p.linha("Recorde", f"{numero(recorde)} voltas" if recorde is not None else "—",
                VERDE if recorde is not None else TEXTO)
        voltas = [s for s in t.historico_volta if not math.isnan(s)]
        p.linha("Melhor volta", tempo(min(voltas)) if voltas else "—")

        p.secao("Evolução — voltas por geração")
        area = p.area(150 * e)
        desenho.desenhar_grafico(sup, area, fontes, [
            ([v / L for v in t.ag.historico_melhor], DESTAQUE, "melhor"),
            ([v / L for v in t.ag.historico_media], AZUL, "média"),
        ], unidade="voltas")
        p.secao("Rede neural do líder")
        area = p.area(fim - p.y)
        if area.h > 70 * e:
            desenho.desenhar_rede(sup, area, fontes, t.rede, t.ag.populacao[lider], f.entradas(np.array([lider]))[0])

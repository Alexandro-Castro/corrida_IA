"""Corrida: você contra a IA — ou um teste de pista sozinho, a partir do editor."""

import math

import numpy as np
import pygame

from .. import config, desenho
from ..app import Tela
from ..carros import Frota
from ..interface import (AZUL, DESTAQUE, FUNDO, LARANJA, PAINEL, ROXO, TEXTO, TEXTO_FRACO, VERDE, VERMELHO, Aviso,
                         Painel, Vista, dividir_tela, kmh, numero, tempo, texto)
from ..rede_neural import Rede

CORES_IA = (AZUL, VERDE, ROXO)
COR_JOGADOR = VERMELHO


class TelaCorrida(Tela):
    def __init__(self, app, pista, cerebro=None, voltas=3, ao_voltar=None):
        """cerebro=None e voltas=None: teste de pista (sozinho e sem fim)."""
        super().__init__(app)
        self.pista = pista.preparar()
        self.cerebro = cerebro
        self.total_voltas = voltas
        self.ao_voltar = ao_voltar
        self.imagem = desenho.imagem_pista(pista).convert()
        pista.liberar_campos()
        self.rede = Rede.padrao()
        self.n_ia = min(3, len(cerebro.genomas)) if cerebro is not None else 0
        if self.n_ia:
            self.camadas = self.rede.desempacotar(cerebro.genomas[:self.n_ia])
        self.aviso = Aviso()
        self.reiniciar()

    def reiniciar(self):
        n = self.n_ia + 1
        self.jogador = n - 1                       # o jogador larga em último no grid
        self.frota = Frota(self.pista, n, modo="corrida", grade=self.n_ia > 0,
                           reposicionar_parados=np.arange(n) != self.jogador)
        self.nomes = [f"IA {i + 1}" for i in range(self.n_ia)] + ["Você"]
        self.cores = list(CORES_IA[:self.n_ia]) + [COR_JOGADOR]
        self.chegada = np.full(n, np.nan)
        self.direcao = 0.0
        self.acumulado = 0.0
        self.batidas_vistas = 0
        self.estado = "contagem" if self.n_ia else "correndo"
        self.contagem = 3.0

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
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                self.voltar()
            elif ev.key == pygame.K_r:
                self.reiniciar()

    def atualizar(self, dt):
        if self.estado == "contagem":
            self.contagem -= dt
            if self.contagem <= 0:
                self.estado = "correndo"
            return
        if self.estado != "correndo":
            return
        # a física anda em passos fixos, em tempo real, mesmo se a tela rodar a menos de 60 quadros/s
        self.acumulado += dt
        passos = 0
        while self.acumulado >= config.DT and passos < 4:
            self.acumulado -= config.DT
            passos += 1
            self._passo()
        if passos == 4:
            self.acumulado = 0.0

    def _passo(self):
        f = self.frota
        ativos = np.flatnonzero(np.isnan(self.chegada))
        direcao = np.zeros(f.n)
        acelerador = np.zeros(f.n)
        ia = ativos[ativos != self.jogador]
        if ia.size:
            saida = self.rede.avaliar(self.rede.selecionar(self.camadas, ia), f.entradas(ia))
            direcao[ia] = saida[:, 0]
            acelerador[ia] = saida[:, 1]
        teclas = pygame.key.get_pressed()
        alvo = int(teclas[pygame.K_RIGHT] or teclas[pygame.K_d]) - int(teclas[pygame.K_LEFT] or teclas[pygame.K_a])
        self.direcao += (alvo - self.direcao) * min(1.0, 12 * config.DT)
        direcao[self.jogador] = self.direcao
        if teclas[pygame.K_UP] or teclas[pygame.K_w]:
            acelerador[self.jogador] = 1.0
        elif teclas[pygame.K_DOWN] or teclas[pygame.K_s]:
            acelerador[self.jogador] = -1.0
        f.passo(direcao[ativos], acelerador[ativos], ativos)

        if f.batidas[self.jogador] > self.batidas_vistas:
            self.batidas_vistas = f.batidas[self.jogador]
            self.aviso.mostrar("Bateu! Você volta para o meio da pista, parado.", 1.5, LARANJA)
        if self.total_voltas:
            chegaram = ativos[f.voltas[ativos] >= self.total_voltas]
            self.chegada[chegaram] = f.tempo
            if self.jogador in chegaram:
                self.estado = "fim"

    def classificacao(self):
        """Quem já chegou, pelo tempo; depois os demais, pela distância percorrida."""
        f = self.frota
        return sorted(range(f.n), key=lambda i: (0, self.chegada[i]) if not np.isnan(self.chegada[i])
                      else (1, -f.progresso[i]))

    # ------------------------------------------------------------------ desenho
    def desenhar(self, sup):
        e = self.app.escala_ui
        fontes = self.app.fontes
        v = self.vista
        f = self.frota
        sup.fill(FUNDO)
        sup.blit(self.fundo, v.rect_mundo)
        for i in range(f.n):
            alfa = 140 if not np.isnan(self.chegada[i]) and i != self.jogador else 255
            contorno = (255, 255, 255) if i == self.jogador else None
            desenho.desenhar_carros(sup, v, f.x[i], f.y[i], f.angulo[i], self.cores[i], alfa, contorno)
            if self.n_ia:
                desenho.desenhar_rotulo(sup, v, fontes, f.x[i], f.y[i], self.nomes[i], self.cores[i])
        self._desenhar_painel(sup)

        centro = v.rect_mundo.center
        if self.estado == "contagem":
            numero_contagem = math.ceil(self.contagem)
            texto(sup, str(numero_contagem), fontes.enorme, DESTAQUE, centro, "center")
        elif self.estado == "correndo" and self.n_ia and f.tempo < 0.8:
            texto(sup, "VAI!", fontes.enorme, VERDE, centro, "center")
        elif self.estado == "fim":
            self._desenhar_resultado(sup)
        self.aviso.desenhar(sup, fontes, v.rect.centerx, v.rect_mundo.bottom - 64 * e)

    def _desenhar_painel(self, sup):
        f = self.frota
        j = self.jogador
        e = self.app.escala_ui
        p = Painel(sup, self.rect_painel, self.app.fontes)
        p.rodape([("↑ ou W", "acelerar"), ("↓ ou S", "frear"), ("← → ou A D", "virar"), ("R", "recomeçar"),
                  ("Esc", "sair")])
        if self.total_voltas:
            p.titulo("Corrida", f"{self.pista.nome} · {self.total_voltas} voltas")
            p.linha("Volta", f"{min(f.voltas[j] + 1, self.total_voltas)} de {self.total_voltas}")
        else:
            p.titulo("Teste de pista", self.pista.nome)
            p.linha("Volta", str(f.voltas[j] + 1))
        p.linha("Tempo total", tempo(f.tempo))
        p.linha("Volta atual", tempo(f.tempo - f.inicio_volta[j]))
        p.linha("Melhor volta", tempo(f.melhor_volta(j)), VERDE)
        p.linha("Batidas", str(f.batidas[j]), LARANJA if f.batidas[j] else TEXTO)
        p.linha("Velocidade", f"{numero(kmh(f.velocidade[j]), 0)} km/h", DESTAQUE)
        if not self.n_ia:
            p.espaco(10)
            p.paragrafo("Teste a sua pista: se estiver difícil demais para você, a IA também vai sofrer — "
                        "mas ela aprende!")
            return
        p.secao("Classificação")
        for posicao, i in enumerate(self.classificacao(), 1):
            if not np.isnan(self.chegada[i]):
                situacao = f"chegou: {tempo(self.chegada[i])}"
            else:
                situacao = f"volta {min(f.voltas[i] + 1, self.total_voltas)}"
            y = p.y
            pygame.draw.circle(sup, self.cores[i], (p.x + 6 * e, y + p.f.normal.get_linesize() / 2), 6 * e)
            p.x += round(18 * e)
            p.w -= round(18 * e)
            p.linha(f"{posicao}º  {self.nomes[i]}", situacao, TEXTO if i != j else DESTAQUE)
            p.x -= round(18 * e)
            p.w += round(18 * e)

    def _desenhar_resultado(self, sup):
        e = self.app.escala_ui
        fontes = self.app.fontes
        ordem = self.classificacao()
        posicao = ordem.index(self.jogador) + 1
        caixa = pygame.Rect(0, 0, round(560 * e), round((150 + 34 * len(ordem)) * e))
        caixa.center = self.vista.rect_mundo.center
        fundo = pygame.Surface(caixa.size, pygame.SRCALPHA)
        pygame.draw.rect(fundo, (*PAINEL, 235), fundo.get_rect(), border_radius=round(16 * e))
        sup.blit(fundo, caixa)
        pygame.draw.rect(sup, DESTAQUE, caixa, max(1, round(2 * e)), border_radius=round(16 * e))
        titulo = "Você venceu!" if posicao == 1 else f"Você chegou em {posicao}º lugar"
        texto(sup, titulo, fontes.titulo, DESTAQUE if posicao == 1 else TEXTO, (caixa.centerx, caixa.y + 20 * e), "midtop")
        y = caixa.y + 80 * e
        f = self.frota
        for pos, i in enumerate(ordem, 1):
            cor = DESTAQUE if i == self.jogador else TEXTO
            texto(sup, f"{pos}º   {self.nomes[i]}", fontes.negrito, cor, (caixa.x + 40 * e, y))
            final = tempo(self.chegada[i]) if not np.isnan(self.chegada[i]) else "ainda correndo"
            texto(sup, f"{final}   ·   melhor volta {tempo(f.melhor_volta(i))}", fontes.normal, TEXTO_FRACO,
                  (caixa.right - 40 * e, y), "topright")
            y += 34 * e
        texto(sup, "R: correr de novo   ·   Esc: sair", fontes.pequena, TEXTO_FRACO,
              (caixa.centerx, caixa.bottom - 30 * e), "midtop")

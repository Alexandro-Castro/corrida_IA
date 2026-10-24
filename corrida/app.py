"""A janela e o laço principal. Cada tela (menu, treino, editor...) cuida de si mesma."""

import pygame

from . import desenho
from .interface import Fontes, configurar_dpi


class Tela:
    """Base das telas. A cada quadro o App chama evento() → atualizar() → desenhar()."""

    def __init__(self, app):
        self.app = app

    def ao_redimensionar(self):
        pass

    def evento(self, ev):
        pass

    def atualizar(self, dt):
        pass

    def desenhar(self, sup):
        pass

    def ao_fechar(self):
        """Chamado quando a janela é fechada nesta tela (para salvar o que estiver em andamento)."""


class App:
    def __init__(self, tamanho=None):
        self.escala_ui = configurar_dpi()
        pygame.init()
        pygame.key.set_repeat(300, 40)
        if tamanho is None:
            largura, altura = pygame.display.get_desktop_sizes()[0]
            tamanho = (max(960, int(largura * 0.9)), max(600, int(altura * 0.85)))
        self.janela = pygame.display.set_mode(tamanho, pygame.RESIZABLE)
        try:
            pygame.Window.from_display_module().minimum_size = (960, 600)
        except (AttributeError, pygame.error):
            pass
        pygame.display.set_caption("Corrida IA — neuroevolução")
        pygame.display.set_icon(desenho.icone())
        self.fontes = Fontes(self.escala_ui)
        self.relogio = pygame.time.Clock()
        self.tela = None
        self.rodando = True

    def ir_para(self, tela):
        self.tela = tela
        tela.ao_redimensionar()

    def sair(self):
        self.rodando = False

    def rodar(self, tela):
        self.ir_para(tela)
        while self.rodando:
            dt = min(self.relogio.tick(60) / 1000, 0.1)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.tela.ao_fechar()
                    self.rodando = False
                elif ev.type == pygame.WINDOWSIZECHANGED:
                    self.janela = pygame.display.get_surface()
                    self.tela.ao_redimensionar()
                else:
                    self.tela.evento(ev)
            if not self.rodando:
                break
            self.tela.atualizar(dt)
            self.tela.desenhar(self.janela)
            pygame.display.flip()
        pygame.quit()

"""Menu principal: escolha a pista e o que fazer com ela."""

import importlib.util

import pygame

from .. import arquivos, config, desenho
from ..app import Tela
from ..interface import (DESTAQUE, FUNDO, LARANJA, LINHA, PAINEL, PAINEL_CLARO, TEXTO, TEXTO_FRACO, VERDE, Aviso,
                         Botao, numero, tempo, texto)
from ..pista import Pista


class TelaMenu(Tela):
    def __init__(self, app, pista_id=None):
        super().__init__(app)
        self.arquivos = Pista.listar()
        ids = [c.stem for c in self.arquivos]
        self.aviso = Aviso()
        self.miniaturas = {}
        self.confirmar_ate = 0
        self.tem_torch = importlib.util.find_spec("torch") is not None
        self.botoes = [
            Botao("Neuroevolução do zero", self.treinar, "T"),
            Botao("Continuar neuroevolução", self.continuar, "C"),
            Botao("Treinar com PPO", self.treinar_ppo, "P"),
            Botao("Assistir a IA", self.assistir, "A"),
            Botao("Correr contra a IA", self.correr, "J"),
            Botao("Comparar as técnicas", self.comparar, "K"),
            Botao("Editar esta pista", self.editar, "E"),
            Botao("Criar nova pista", self.nova_pista, "N"),
            Botao("Sair", app.sair, "Esc"),
        ]
        self.atalhos = dict(zip((pygame.K_t, pygame.K_c, pygame.K_p, pygame.K_a, pygame.K_j, pygame.K_k, pygame.K_e,
                                 pygame.K_n), self.botoes))
        self.seta_esq = self.seta_dir = self.cartao = self.rect_miniatura = pygame.Rect(0, 0, 0, 0)
        self._selecionar(ids.index(pista_id) if pista_id in ids else 0)

    # ------------------------------------------------------------------ seleção
    def _selecionar(self, indice):
        self.pista = None
        self.indice = 0
        if self.arquivos:
            self.indice = indice % len(self.arquivos)
            self.pista = Pista.carregar(self.arquivos[self.indice])
        self.cerebro = arquivos.cerebro_da_pista(self.pista) if self.pista else None
        self.cerebro_ppo = arquivos.cerebro_da_pista(self.pista, "ppo") if self.pista else None
        self.geracao_salva = arquivos.geracao_do_treino(self.pista) if self.pista else None
        self.ppo_salvo = self.pista is not None and arquivos.caminho_treino_ppo(self.pista.id).exists()
        self.algum_cerebro = self.cerebro or self.cerebro_ppo or next(iter(arquivos.cerebros_disponiveis()), None)
        tem_pista = self.pista is not None
        self.botoes[0].ativo = tem_pista
        self.botoes[1].ativo = self.geracao_salva is not None
        self.botoes[2].ativo = tem_pista and self.tem_torch
        self.botoes[2].rotulo = ("PPO (instale o PyTorch)" if not self.tem_torch
                                 else "Continuar PPO" if self.ppo_salvo else "Treinar com PPO")
        self.botoes[3].ativo = self.botoes[4].ativo = tem_pista and self.algum_cerebro is not None
        self.botoes[5].ativo = self.algum_cerebro is not None
        self.botoes[6].ativo = tem_pista
        self.confirmar_ate = 0

    # ------------------------------------------------------------------ ações
    def treinar(self):
        agora = pygame.time.get_ticks()
        if self.geracao_salva is not None and agora > self.confirmar_ate:
            self.confirmar_ate = agora + 4000
            self.aviso.mostrar("Isso recomeça do zero o treino salvo (o melhor cérebro é mantido). "
                               "Clique de novo para confirmar.", 4, LARANJA)
            return
        from .treino import TelaTreino
        self.app.ir_para(TelaTreino(self.app, self.pista))

    def continuar(self):
        from .treino import TelaTreino
        self.app.ir_para(TelaTreino(self.app, self.pista, continuar=True))

    def treinar_ppo(self):
        from .ppo import TelaPPO
        self.app.ir_para(TelaPPO(self.app, [self.pista], continuar=True))

    def assistir(self):
        from .assistir import TelaAssistir
        self.app.ir_para(TelaAssistir(self.app, self.pista))

    def correr(self):
        from .corrida import TelaCorrida
        from ..piloto import adversarios_padrao
        self.app.ir_para(TelaCorrida(self.app, self.pista, adversarios_padrao(self.pista)))

    def comparar(self):
        from .comparacao import TelaComparacao
        self.app.ir_para(TelaComparacao(self.app, self.pista.id if self.pista else None))

    def editar(self):
        from .editor import TelaEditor
        self.app.ir_para(TelaEditor(self.app, self.pista))

    def nova_pista(self):
        from .editor import TelaEditor
        self.app.ir_para(TelaEditor(self.app))

    # ------------------------------------------------------------------ laço
    def ao_redimensionar(self):
        largura, altura = self.app.janela.get_size()
        e = self.app.escala_ui
        m = round(40 * e)
        topo = round(150 * e)
        largura_botoes = round(330 * e)
        x_botoes = largura - m - largura_botoes
        espaco = round(12 * e)
        cabe = (altura - topo - m - espaco * (len(self.botoes) + 1)) / len(self.botoes)
        altura_botao = round(max(30 * e, min(52 * e, cabe)))
        for k, botao in enumerate(self.botoes):
            extra = espaco * 2 if k == len(self.botoes) - 1 else 0
            botao.rect = pygame.Rect(x_botoes, topo + k * (altura_botao + espaco) + extra, largura_botoes, altura_botao)
        self.cartao = pygame.Rect(m, topo, x_botoes - m - round(30 * e), altura - topo - m)
        pad = round(22 * e)
        area = pygame.Rect(self.cartao.x + pad, self.cartao.y + round(78 * e),
                           self.cartao.w - 2 * pad, self.cartao.h - round(78 * e) - round(96 * e))
        escala = min(area.w / config.LARGURA_MUNDO, area.h / config.ALTURA_MUNDO)
        self.rect_miniatura = pygame.Rect(0, 0, round(config.LARGURA_MUNDO * escala), round(config.ALTURA_MUNDO * escala))
        self.rect_miniatura.center = area.center
        lado = round(38 * e)
        self.seta_dir = pygame.Rect(self.cartao.right - pad - lado, self.cartao.y + pad, lado, lado)
        self.seta_esq = self.seta_dir.move(-lado - round(8 * e), 0)

    def evento(self, ev):
        for botao in self.botoes:
            if botao.tratar(ev):
                return
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.seta_esq.collidepoint(ev.pos):
                self._selecionar(self.indice - 1)
            elif self.seta_dir.collidepoint(ev.pos):
                self._selecionar(self.indice + 1)
        elif ev.type == pygame.MOUSEWHEEL and self.cartao.collidepoint(pygame.mouse.get_pos()):
            self._selecionar(self.indice - ev.y)
        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                self.app.sair()
            elif ev.key == pygame.K_LEFT:
                self._selecionar(self.indice - 1)
            elif ev.key == pygame.K_RIGHT:
                self._selecionar(self.indice + 1)
            elif ev.key in self.atalhos and self.atalhos[ev.key].ativo:
                self.atalhos[ev.key].acao()

    def _miniatura(self):
        chave = (self.indice, self.rect_miniatura.size)
        if chave not in self.miniaturas:
            escala = min(1.0, self.rect_miniatura.w / config.LARGURA_MUNDO)
            imagem = desenho.imagem_pista(self.pista, escala)
            self.miniaturas[chave] = pygame.transform.smoothscale(imagem, self.rect_miniatura.size)
        return self.miniaturas[chave]

    def desenhar(self, sup):
        e = self.app.escala_ui
        f = self.app.fontes
        sup.fill(FUNDO)
        m = round(40 * e)
        r = texto(sup, "CORRIDA IA", f.enorme, DESTAQUE, (m, round(14 * e)))
        subtitulo = "redes neurais aprendendo a pilotar: neuroevolução × aprendizado por reforço"
        if r.right + round(28 * e) + f.subtitulo.size(subtitulo)[0] < sup.get_width() - m:
            texto(sup, subtitulo, f.subtitulo, TEXTO_FRACO, (r.right + round(28 * e), r.bottom - round(22 * e)),
                  "bottomleft")

        c = self.cartao
        pad = round(22 * e)
        pygame.draw.rect(sup, PAINEL, c, border_radius=round(14 * e))
        if self.pista is None:
            texto(sup, "Nenhuma pista na pasta 'pistas'. Crie uma no editor!", f.subtitulo, TEXTO_FRACO, c.center, "center")
        else:
            texto(sup, self.pista.nome, f.titulo, TEXTO, (c.x + pad, c.y + pad))
            texto(sup, f"pista {self.indice + 1} de {len(self.arquivos)}   ·   ← → para trocar", f.pequena, TEXTO_FRACO,
                  (c.x + pad, c.y + pad + f.titulo.get_linesize()))
            self._desenhar_setas(sup)
            sup.blit(self._miniatura(), self.rect_miniatura)
            pygame.draw.rect(sup, LINHA, self.rect_miniatura, 1)
            y = self.rect_miniatura.bottom + round(14 * e)
            p = self.pista
            texto(sup, f"Comprimento: {numero(p.comprimento / 10, 0)} m    ·    largura: {numero(p.largura / 10, 1)} m",
                  f.normal, TEXTO_FRACO, (c.x + pad, y))
            segundos = numero(config.TEMPO_MAX_GERACAO, 0)
            for cerebro, nome, cor, detalhe in (
                    (self.cerebro, "Neuroevolução", VERDE,
                     lambda cb: f"geração {cb.geracao}"),
                    (self.cerebro_ppo, "PPO", DESTAQUE,
                     lambda cb: f"{numero(cb.experiencia / 3600, 1)} h de direção")):
                y += f.normal.get_linesize() + round(4 * e)
                if cerebro is None:
                    texto(sup, f"{nome}: ainda não treinado nesta pista", f.normal, TEXTO_FRACO, (c.x + pad, y))
                else:
                    texto(sup, f"{nome} ({detalhe(cerebro)}): {numero(cerebro.voltas)} voltas em {segundos} s  ·  "
                               f"melhor volta {tempo(cerebro.melhor_volta)}", f.negrito, cor, (c.x + pad, y))

        mouse = pygame.mouse.get_pos()
        for botao in self.botoes:
            botao.desenhar(sup, f, mouse)
        self.aviso.desenhar(sup, f, sup.get_width() / 2, sup.get_height() - round(70 * e))

    def _desenhar_setas(self, sup):
        mouse = pygame.mouse.get_pos()
        for rect, sentido in ((self.seta_esq, -1), (self.seta_dir, 1)):
            sobre = rect.collidepoint(mouse)
            pygame.draw.rect(sup, (48, 54, 68) if sobre else PAINEL_CLARO, rect, border_radius=round(8 * self.app.escala_ui))
            cx, cy = rect.center
            t = rect.w * 0.22
            pontos = [(cx + sentido * t, cy), (cx - sentido * t * 0.8, cy - t), (cx - sentido * t * 0.8, cy + t)]
            pygame.draw.polygon(sup, DESTAQUE if sobre else TEXTO, pontos)

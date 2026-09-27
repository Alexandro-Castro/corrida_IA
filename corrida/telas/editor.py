"""Editor de pistas: clique para criar pontos, arraste para ajustar — a pista se desenha sozinha."""

import math
from pathlib import Path

import numpy as np
import pygame
import pygame.gfxdraw

from .. import arquivos, config
from ..app import Tela
from ..interface import (DESTAQUE, FUNDO, LARANJA, PAINEL, PAINEL_CLARO, TEXTO, TEXTO_FRACO, VERDE, VERMELHO, Aviso,
                         Painel, Vista, dividir_tela, numero, texto)
from ..pista import LARGURA_MAX, LARGURA_MIN, Pista, nome_para_arquivo

GRAMA = (56, 120, 58)
GRADE = (64, 131, 66)
ASFALTO = (63, 66, 73)
BORDA = (228, 228, 228)
CIANO = (60, 220, 235)


class TelaEditor(Tela):
    def __init__(self, app, pista=None):
        super().__init__(app)
        if pista is not None:
            self.pontos = pista.pontos.tolist()
            self.largura = pista.largura
            self.nome = pista.nome
            self.arquivo = pista.arquivo
            self.ordem = pista.ordem
        else:
            self.pontos = []
            self.largura = 80.0
            self.nome = "Nova pista"
            self.arquivo = None
            self.ordem = None
        self.historico = []
        self.arrastando = None
        self.alterado = False
        self.digitando = None          # nome sendo digitado na hora de salvar
        self.sobrescrever = None
        self.sair_ate = 0
        self.verificar_em = None
        self.aviso = Aviso()
        self.vista = None
        self._reconstruir()

    # ------------------------------------------------------------------ estado
    def _guardar(self):
        """Guarda o estado atual para o Ctrl+Z."""
        self.historico.append(([list(p) for p in self.pontos], self.largura))
        del self.historico[:-100]
        self.alterado = True

    def _desfazer(self):
        if self.historico:
            self.pontos, self.largura = self.historico.pop()
            self._reconstruir()

    def _reconstruir(self, verificar=True):
        self.previa = Pista(self.pontos, self.largura, self.nome) if len(self.pontos) >= 3 else None
        self.fundo_previa = None
        if verificar:
            self._verificar()

    def _verificar(self):
        """Procura trechos que se encostam e trechos fora do mapa (é um cálculo mais pesado)."""
        self.verificar_em = None
        if self.previa is None:
            self.problema_perto = self.problema_fora = np.zeros((0, 2))
            return
        self.problema_perto = self.previa.centro[self.previa.trechos_sobrepostos()]
        self.problema_fora = self.previa.centro[self.previa.fora_do_mapa()]

    def _pista_atual(self):
        return Pista(self.pontos, self.largura, self.nome, self.arquivo, self.ordem)

    # ------------------------------------------------------------------ ações
    def _adicionar(self, x, y):
        if any(math.hypot(x - px, y - py) < 15 for px, py in self.pontos):
            return
        self._guardar()
        if len(self.pontos) < 3:
            self.pontos.append([x, y])
        else:
            # entra no trecho (entre dois pontos vizinhos) mais próximo do clique
            p = np.array(self.pontos)
            d = np.roll(p, -1, axis=0) - p
            t = np.clip(((x - p[:, 0]) * d[:, 0] + (y - p[:, 1]) * d[:, 1]) / np.maximum((d ** 2).sum(axis=1), 1e-9), 0, 1)
            distancia = np.hypot(p[:, 0] + t * d[:, 0] - x, p[:, 1] + t * d[:, 1] - y)
            self.pontos.insert(int(np.argmin(distancia)) + 1, [x, y])
        self._reconstruir()

    def _apagar(self, i):
        self._guardar()
        del self.pontos[i]
        self._reconstruir()

    def _mudar_largura(self, delta):
        nova = min(max(self.largura + delta, LARGURA_MIN), LARGURA_MAX)
        if nova != self.largura:
            self._guardar()
            self.largura = nova
            self._reconstruir()

    def _inverter(self):
        if len(self.pontos) >= 3:
            self._guardar()
            self.pontos = [self.pontos[0]] + self.pontos[:0:-1]
            self._reconstruir()

    def _largada_no_mouse(self):
        if not self.pontos:
            return
        x, y = self.vista.mundo(*pygame.mouse.get_pos())
        i = int(np.argmin([math.hypot(x - px, y - py) for px, py in self.pontos]))
        self._guardar()
        self.pontos = self.pontos[i:] + self.pontos[:i]
        self._reconstruir()

    def _nova(self):
        self._guardar()
        self.pontos = []
        self.nome = "Nova pista"
        self.arquivo = None
        self.ordem = None
        self._reconstruir()

    def _testar(self):
        if self.previa is None:
            self.aviso.mostrar("A pista precisa de pelo menos 3 pontos.", 2, LARANJA)
            return
        from .corrida import TelaCorrida
        self.app.ir_para(TelaCorrida(self.app, self._pista_atual(), voltas=None, ao_voltar=lambda: self.app.ir_para(self)))

    def _ver_ia(self):
        if self.previa is None:
            self.aviso.mostrar("A pista precisa de pelo menos 3 pontos.", 2, LARANJA)
            return
        if not arquivos.cerebros_disponiveis():
            self.aviso.mostrar("Ainda não há nenhum cérebro treinado. Treine a IA em alguma pista primeiro.", 3, LARANJA)
            return
        from .assistir import TelaAssistir
        self.app.ir_para(TelaAssistir(self.app, self._pista_atual(), ao_voltar=lambda: self.app.ir_para(self)))

    def _pedir_nome(self):
        if self.previa is None:
            self.aviso.mostrar("A pista precisa de pelo menos 3 pontos.", 2, LARANJA)
            return
        self.digitando = "" if self.nome == "Nova pista" else self.nome
        self.sobrescrever = None
        pygame.key.start_text_input()

    def _salvar(self, nome):
        nome = " ".join(nome.split()) or "Nova pista"
        destino = config.PASTA_PISTAS / f"{nome_para_arquivo(nome)}.json"
        mesmo_arquivo = self.arquivo is not None and destino.resolve() == Path(self.arquivo).resolve()
        if destino.exists() and not mesmo_arquivo and self.sobrescrever != destino:
            self.sobrescrever = destino
            self.aviso.mostrar(f"Já existe uma pista em {destino.name}. Enter de novo para substituir.", 4, LARANJA)
            return
        pista = Pista(self.pontos, self.largura, nome, ordem=self.ordem if mesmo_arquivo else None)
        pista.salvar(destino)
        self.nome, self.arquivo, self.ordem = nome, destino, pista.ordem
        self.alterado = False
        self.digitando = None
        self.aviso.mostrar(f"Pista salva em pistas/{destino.name}", 3, VERDE)

    def _sair(self):
        agora = pygame.time.get_ticks()
        if self.alterado and agora > self.sair_ate:
            self.sair_ate = agora + 3000
            self.aviso.mostrar("Há alterações não salvas. Esc de novo para sair assim mesmo (S salva).", 3, LARANJA)
            return
        from .menu import TelaMenu
        self.app.ir_para(TelaMenu(self.app, self.arquivo.stem if self.arquivo else None))

    # ------------------------------------------------------------------ laço
    def ao_redimensionar(self):
        area, self.rect_painel = dividir_tela(self.app.janela.get_size(), self.app.escala_ui)
        self.vista = Vista(area, margem=round(10 * self.app.escala_ui))
        self.fundo_previa = None

    def _ponto_sob(self, pos):
        raio = 11 * self.app.escala_ui
        melhor = None
        for i, (x, y) in enumerate(self.pontos):
            px, py = self.vista.tela(x, y)
            d = math.hypot(px - pos[0], py - pos[1])
            if d <= raio:
                melhor, raio = i, d
        return melhor

    def evento(self, ev):
        if self.digitando is not None:
            self._evento_nome(ev)
            return
        if ev.type == pygame.MOUSEBUTTONDOWN:
            i = self._ponto_sob(ev.pos)
            if ev.button == 1:
                if i is not None:
                    self._guardar()
                    self.arrastando = i
                elif self.vista.rect_mundo.collidepoint(ev.pos):
                    self._adicionar(*self.vista.mundo(*ev.pos))
            elif ev.button == 3 and i is not None:
                self._apagar(i)
        elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1 and self.arrastando is not None:
            self.arrastando = None
            self._verificar()
        elif ev.type == pygame.MOUSEMOTION and self.arrastando is not None:
            x, y = self.vista.mundo(*ev.pos)
            self.pontos[self.arrastando] = [min(max(x, 0), config.LARGURA_MUNDO), min(max(y, 0), config.ALTURA_MUNDO)]
            self._reconstruir(verificar=False)
            self.verificar_em = pygame.time.get_ticks() + 120
        elif ev.type == pygame.MOUSEWHEEL:
            self._mudar_largura(5 * ev.y)
        elif ev.type == pygame.KEYDOWN:
            self._tecla(ev)

    def _tecla(self, ev):
        k = ev.key
        if k == pygame.K_z and ev.mod & pygame.KMOD_CTRL:
            self._desfazer()
        elif k == pygame.K_ESCAPE:
            self._sair()
        elif k == pygame.K_s:
            self._pedir_nome()
        elif k == pygame.K_t:
            self._testar()
        elif k == pygame.K_a:
            self._ver_ia()
        elif k == pygame.K_n:
            self._nova()
        elif k == pygame.K_i:
            self._inverter()
        elif k == pygame.K_p:
            self._largada_no_mouse()
        elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self._mudar_largura(5)
        elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self._mudar_largura(-5)
        elif k in (pygame.K_DELETE, pygame.K_BACKSPACE):
            i = self._ponto_sob(pygame.mouse.get_pos())
            if i is not None:
                self._apagar(i)

    def _evento_nome(self, ev):
        if ev.type == pygame.TEXTINPUT:
            if len(self.digitando) < 30:
                self.digitando += ev.text
        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                self.digitando = None
            elif ev.key == pygame.K_BACKSPACE:
                self.digitando = self.digitando[:-1]
            elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._salvar(self.digitando)

    def atualizar(self, dt):
        if self.verificar_em is not None and pygame.time.get_ticks() >= self.verificar_em:
            self._verificar()

    # ------------------------------------------------------------------ desenho
    def _desenhar_previa(self):
        v = self.vista
        e = v.escala
        sup = pygame.Surface(v.rect_mundo.size)
        sup.fill(GRAMA)
        for x in range(100, config.LARGURA_MUNDO, 100):
            pygame.draw.line(sup, GRADE, (x * e, 0), (x * e, sup.get_height()))
        for y in range(100, config.ALTURA_MUNDO, 100):
            pygame.draw.line(sup, GRADE, (0, y * e), (sup.get_width(), y * e))
        if self.previa is not None:
            centro = self.previa.centro * e
            w = self.previa.meia_largura * e
            for x, y in centro:
                pygame.draw.circle(sup, BORDA, (x, y), w + 4 * e)
            for x, y in centro:
                pygame.draw.circle(sup, ASFALTO, (x, y), w)
            for i in range(0, len(centro) - 3, 6):
                pygame.draw.line(sup, (150, 152, 158), centro[i], centro[i + 3], 1)
        self.fundo_previa = sup

    def desenhar(self, sup):
        e = self.app.escala_ui
        fontes = self.app.fontes
        v = self.vista
        sup.fill(FUNDO)
        if self.fundo_previa is None:
            self._desenhar_previa()
        sup.blit(self.fundo_previa, v.rect_mundo)

        for pontos, cor in ((self.problema_perto, VERMELHO), (self.problema_fora, CIANO)):
            for x, y in pontos:
                pygame.draw.circle(sup, cor, v.tela(x, y), max(2, round(4 * e)))
        if len(self.pontos) >= 2:
            pygame.draw.lines(sup, (205, 215, 205), len(self.pontos) >= 3, [v.tela(x, y) for x, y in self.pontos], 1)
        if self.previa is not None:
            self._desenhar_largada(sup)

        sobre = self._ponto_sob(pygame.mouse.get_pos())
        for i, (x, y) in enumerate(self.pontos):
            px, py = (round(c) for c in v.tela(x, y))
            destacado = i in (sobre, self.arrastando)
            raio = round((9 if destacado else 7) * e)
            cor = VERDE if i == 0 else (DESTAQUE if destacado else (245, 245, 245))
            pygame.gfxdraw.filled_circle(sup, px, py, raio, cor)
            pygame.gfxdraw.aacircle(sup, px, py, raio, (20, 22, 26))
        if self.pontos:
            x, y = v.tela(*self.pontos[0])
            rotulo = fontes.pequena.render("largada", True, (255, 255, 255))
            caixa = rotulo.get_rect(midbottom=(x, y - 12 * e)).inflate(10 * e, 4 * e)
            pygame.draw.rect(sup, (20, 60, 32), caixa, border_radius=round(5 * e))
            sup.blit(rotulo, rotulo.get_rect(center=caixa.center))
        if not self.pontos:
            self._dica_inicial(sup)

        self._desenhar_painel(sup)
        if self.digitando is not None:
            self._desenhar_caixa_nome(sup)
        self.aviso.desenhar(sup, fontes, v.rect.centerx, v.rect_mundo.bottom - 64 * e)

    def _desenhar_largada(self, sup):
        v = self.vista
        x0, y0 = self.previa.centro[0]
        a = self.previa.angulo_largada
        w = self.previa.meia_largura
        nx, ny = -math.sin(a), math.cos(a)
        pygame.draw.line(sup, (255, 255, 255), v.tela(x0 - nx * w, y0 - ny * w), v.tela(x0 + nx * w, y0 + ny * w),
                         max(2, round(4 * v.escala)))
        # seta indicando o sentido da corrida
        ponta = (x0 + math.cos(a) * 70, y0 + math.sin(a) * 70)
        pygame.draw.line(sup, VERDE, v.tela(x0, y0), v.tela(*ponta), max(2, round(4 * v.escala)))
        asa = 18
        lados = [(ponta[0] - math.cos(a + s) * asa, ponta[1] - math.sin(a + s) * asa) for s in (0.5, -0.5)]
        pygame.draw.polygon(sup, VERDE, [v.tela(*ponta)] + [v.tela(*p) for p in lados])

    def _dica_inicial(self, sup):
        e = self.app.escala_ui
        fontes = self.app.fontes
        linhas = ["Clique no mapa para criar os pontos da pista,",
                  "na ordem em que os carros vão percorrê-la.",
                  "A curva passa suavemente por todos os pontos."]
        altura = fontes.subtitulo.get_linesize() * len(linhas) + round(30 * e)
        largura = max(fontes.subtitulo.size(linha)[0] for linha in linhas) + round(48 * e)
        caixa = pygame.Rect(0, 0, largura, altura)
        caixa.center = self.vista.rect_mundo.center
        fundo = pygame.Surface(caixa.size, pygame.SRCALPHA)
        pygame.draw.rect(fundo, (12, 14, 18, 190), fundo.get_rect(), border_radius=round(12 * e))
        sup.blit(fundo, caixa)
        for k, linha in enumerate(linhas):
            texto(sup, linha, fontes.subtitulo, TEXTO, (caixa.centerx, caixa.y + 15 * e + k * fontes.subtitulo.get_linesize()),
                  "midtop")

    def _desenhar_caixa_nome(self, sup):
        e = self.app.escala_ui
        fontes = self.app.fontes
        escurecer = pygame.Surface(sup.get_size(), pygame.SRCALPHA)
        escurecer.fill((0, 0, 0, 120))
        sup.blit(escurecer, (0, 0))
        caixa = pygame.Rect(0, 0, round(520 * e), round(170 * e))
        caixa.center = self.vista.rect.center
        pygame.draw.rect(sup, PAINEL, caixa, border_radius=round(14 * e))
        pygame.draw.rect(sup, DESTAQUE, caixa, max(1, round(2 * e)), border_radius=round(14 * e))
        texto(sup, "Nome da pista", fontes.subtitulo, TEXTO, (caixa.x + 24 * e, caixa.y + 20 * e))
        campo = pygame.Rect(caixa.x + 24 * e, caixa.y + 62 * e, caixa.w - 48 * e, 44 * e)
        pygame.draw.rect(sup, PAINEL_CLARO, campo, border_radius=round(8 * e))
        cursor = "|" if pygame.time.get_ticks() // 500 % 2 == 0 else " "
        texto(sup, self.digitando + cursor, fontes.normal, TEXTO, (campo.x + 12 * e, campo.centery), "midleft")
        texto(sup, "Enter salva   ·   Esc cancela", fontes.pequena, TEXTO_FRACO, (caixa.x + 24 * e, campo.bottom + 14 * e))

    def _desenhar_painel(self, sup):
        p = Painel(sup, self.rect_painel, self.app.fontes)
        p.rodape([("Clique", "adicionar ponto"), ("Arrastar", "mover ponto"), ("Botão direito", "apagar ponto"),
                  ("Roda  + -", "largura da pista"), ("P", "largada no ponto do mouse"), ("I", "inverter o sentido"),
                  ("Ctrl+Z", "desfazer"), ("T", "testar dirigindo"), ("A", "ver a IA nesta pista"), ("S", "salvar"),
                  ("N", "nova pista"), ("Esc", "voltar ao menu")])
        p.titulo("Editor de pistas", self.nome + ("  (não salva)" if self.alterado else ""))
        p.linha("Pontos", str(len(self.pontos)))
        p.linha("Largura", f"{numero(self.largura, 0)} px")
        p.linha("Comprimento", f"{numero(self.previa.comprimento, 0)} px" if self.previa else "—")
        p.secao("Situação")
        if self.previa is None:
            p.paragrafo(f"Adicione pelo menos {3 - len(self.pontos)} ponto(s) para formar a pista.", LARANJA)
        else:
            if len(self.problema_perto):
                p.paragrafo("Trechos perto demais (em vermelho): o asfalto se encosta e o carro pode pular de um "
                            "trecho para o outro. Afaste os pontos ou diminua a largura.", VERMELHO)
            if len(self.problema_fora):
                p.paragrafo("Trechos saindo do mapa (em azul): a borda da tela vira parede.", CIANO)
            if not len(self.problema_perto) and not len(self.problema_fora):
                p.paragrafo("Pista pronta! Teste dirigindo (T) ou veja a IA nela (A).", VERDE)
        p.secao("Dicas")
        p.paragrafo("A largada fica no ponto verde e a seta mostra o sentido. Curvas fechadas obrigam a frear: "
                    "ótimo para ver a IA aprender a dosar o acelerador.")

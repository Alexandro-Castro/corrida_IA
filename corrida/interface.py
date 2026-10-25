"""Peças da interface: cores, fontes, textos, botões, painel lateral e conversão mundo ↔ tela."""

import sys

import pygame

from . import config
from .formato import kmh, numero, tempo  # noqa: F401 (usados pelas telas)

# --- Cores -------------------------------------------------------------------
FUNDO = (16, 18, 23)
PAINEL = (23, 26, 33)
PAINEL_CLARO = (33, 37, 46)
LINHA = (52, 57, 68)
TEXTO = (232, 234, 237)
TEXTO_FRACO = (148, 154, 164)
DESTAQUE = (255, 206, 0)
AZUL = (79, 179, 255)
LARANJA = (255, 150, 56)
VERMELHO = (255, 84, 84)
VERDE = (84, 214, 120)
ROXO = (186, 132, 255)


def misturar(c1, c2, t):
    t = max(0.0, min(1.0, float(t)))
    return tuple(round(a + (b - a) * t) for a, b in zip(c1, c2))


def escurecer(cor, fator):
    return tuple(round(v * fator) for v in cor)


# --- Tela --------------------------------------------------------------------
def configurar_dpi():
    """No Windows, pede a resolução nativa (sem o borrado da escala do sistema) e
    devolve o fator de escala da interface (1.0 = 96 dpi = 100%)."""
    if sys.platform != "win32":
        return 1.0
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except (AttributeError, OSError):
            ctypes.windll.user32.SetProcessDPIAware()
        return max(1.0, ctypes.windll.user32.GetDpiForSystem() / 96)
    except (AttributeError, OSError):
        return 1.0


class Fontes:
    def __init__(self, escala):
        self.escala = escala

        def fonte(tamanho, negrito=False, familia="segoeui,arial"):
            return pygame.font.SysFont(familia, max(8, round(tamanho * escala)), bold=negrito)

        titulo = "bahnschrift,segoeui,arial"
        self.enorme = fonte(96, True, titulo)
        self.titulo = fonte(34, True, titulo)
        self.subtitulo = fonte(20, familia="segoeuisemibold,segoeui,arial")
        self.numero = fonte(19, True, titulo)
        self.normal = fonte(16)
        self.negrito = fonte(16, True)
        self.pequena = fonte(13)
        self.mono = fonte(14, familia="consolas,couriernew")


def texto(sup, conteudo, fonte, cor, pos, ancora="topleft"):
    img = fonte.render(str(conteudo), True, cor)
    r = img.get_rect(**{ancora: (round(pos[0]), round(pos[1]))})
    sup.blit(img, r)
    return r


def quebrar_linhas(conteudo, fonte, largura):
    """Divide um texto em linhas que caibam na largura dada."""
    linhas, linha = [], ""
    for palavra in conteudo.split():
        tentativa = f"{linha} {palavra}".strip()
        if linha and fonte.size(tentativa)[0] > largura:
            linhas.append(linha)
            linha = palavra
        else:
            linha = tentativa
    if linha:
        linhas.append(linha)
    return linhas


def dividir_tela(tamanho, escala_ui, largura_painel=360):
    """Área da pista (esquerda) e do painel lateral (direita)."""
    largura, altura = tamanho
    lp = min(round(largura_painel * escala_ui), largura // 2)
    return pygame.Rect(0, 0, largura - lp, altura), pygame.Rect(largura - lp, 0, lp, altura)


class Vista:
    """Converte coordenadas do mundo em coordenadas da tela (e vice-versa) para uma área retangular."""

    def __init__(self, rect, margem=0):
        self.rect = pygame.Rect(rect).inflate(-2 * margem, -2 * margem)
        self.escala = min(self.rect.w / config.LARGURA_MUNDO, self.rect.h / config.ALTURA_MUNDO)
        w = config.LARGURA_MUNDO * self.escala
        h = config.ALTURA_MUNDO * self.escala
        self.ox = self.rect.x + (self.rect.w - w) / 2
        self.oy = self.rect.y + (self.rect.h - h) / 2
        self.rect_mundo = pygame.Rect(round(self.ox), round(self.oy), round(w), round(h))

    def tela(self, x, y):
        return self.ox + x * self.escala, self.oy + y * self.escala

    def mundo(self, px, py):
        return (px - self.ox) / self.escala, (py - self.oy) / self.escala

    def ajustar(self, imagem):
        """Imagem do mundo inteiro redimensionada para esta vista."""
        return pygame.transform.smoothscale(imagem, self.rect_mundo.size)


class Botao:
    def __init__(self, rotulo, acao, atalho=None, ativo=True):
        self.rotulo = rotulo
        self.acao = acao
        self.atalho = atalho
        self.ativo = ativo
        self.rect = pygame.Rect(0, 0, 0, 0)

    def desenhar(self, sup, fontes, mouse):
        e = fontes.escala
        sobre = self.ativo and self.rect.collidepoint(mouse)
        fundo = (48, 54, 68) if sobre else (PAINEL_CLARO if self.ativo else (27, 30, 37))
        raio = round(10 * e)
        pygame.draw.rect(sup, fundo, self.rect, border_radius=raio)
        pygame.draw.rect(sup, DESTAQUE if sobre else LINHA, self.rect, width=max(1, round(1.5 * e)),
                         border_radius=raio)
        cor = TEXTO if self.ativo else (92, 97, 108)
        texto(sup, self.rotulo, fontes.negrito, cor, (self.rect.x + 18 * e, self.rect.centery), "midleft")
        if self.atalho:
            texto(sup, self.atalho, fontes.pequena, TEXTO_FRACO if self.ativo else (80, 85, 95),
                  (self.rect.right - 16 * e, self.rect.centery), "midright")

    def tratar(self, ev):
        if (self.ativo and ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
                and self.rect.collidepoint(ev.pos)):
            self.acao()
            return True
        return False


class Painel:
    """Desenha o painel lateral e escreve nele de cima para baixo."""

    def __init__(self, sup, rect, fontes):
        self.sup = sup
        self.rect = rect
        self.f = fontes
        self.e = fontes.escala
        pygame.draw.rect(sup, PAINEL, rect)
        pygame.draw.line(sup, LINHA, rect.topleft, rect.bottomleft, 1)
        self.pad = round(18 * self.e)
        self.x = rect.x + self.pad
        self.y = rect.y + self.pad
        self.w = rect.w - 2 * self.pad

    def titulo(self, titulo, subtitulo=None):
        r = texto(self.sup, titulo, self.f.titulo, DESTAQUE, (self.x, self.y))
        self.y = r.bottom + round(2 * self.e)
        if subtitulo:
            r = texto(self.sup, subtitulo, self.f.normal, TEXTO_FRACO, (self.x, self.y))
            self.y = r.bottom
        self.y += round(10 * self.e)

    def secao(self, rotulo):
        self.y += round(8 * self.e)
        r = texto(self.sup, rotulo.upper(), self.f.pequena, TEXTO_FRACO, (self.x, self.y))
        pygame.draw.line(self.sup, LINHA, (r.right + 8 * self.e, r.centery), (self.x + self.w, r.centery), 1)
        self.y = r.bottom + round(6 * self.e)

    def linha(self, rotulo, valor, cor=TEXTO):
        r = texto(self.sup, rotulo, self.f.normal, TEXTO_FRACO, (self.x, self.y))
        texto(self.sup, valor, self.f.numero, cor, (self.x + self.w, r.centery), "midright")
        self.y = r.bottom + round(4 * self.e)

    def texto(self, conteudo, cor=TEXTO_FRACO, fonte=None):
        r = texto(self.sup, conteudo, fonte or self.f.normal, cor, (self.x, self.y))
        self.y = r.bottom + round(2 * self.e)

    def paragrafo(self, conteudo, cor=TEXTO_FRACO, fonte=None):
        """Texto com quebra de linha automática na largura do painel."""
        fonte = fonte or self.f.normal
        for linha in quebrar_linhas(conteudo, fonte, self.w):
            self.texto(linha, cor, fonte)

    def espaco(self, px):
        self.y += round(px * self.e)

    def area(self, altura):
        r = pygame.Rect(self.x, self.y, self.w, max(0, round(altura)))
        self.y = r.bottom + round(8 * self.e)
        return r

    def rodape(self, atalhos):
        """Lista de atalhos presa ao pé do painel. Devolve o y onde ela começa."""
        altura_linha = self.f.pequena.get_linesize()
        topo = self.rect.bottom - self.pad - altura_linha * len(atalhos)
        pygame.draw.line(self.sup, LINHA, (self.x, topo - 8 * self.e), (self.x + self.w, topo - 8 * self.e), 1)
        largura_tecla = max(self.f.pequena.size(t)[0] for t, _ in atalhos) + round(10 * self.e)
        for k, (tecla, descricao) in enumerate(atalhos):
            y = topo + k * altura_linha
            texto(self.sup, tecla, self.f.pequena, DESTAQUE, (self.x, y))
            texto(self.sup, descricao, self.f.pequena, TEXTO_FRACO, (self.x + largura_tecla, y))
        return topo - round(16 * self.e)


class Aviso:
    """Mensagem curta que aparece por alguns segundos."""

    def __init__(self):
        self.mensagem = None
        self.cor = TEXTO
        self.fim = 0

    def mostrar(self, mensagem, segundos=2.5, cor=TEXTO):
        self.mensagem = mensagem
        self.cor = cor
        self.fim = pygame.time.get_ticks() + int(segundos * 1000)

    def desenhar(self, sup, fontes, centro_x, y):
        if not self.mensagem or pygame.time.get_ticks() > self.fim:
            return
        img = fontes.negrito.render(self.mensagem, True, self.cor)
        pad = round(12 * fontes.escala)
        caixa = img.get_rect(midtop=(centro_x, y)).inflate(2 * pad, pad)
        fundo = pygame.Surface(caixa.size, pygame.SRCALPHA)
        pygame.draw.rect(fundo, (12, 14, 18, 215), fundo.get_rect(), border_radius=round(8 * fontes.escala))
        sup.blit(fundo, caixa)
        sup.blit(img, img.get_rect(center=caixa.center))

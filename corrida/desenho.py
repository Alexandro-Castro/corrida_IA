"""Tudo o que é desenhado na tela: pista, carros, sensores, gráfico de evolução e a rede neural."""

import math

import numpy as np
import pygame
import pygame.gfxdraw

from . import config
from .interface import (AZUL, LARANJA, LINHA, PAINEL_CLARO, TEXTO, TEXTO_FRACO, VERDE, VERMELHO,
                        escurecer, misturar, numero, texto)


# --- Pista -------------------------------------------------------------------
def imagem_pista(pista, escala=1.0, arvores=True):
    """Desenha a pista (grama, asfalto, zebras, linha de chegada e árvores) numa Surface
    que cobre o mundo inteiro na escala dada."""
    c = pista.campos(escala)
    dist, s, lado = c["dist"], c["s"], c["lado"]
    H, W = dist.shape
    w = pista.meia_largura
    L = pista.comprimento
    rng = np.random.default_rng(7)
    yy, xx = (np.mgrid[0:H, 0:W].astype(np.float32) + 0.5) / escala

    # grama, com as faixas de corte
    img = np.empty((H, W, 3), np.float32)
    img[:] = (56, 120, 58)
    img += ((((xx * 0.8 + yy * 0.45) // 90) % 2) * 9 - 4)[..., None]
    img += rng.normal(0, 3.0, (H, W, 1)).astype(np.float32)
    # sombra suave em volta do asfalto
    sombra = np.where(dist > w, np.clip((w + 7 - dist) / 7, 0, 1), 0)
    img *= (1 - 0.35 * sombra)[..., None]
    # asfalto
    asfalto = dist <= w
    img[asfalto] = np.array((63, 66, 73), np.float32) + rng.normal(0, 3.5, (int(asfalto.sum()), 1))
    # zebras nas bordas
    zebra = asfalto & (dist > w - 6)
    listra = ((s // 14) % 2).astype(bool)
    img[zebra & listra] = (206, 48, 48)
    img[zebra & ~listra] = (236, 236, 236)
    # faixa central tracejada
    tracejado = (dist < max(1.3, 0.8 / escala)) & ((s // 26) % 2 == 0)
    img[tracejado] = img[tracejado] * 0.35 + np.array((222, 222, 222)) * 0.65
    # linha de chegada quadriculada (em s = 0)
    s_rel = np.where(s > L / 2, s - L, s)
    chegada = asfalto & (dist <= w - 6) & (np.abs(s_rel) < 6)
    quadrado = ((np.floor(s_rel / 6) + np.floor((lado + w) / 6)) % 2).astype(bool)
    img[chegada & quadrado] = (22, 22, 22)
    img[chegada & ~quadrado] = (245, 245, 245)

    rgb = np.clip(img, 0, 255).astype(np.uint8)
    sup = pygame.surfarray.make_surface(np.ascontiguousarray(rgb.transpose(1, 0, 2)))
    if arvores:
        _desenhar_arvores(sup, dist, escala, w, rng)
    return sup


def _desenhar_arvores(sup, dist, escala, w, rng, quantidade=70):
    H, W = dist.shape
    angulos = np.linspace(0, 2 * math.pi, 8, endpoint=False)
    arvores = []
    for _ in range(4000):
        if len(arvores) >= quantidade:
            break
        x = rng.uniform(0, config.LARGURA_MUNDO)
        y = rng.uniform(0, config.ALTURA_MUNDO)
        r = rng.uniform(10, 22)
        # nenhuma parte da copa pode ficar perto do asfalto
        px = np.clip(((x + np.append(np.cos(angulos) * (r + 8), 0)) * escala).astype(int), 0, W - 1)
        py = np.clip(((y + np.append(np.sin(angulos) * (r + 8), 0)) * escala).astype(int), 0, H - 1)
        if np.any(dist[py, px] < w + 10):
            continue
        if any(math.hypot(x - ax, y - ay) < 0.8 * (r + ar) for ax, ay, ar in arvores):
            continue
        arvores.append((x, y, r))

    sombras = pygame.Surface(sup.get_size(), pygame.SRCALPHA)
    for x, y, r in arvores:
        pygame.draw.circle(sombras, (0, 0, 0, 70), ((x + 4) * escala, (y + 5) * escala), r * escala)
    sup.blit(sombras, (0, 0))
    for x, y, r in arvores:
        pygame.draw.circle(sup, (36, 92, 44), (x * escala, y * escala), r * escala)
        pygame.draw.circle(sup, (50, 118, 56), ((x - r * 0.22) * escala, (y - r * 0.22) * escala), r * 0.66 * escala)
        pygame.draw.circle(sup, (72, 146, 72), ((x - r * 0.4) * escala, (y - r * 0.4) * escala), r * 0.28 * escala)


# --- Carros ------------------------------------------------------------------
_ESCALA_FORMA = (config.COMPRIMENTO_CARRO / 24, config.LARGURA_CARRO / 12)
_CARROCERIA = np.array([(12, -4), (12, 4), (9, 6), (-10, 6), (-12, 4.5), (-12, -4.5), (-10, -6), (9, -6)],
                       dtype=float) * _ESCALA_FORMA
_PARABRISA = np.array([(6.5, -4.4), (6.5, 4.4), (2.0, 5.0), (2.0, -5.0)], dtype=float) * _ESCALA_FORMA


def _poligonos(forma, x, y, ang, vista):
    c = np.cos(ang)[:, None]
    s = np.sin(ang)[:, None]
    px = vista.ox + (x[:, None] + forma[:, 0] * c - forma[:, 1] * s) * vista.escala
    py = vista.oy + (y[:, None] + forma[:, 0] * s + forma[:, 1] * c) * vista.escala
    return np.stack([px, py], axis=-1).round().astype(int).tolist()


def desenhar_carros(sup, vista, x, y, ang, cor, alfa=255, contorno=None, parabrisa=True):
    x, y, ang = np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(ang)
    if x.size == 0:
        return
    cor_corpo = (*cor, alfa)
    cor_borda = (*(contorno or escurecer(cor, 0.45)), alfa)
    for pontos in _poligonos(_CARROCERIA, x, y, ang, vista):
        pygame.gfxdraw.filled_polygon(sup, pontos, cor_corpo)
        pygame.gfxdraw.aapolygon(sup, pontos, cor_borda)
    if parabrisa:
        for pontos in _poligonos(_PARABRISA, x, y, ang, vista):
            pygame.gfxdraw.filled_polygon(sup, pontos, (28, 34, 46, alfa))


def desenhar_sensores(sup, vista, x, y, ang, angulos_sensores, distancias, destaque=True):
    inicio = vista.tela(x, y)
    for a, d in zip(angulos_sensores, distancias):
        fim = vista.tela(x + math.cos(ang + a) * d, y + math.sin(ang + a) * d)
        if destaque:
            cor = misturar(VERMELHO, VERDE, d / config.ALCANCE_SENSOR)
            pygame.draw.line(sup, cor, inicio, fim, 2)
            if d < config.ALCANCE_SENSOR:
                pygame.draw.circle(sup, cor, fim, max(2, round(4 * vista.escala)))
        else:
            pygame.draw.aaline(sup, (205, 210, 220), inicio, fim)


def icone():
    """Ícone da janela: um carrinho amarelo."""
    sup = pygame.Surface((64, 64), pygame.SRCALPHA)
    pygame.draw.rect(sup, (23, 26, 33), sup.get_rect(), border_radius=14)

    class _Vista:
        ox, oy, escala = 32, 32, 2.1

    desenhar_carros(sup, _Vista, 0.0, 0.0, -math.pi / 4, (255, 206, 0))
    return sup


def desenhar_rotulo(sup, vista, fontes, x, y, rotulo, cor):
    """Nome flutuando acima de um carro."""
    px, py = vista.tela(x, y)
    texto(sup, rotulo, fontes.pequena, cor, (px, py - 16 * vista.escala - 4), "midbottom")


# --- Gráfico de evolução -----------------------------------------------------
def desenhar_grafico(sup, rect, fontes, series, unidade=""):
    """series: lista de (valores, cor, legenda)."""
    e = fontes.escala
    pygame.draw.rect(sup, PAINEL_CLARO, rect, border_radius=round(8 * e))
    n = max((len(v) for v, _, _ in series), default=0)
    if n == 0:
        texto(sup, "o gráfico aparece após a 1ª geração", fontes.pequena, TEXTO_FRACO, rect.center, "center")
        return
    topo = max((max(v) for v, _, _ in series if len(v)), default=1.0)
    topo = max(0.5, math.ceil(topo * 1.1 / 0.5) * 0.5)
    pad = round(10 * e)
    largura_eixo = fontes.pequena.size(numero(topo / 2, 2))[0] + round(6 * e)
    area = pygame.Rect(rect.x + pad + largura_eixo, rect.y + pad + fontes.pequena.get_linesize(),
                       rect.w - 2 * pad - largura_eixo, rect.h - 2 * pad - 2 * fontes.pequena.get_linesize())
    for frac in (0.0, 0.5, 1.0):
        y = area.bottom - frac * area.h
        pygame.draw.line(sup, LINHA, (area.x, y), (area.right, y), 1)
        valor = topo * frac
        casas = 1 if abs(valor * 10 - round(valor * 10)) < 1e-9 else 2
        texto(sup, numero(valor, casas), fontes.pequena, TEXTO_FRACO, (area.x - 6 * e, y), "midright")
    texto(sup, unidade, fontes.pequena, TEXTO_FRACO, (area.x, rect.y + pad), "topleft")
    texto(sup, f"{n} {'geração concluída' if n == 1 else 'gerações concluídas'}", fontes.pequena, TEXTO_FRACO,
          (area.right, area.bottom + 4 * e), "topright")

    legenda_x = area.right
    for valores, cor, legenda in reversed(series):
        r = texto(sup, legenda, fontes.pequena, cor, (legenda_x, rect.y + pad), "topright")
        legenda_x = r.x - round(12 * e)
        if not len(valores):
            continue
        pontos = [(area.x + (i / max(n - 1, 1)) * area.w, area.bottom - min(v / topo, 1.0) * area.h)
                  for i, v in enumerate(valores)]
        if len(pontos) == 1:
            pygame.draw.circle(sup, cor, pontos[0], round(3 * e))
        else:
            pygame.draw.lines(sup, cor, False, pontos, max(2, round(2 * e)))


# --- Rede neural -------------------------------------------------------------
MAX_CONEXOES = 300     # por camada


def rotulos_sensores():
    return [f"{a:+d}°" if a else "0°" for a in config.ANGULOS_SENSORES] + ["vel."]


def desenhar_rede(sup, rect, fontes, rede, genoma, entradas):
    """Desenha a rede de um carro: neurônios coloridos pela ativação (laranja = positivo,
    azul = negativo) e conexões acesas pelo sinal que passa por elas agora."""
    e = fontes.escala
    camadas = rede.desempacotar(genoma[None, :])
    ativacoes = [a[0] for a in rede.avaliar(camadas, entradas[None, :], com_ativacoes=True)]
    tamanhos = rede.tamanhos
    rotulos = rotulos_sensores()

    margem_esq = max(fontes.pequena.size(r)[0] for r in rotulos) + round(12 * e)
    margem_dir = fontes.pequena.size("acelerador")[0] + round(14 * e)
    xs = np.linspace(rect.x + margem_esq, rect.right - margem_dir, len(tamanhos))
    raio = max(2, min(round(7 * e), round(rect.h / (max(tamanhos) * 2.6))))

    def alturas(n):
        passo = min(rect.h / n, 34 * e)
        return [rect.centery + (i - (n - 1) / 2) * passo for i in range(n)]

    posicoes = [[(xs[k], y) for y in alturas(n)] for k, n in enumerate(tamanhos)]

    for k, (W, _) in enumerate(camadas):
        sinais = W[0] * ativacoes[k][:, None]
        # redes grandes (como as do PPO) têm milhares de conexões: desenha só as mais fortes agora
        ordem = np.argsort(np.abs(sinais), axis=None)[-MAX_CONEXOES:]
        for i, j in zip(*np.unravel_index(ordem, sinais.shape)):
            sinal = sinais[i, j]
            cor = misturar(PAINEL_CLARO, LARANJA if sinal > 0 else AZUL, 0.12 + abs(sinal) / 2.5)
            pygame.draw.line(sup, cor, posicoes[k][i], posicoes[k + 1][j], 2 if abs(sinal) > 1.2 else 1)

    for k, n in enumerate(tamanhos):
        for i, (x, y) in enumerate(posicoes[k]):
            a = ativacoes[k][i]
            cor = misturar((72, 76, 86), LARANJA if a > 0 else AZUL, abs(a))
            pygame.gfxdraw.filled_circle(sup, round(x), round(y), raio, cor)
            pygame.gfxdraw.aacircle(sup, round(x), round(y), raio, (190, 195, 205))

    for i, (x, y) in enumerate(posicoes[0]):
        texto(sup, rotulos[i], fontes.pequena, TEXTO_FRACO, (x - raio - 6 * e, y), "midright")

    direcao, acelerador = ativacoes[-1]
    saidas = [("direita" if direcao > 0 else "esquerda", direcao),
              ("acelera" if acelerador > 0 else "freia", acelerador)]
    for (rotulo, valor), (x, y) in zip(saidas, posicoes[-1]):
        texto(sup, rotulo, fontes.pequena, TEXTO, (x + raio + 6 * e, y - 1), "bottomleft")
        barra = pygame.Rect(x + raio + 6 * e, y + 3 * e, margem_dir - raio - 12 * e, max(3, round(4 * e)))
        pygame.draw.rect(sup, LINHA, barra)
        meio = barra.centerx
        largura = abs(valor) * barra.w / 2
        cor = LARANJA if valor > 0 else AZUL
        pygame.draw.rect(sup, cor, (meio if valor > 0 else meio - largura, barra.y, largura, barra.h))

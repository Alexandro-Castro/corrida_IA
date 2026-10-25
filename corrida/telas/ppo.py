"""
Tela de treino por PPO.

O treino roda numa thread separada (as atualizações da rede levam alguns décimos de
segundo e travariam a janela). Na pista aparecem:

- os carros do treino, explorando: cada um recomeçou num ponto aleatório e sorteia
  os comandos em volta do que a rede sugere — por isso dirigem meio "nervosos";
- o carro de teste (dourado): a política atual sem sorteio, largando da linha de
  chegada — é ele que mostra o que a rede já sabe fazer.

Como isso é bem diferente das gerações da neuroevolução, a tela abre com uma
explicação (tecla H).
"""

import math
import threading
import time

import numpy as np
import pygame
import torch

from .. import arquivos, desenho
from ..app import Tela
from ..carros import Frota
from ..interface import (DESTAQUE, FUNDO, LARANJA, PAINEL, TEXTO, TEXTO_FRACO, VERDE, VERMELHO, Aviso, Painel,
                         Vista, dividir_tela, numero, quebrar_linhas, tempo, texto)
from ..rl.ppo import TreinadorPPO

COR_EXPLORANDO = (86, 170, 255)

EXPLICACAO = [
    "Na neuroevolução há 100 redes diferentes, uma por carro, e cada uma só recebe nota no fim da "
    "geração. Por isso todos largam juntos e correm até ser eliminados.",
    "No PPO há uma rede só, dirigindo todos os carros. Ela aprende com cada passo de cada carro: "
    "avançou, ganha recompensa; bateu, perde tudo o que ainda andaria. Ninguém espera ninguém — "
    "quando um carro bate, só ele recomeça.",
    None,     # depende de onde os carros largam (abaixo)
    "Os carros azuis dirigem \"nervosos\" porque sorteiam cada comando em volta do que a rede sugere: "
    "é assim que o PPO experimenta. O carro dourado é a rede de verdade, sem sorteio, largando da linha "
    "— o equivalente ao líder da neuroevolução.",
]
LARGADA_ALEATORIA = ("Os carros recomeçam em pontos aleatórios para praticar a pista inteira desde o início. "
                     "Largando todos da linha, eles só treinariam até a primeira curva difícil e travariam nela "
                     "(na teste01, ficaram presos em 0,25 volta).")
LARGADA_NA_LINHA = ("Neste treino todos largam da linha de chegada, como na neuroevolução (--largada-na-linha). "
                    "Repare que eles só praticam o começo da pista: numa curva difícil, o treino pode travar.")


def curva_neuroevolucao(pista):
    """Fitness do melhor de cada geração × experiência, do treino de neuroevolução salvo desta pista."""
    estado = arquivos.carregar_treino(pista)
    if estado is None or "historico_experiencia" not in estado:
        return [], []
    melhor, experiencia = estado["historico_melhor"], estado["historico_experiencia"]
    if len(melhor) != len(experiencia):
        return [], []
    return list(experiencia / 3600), list(melhor / pista.comprimento)


class TelaPPO(Tela):
    def __init__(self, app, pistas, parametros=None, dispositivo="auto", continuar=True):
        super().__init__(app)
        self.pistas = pistas
        self.parametros = parametros
        self.dispositivo = dispositivo
        self.i_pista = 0
        self.sensores = True
        self.mostrar_treino = True
        self.recomecar_ate = 0
        self.vista = None
        self.aviso = Aviso()
        # a explicação aparece sozinha na primeira vez que a tela abre (depois, tecla H)
        self.explicando = not getattr(app, "ppo_explicado", False)
        app.ppo_explicado = True
        self._iniciar(continuar)
        self._mostrar_pista(0)

    # ------------------------------------------------------------------ treino (thread)
    def _iniciar(self, continuar):
        self.treinador = TreinadorPPO(self.pistas, self.parametros, self.dispositivo, continuar=continuar)
        self.parar = threading.Event()
        self.pausado = False
        self.erro = None
        self.recorde_novo = None
        self._publicar()
        self.thread = threading.Thread(target=self._treinar, daemon=True)
        self.thread.start()
        if self.treinador.passos:
            self.aviso.mostrar(f"Continuando o treino: {numero(self.treinador.passos / 1e6, 1)} milhões de passos já feitos", 3)

    def _treinar(self):
        try:
            while not self.parar.is_set():
                if self.pausado:
                    time.sleep(0.05)
                    continue
                registro = self.treinador.iteracao()
                self._publicar()
                if registro and registro["novo_recorde"]:
                    self.recorde_novo = registro
        except Exception as erro:          # mostra o erro na tela em vez de morrer em silêncio
            self.erro = erro

    def _publicar(self):
        """Copia a política atual para o carro de teste (as telas não mexem no PyTorch)."""
        rede = self.treinador.rede()
        self.rede = rede
        self.genoma = self.treinador.modelo.genoma()
        self.camadas = rede.desempacotar(self.genoma)

    def _encerrar(self):
        self.parar.set()
        self.thread.join(timeout=10)
        if self.treinador.salvar:
            self.treinador._salvar_checkpoint()

    def ao_fechar(self):
        self._encerrar()

    def voltar(self):
        self._encerrar()
        from .menu import TelaMenu
        self.app.ir_para(TelaMenu(self.app, self.pistas[self.i_pista].id))

    # ------------------------------------------------------------------ pista exibida
    def _mostrar_pista(self, indice):
        self.i_pista = indice % len(self.pistas)
        pista = self.pistas[self.i_pista]
        self.imagem = desenho.imagem_pista(pista).convert()
        pista.liberar_campos()
        if self.vista is not None:
            self.fundo = self.vista.ajustar(self.imagem)
        self.curva_ne = curva_neuroevolucao(pista) if len(self.pistas) == 1 else ([], [])
        self._reiniciar_teste()

    def _reiniciar_teste(self):
        self.teste = Frota(self.pistas[self.i_pista], 1, modo="corrida", reposicionar_parados=True)

    # ------------------------------------------------------------------ laço
    def ao_redimensionar(self):
        area, self.rect_painel = dividir_tela(self.app.janela.get_size(), self.app.escala_ui)
        self.vista = Vista(area, margem=round(10 * self.app.escala_ui))
        self.fundo = self.vista.ajustar(self.imagem)

    def evento(self, ev):
        if ev.type == pygame.MOUSEBUTTONDOWN and self.explicando:
            self.explicando = False
        if ev.type != pygame.KEYDOWN:
            return
        k = ev.key
        if k == pygame.K_h:
            self.explicando = not self.explicando
        elif k == pygame.K_ESCAPE:
            self.voltar()
        elif k == pygame.K_SPACE:
            self.pausado = not self.pausado
        elif k in (pygame.K_LEFT, pygame.K_RIGHT) and len(self.pistas) > 1:
            self._mostrar_pista(self.i_pista + (1 if k == pygame.K_RIGHT else -1))
        elif k == pygame.K_s:
            self.sensores = not self.sensores
        elif k == pygame.K_c:
            self.mostrar_treino = not self.mostrar_treino
        elif k == pygame.K_r:
            self._reiniciar_teste()
        elif k == pygame.K_n:
            agora = pygame.time.get_ticks()
            if agora > self.recomecar_ate:
                self.recomecar_ate = agora + 3000
                self.aviso.mostrar("N de novo para recomeçar o PPO do zero (o melhor cérebro é mantido).", 3, LARANJA)
            else:
                self._encerrar()
                self._iniciar(continuar=False)
                self._reiniciar_teste()
                self.aviso.mostrar("Treino PPO recomeçado do zero.", 2)

    def atualizar(self, dt):
        f = self.teste
        comandos = self.rede.avaliar(self.camadas, f.entradas(np.array([0])))
        f.passo(comandos[:, 0], comandos[:, 1])
        if self.recorde_novo is not None:
            r, self.recorde_novo = self.recorde_novo, None
            self.aviso.mostrar(f"Novo recorde: {numero(r['voltas'])} voltas — cérebro PPO salvo!", 3, DESTAQUE)

    # ------------------------------------------------------------------ desenho
    def desenhar(self, sup):
        e = self.app.escala_ui
        v = self.vista
        sup.fill(FUNDO)
        sup.blit(self.fundo, v.rect_mundo)
        if self.mostrar_treino:
            f = self.treinador.ambiente.frotas[self.i_pista]
            desenho.desenhar_carros(sup, v, f.x.copy(), f.y.copy(), f.angulo.copy(), COR_EXPLORANDO, alfa=170,
                                    parabrisa=False)
        t = self.teste
        if self.sensores:
            desenho.desenhar_sensores(sup, v, t.x[0], t.y[0], t.angulo[0], t.angulos_sensores, t.distancias[0])
        desenho.desenhar_carros(sup, v, t.x, t.y, t.angulo, DESTAQUE, contorno=(255, 255, 255))
        self._desenhar_legenda(sup)
        self._desenhar_painel(sup)
        if self.pausado:
            texto(sup, "TREINO PAUSADO", self.app.fontes.titulo, TEXTO, (v.rect.centerx, v.rect_mundo.y + 16 * e), "midtop")
        if self.erro is not None:
            texto(sup, f"Erro no treino: {self.erro}", self.app.fontes.negrito, VERMELHO,
                  (v.rect.centerx, v.rect_mundo.y + 60 * e), "midtop")
        if self.explicando:
            self._desenhar_explicacao(sup)
        self.aviso.desenhar(sup, self.app.fontes, v.rect.centerx, v.rect_mundo.bottom - 64 * e)

    def _desenhar_explicacao(self, sup):
        e = self.app.escala_ui
        f = self.app.fontes
        largura = min(round(760 * e), self.vista.rect.w - round(40 * e))
        pad = round(26 * e)
        paragrafos = [p or (LARGADA_ALEATORIA if self.treinador.p.largada_aleatoria else LARGADA_NA_LINHA)
                      for p in EXPLICACAO]
        blocos = [quebrar_linhas(p, f.normal, largura - 2 * pad) for p in paragrafos]
        linha = f.normal.get_linesize()
        altura = (f.subtitulo.get_linesize() + round(14 * e) + sum(len(b) * linha + round(10 * e) for b in blocos)
                  + f.pequena.get_linesize() + 2 * pad)
        caixa = pygame.Rect(0, 0, largura, altura)
        caixa.center = self.vista.rect_mundo.center
        fundo = pygame.Surface(caixa.size, pygame.SRCALPHA)
        pygame.draw.rect(fundo, (*PAINEL, 240), fundo.get_rect(), border_radius=round(14 * e))
        sup.blit(fundo, caixa)
        pygame.draw.rect(sup, DESTAQUE, caixa, max(1, round(2 * e)), border_radius=round(14 * e))
        y = caixa.y + pad
        r = texto(sup, "Por que o treino do PPO é diferente?", f.subtitulo, DESTAQUE, (caixa.x + pad, y))
        y = r.bottom + round(14 * e)
        for bloco in blocos:
            for conteudo in bloco:
                texto(sup, conteudo, f.normal, TEXTO, (caixa.x + pad, y))
                y += linha
            y += round(10 * e)
        texto(sup, "H ou um clique para fechar", f.pequena, TEXTO_FRACO, (caixa.x + pad, y))

    def _desenhar_legenda(self, sup):
        f = self.app.fontes
        e = self.app.escala_ui
        itens = [(DESTAQUE, "carro de teste: a política atual, sem sorteio")]
        if self.mostrar_treino:
            itens.append((COR_EXPLORANDO, "carros do treino, explorando"))
        x = self.vista.rect_mundo.x + round(12 * e)
        y = self.vista.rect_mundo.bottom - round(12 * e) - f.pequena.get_linesize()
        for cor, rotulo in itens:
            pygame.draw.circle(sup, cor, (x + 5 * e, y + f.pequena.get_linesize() / 2), 5 * e)
            r = texto(sup, rotulo, f.pequena, (235, 238, 240), (x + 14 * e, y))
            x = r.right + round(16 * e)

    def _desenhar_painel(self, sup):
        t = self.treinador
        e = self.app.escala_ui
        fontes = self.app.fontes
        p = Painel(sup, self.rect_painel, fontes)
        atalhos = [("H", "por que é diferente da neuroevolução?"),
                   ("Espaço", "pausar / continuar o treino"), ("S", "sensores"), ("C", "carros do treino"),
                   ("R", "recomeçar o carro de teste"), ("N  N", "recomeçar o PPO do zero"),
                   ("Esc", "voltar ao menu (o treino fica salvo)")]
        if len(self.pistas) > 1:
            atalhos.insert(0, ("← →", "pista exibida"))
        fim = p.rodape(atalhos)
        subtitulo = self.pistas[self.i_pista].nome
        if len(self.pistas) > 1:
            subtitulo += f"  ({self.i_pista + 1} de {len(self.pistas)} pistas)"
        p.titulo("Treino PPO", subtitulo)
        gpu = t.dispositivo.type == "cuda"
        nome_gpu = torch.cuda.get_device_name(t.dispositivo).replace("NVIDIA GeForce ", "") if gpu else ""
        p.linha("Rodando em", f"GPU {nome_gpu}" if gpu else "CPU", VERDE if gpu else TEXTO)
        p.linha("Rede", " · ".join(str(n) for n in t.modelo.tamanhos))
        p.linha("Carros em paralelo", str(t.ambiente.n))
        p.linha("Passos", f"{numero(t.passos / 1e6, 2)} mi")
        p.linha("Experiência", f"{numero(t.experiencia / 3600, 1)} h de direção")
        p.linha("Ritmo", f"{numero(t.passos / max(t.tempo_treino, 1e-9), 0)} passos/s")
        ultima = t.historico[-1] if t.historico else None
        p.linha("Última avaliação", f"{numero(ultima['voltas'])} voltas" if ultima else "—", DESTAQUE)
        p.linha("Recorde", f"{numero(t.recorde)} voltas" if math.isfinite(t.recorde) else "—", VERDE)
        voltas = [h["melhor_volta"] for h in t.historico if not math.isnan(h["melhor_volta"])]
        p.linha("Melhor volta", tempo(min(voltas)) if voltas else "—")
        p.linha("Exploração (desvio)", numero(t.ultima["std"]) if t.ultima else "—")

        p.secao("Avaliação justa × experiência")
        area = p.area(150 * e)
        series = [([h["experiencia"] / 3600 for h in t.historico], [h["voltas"] for h in t.historico], DESTAQUE, "PPO")]
        if self.curva_ne[0]:
            series.append((*self.curva_ne, VERDE, "neuroevolução"))
        desenho.desenhar_curvas(sup, area, fontes, series, "h de direção", "voltas em 45 s",
                                vazio="a 1ª avaliação sai em alguns segundos")
        p.secao("Rede do carro de teste")
        area = p.area(fim - p.y)
        if area.h > 70 * e:
            desenho.desenhar_rede(sup, area, fontes, self.rede, self.genoma, self.teste.entradas(np.array([0]))[0])

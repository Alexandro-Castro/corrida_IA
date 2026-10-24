"""
Comparação entre as técnicas de treino (neuroevolução × PPO).

1. Generalização: todos os cérebros salvos dirigem em todas as pistas — inclusive
   nas que nunca viram — com a mesma avaliação justa (largada parada, 45 s, bateu acabou).
2. Curvas de aprendizado: as duas técnicas treinam do zero com o mesmo orçamento de
   experiência (horas de direção simulada, somando todos os carros), sem mexer nos
   cérebros salvos. Mostra quem aprende com menos experiência e quem aprende em
   menos tempo de relógio.
"""

import json
import math
import time
from datetime import datetime

from . import arquivos, config
from .avaliacao import avaliar_todos
from .formato import numero, tempo
from .piloto import Piloto
from .pista import Pista

PASTA_RESULTADOS = config.RAIZ / "resultados"


# --- 1. Generalização ----------------------------------------------------------
def matriz_generalizacao(cerebros=None, arquivos_pista=None, ao_avancar=None):
    """Devolve (cerebros, pistas, resultados[i_pista][i_cerebro])."""
    cerebros = arquivos.cerebros_disponiveis() if cerebros is None else cerebros
    pistas = [Pista.carregar(c) for c in (Pista.listar() if arquivos_pista is None else arquivos_pista)]
    pilotos = [Piloto(c) for c in cerebros]
    resultados = []
    for k, pista in enumerate(pistas):
        resultados.append(avaliar_todos(pista.preparar(), [p.agir for p in pilotos]))
        pista.liberar_campos()
        if ao_avancar:
            ao_avancar(k + 1, len(pistas))
    return cerebros, pistas, resultados


def treinou_em(cerebro, pista):
    """O cérebro viu esta pista no treino? ('varias' guarda as pistas no nome)."""
    if cerebro.pista_id == pista.id:
        return True
    return cerebro.pista_id == "varias" and pista.nome in cerebro.pista_nome


def tabela_generalizacao(cerebros, pistas, resultados):
    """Tabela em Markdown: voltas em 45 s de cada cérebro (linhas) em cada pista (colunas).
    Em negrito: a pista em que o cérebro treinou. 'x' = bateu antes do tempo."""
    cabecalho = ["Cérebro", *[p.nome for p in pistas], "Média nas pistas vistas", "Média nas inéditas"]
    linhas = ["| " + " | ".join(cabecalho) + " |", "|" + "---|" * len(cabecalho)]
    for j, cerebro in enumerate(cerebros):
        celulas, vistas, ineditas = [], [], []
        for i, pista in enumerate(pistas):
            r = resultados[i][j]
            texto = numero(r.voltas) + (" x" if r.bateu else "")
            if treinou_em(cerebro, pista):
                texto = f"**{texto}**"
                vistas.append(r.voltas)
            else:
                ineditas.append(r.voltas)
            celulas.append(texto)
        media = lambda v: numero(sum(v) / len(v)) if v else "—"
        linhas.append(f"| {cerebro.descricao} | " + " | ".join(celulas) + f" | {media(vistas)} | {media(ineditas)} |")
    return "\n".join(linhas)


# --- 2. Curvas de aprendizado --------------------------------------------------
def curvas_aprendizado(pista, horas, dispositivo="auto", mostrar=print):
    """Treina as duas técnicas do zero, sem salvar cérebros, até `horas` de experiência cada.
    Cada curva: experiência (h), tempo de relógio (s) e voltas na avaliação justa."""
    from .treino import Treinador
    limite = horas * 3600
    ne = {"experiencia": [], "relogio": [], "voltas": [], "melhor_volta": []}
    treinador = Treinador(pista, salvar=False)
    inicio = time.perf_counter()
    while treinador.experiencia < limite:
        while not treinador.passo():
            pass
        r = treinador.resumo
        for chave, valor in (("experiencia", treinador.experiencia / 3600), ("relogio", time.perf_counter() - inicio),
                             ("voltas", r["melhor"]), ("melhor_volta", r["melhor_volta"])):
            ne[chave].append(valor)
        mostrar(f"  neuroevolução: geração {r['geracao']:3d} | {numero(ne['experiencia'][-1], 1):>5} h"
                f" | {numero(r['melhor']):>5} voltas")

    from .rl.ppo import ParametrosPPO, TreinadorPPO
    ppo = {"experiencia": [], "relogio": [], "voltas": [], "melhor_volta": []}
    parametros = ParametrosPPO(passos_planejados=int(limite / config.DT))
    treinador = TreinadorPPO([pista], parametros, dispositivo, salvar=False)
    while treinador.experiencia < limite:
        registro = treinador.iteracao()
        if registro is None:
            continue
        for chave, valor in (("experiencia", registro["experiencia"] / 3600), ("relogio", registro["tempo_treino"]),
                             ("voltas", registro["voltas"]), ("melhor_volta", registro["melhor_volta"])):
            ppo[chave].append(valor)
        mostrar(f"  PPO: {numero(treinador.passos / 1e6, 2)} mi passos | {numero(ppo['experiencia'][-1], 1):>5} h"
                f" | {numero(registro['voltas']):>5} voltas")
    return {"pista_id": pista.id, "pista_nome": pista.nome, "horas": horas,
            "dispositivo": treinador.dispositivo.type, "rede_ppo": list(treinador.modelo.tamanhos),
            "data": datetime.now().isoformat(timespec="seconds"), "neuroevolucao": ne, "ppo": ppo}


def marcos(curva):
    """Quando a técnica completou a primeira volta, e onde terminou."""
    primeira = next((k for k, v in enumerate(curva["voltas"]) if v >= 1.0), None)
    voltas = [v for v in curva["melhor_volta"] if not math.isnan(v)]
    return {
        "primeira_volta_h": curva["experiencia"][primeira] if primeira is not None else math.nan,
        "primeira_volta_s": curva["relogio"][primeira] if primeira is not None else math.nan,
        "voltas_final": max(curva["voltas"]) if curva["voltas"] else math.nan,
        "melhor_volta": min(voltas) if voltas else math.nan,
        "relogio_total": curva["relogio"][-1] if curva["relogio"] else math.nan,
    }


def caminho_curvas(pista_id):
    return PASTA_RESULTADOS / f"curvas_{pista_id}.json"


def salvar_curvas(dados):
    PASTA_RESULTADOS.mkdir(exist_ok=True)
    caminho = caminho_curvas(dados["pista_id"])
    caminho.write_text(json.dumps(dados, ensure_ascii=False, indent=1).replace("NaN", "null"), encoding="utf-8")
    return caminho


def carregar_curvas(pista_id):
    caminho = caminho_curvas(pista_id)
    if not caminho.exists():
        return None
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    for tecnica in ("neuroevolucao", "ppo"):
        dados[tecnica]["melhor_volta"] = [math.nan if v is None else v for v in dados[tecnica]["melhor_volta"]]
    return dados


def tabela_curvas(dados):
    linhas = ["| | Neuroevolução | PPO |", "|---|---|---|"]
    m = {t: marcos(dados[t]) for t in ("neuroevolucao", "ppo")}

    def h(v):
        return "não chegou" if math.isnan(v) else f"{numero(v, 1)} h"

    def s(v):
        return "não chegou" if math.isnan(v) else f"{numero(v, 0)} s"

    for rotulo, chave, fmt in (("1ª volta completa — experiência", "primeira_volta_h", h),
                               ("1ª volta completa — relógio", "primeira_volta_s", s),
                               ("Voltas em 45 s (melhor avaliação)", "voltas_final", numero),
                               ("Melhor volta", "melhor_volta", tempo),
                               ("Tempo total de relógio", "relogio_total", s)):
        linhas.append(f"| {rotulo} | {fmt(m['neuroevolucao'][chave])} | {fmt(m['ppo'][chave])} |")
    return "\n".join(linhas)


def imagem_curvas(dados, caminho):
    """Salva um PNG com as curvas: voltas × experiência e voltas × tempo de relógio."""
    import pygame

    from . import desenho
    from .interface import DESTAQUE, FUNDO, TEXTO, VERDE, Fontes, texto
    pygame.font.init()
    fontes = Fontes(1.25)
    sup = pygame.Surface((1600, 620))
    sup.fill(FUNDO)
    texto(sup, f"Neuroevolução × PPO — {dados['pista_nome']}", fontes.titulo, TEXTO, (40, 24))
    texto(sup, f"mesmo orçamento: {numero(dados['horas'], 0)} h de direção simulada para cada técnica · "
               f"PPO rodando em {dados['dispositivo'].upper()}", fontes.normal, (150, 156, 166), (40, 80))
    ne, ppo = dados["neuroevolucao"], dados["ppo"]
    for k, (eixo, rotulo, escala) in enumerate((("experiencia", "h de direção", 1), ("relogio", "min de relógio", 60))):
        area = pygame.Rect(40 + k * 780, 130, 740, 450)
        desenho.desenhar_curvas(sup, area, fontes, [
            ([x / escala for x in ne[eixo]], ne["voltas"], VERDE, "neuroevolução"),
            ([x / escala for x in ppo[eixo]], ppo["voltas"], DESTAQUE, "PPO"),
        ], rotulo, "voltas em 45 s")
    pygame.image.save(sup, str(caminho))


# --- Relatório -----------------------------------------------------------------
def escrever_relatorio(cerebros, pistas, resultados):
    """resultados/comparacao.md: a matriz de generalização e as curvas que já foram medidas."""
    PASTA_RESULTADOS.mkdir(exist_ok=True)
    partes = [
        "# Neuroevolução × PPO",
        f"Gerado em {datetime.now():%d/%m/%Y %H:%M} por `python main.py comparar`.",
        "## Generalização: voltas em 45 s em cada pista",
        "Mesma avaliação para todos: larga parado da linha de chegada e dirige até bater ou acabar o tempo. "
        "Em **negrito**, a pista (ou pistas) em que o cérebro treinou; `x` = bateu antes do tempo.",
        tabela_generalizacao(cerebros, pistas, resultados),
    ]
    curvas = [carregar_curvas(p.id) for p in pistas]
    curvas = [c for c in curvas if c is not None]
    if curvas:
        partes.append("## Curvas de aprendizado (do zero, mesmo orçamento de experiência)")
    for dados in curvas:
        partes += [f"### {dados['pista_nome']} — {numero(dados['horas'], 0)} h de direção para cada técnica",
                   f"![curvas](curvas_{dados['pista_id']}.png)", tabela_curvas(dados)]
    caminho = PASTA_RESULTADOS / "comparacao.md"
    caminho.write_text("\n\n".join(partes) + "\n", encoding="utf-8")
    return caminho

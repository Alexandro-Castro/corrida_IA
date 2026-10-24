"""
Corrida IA — redes neurais aprendendo a pilotar, por neuroevolução e por Aprendizado por Reforço (PPO).

    python main.py                              abre o menu
    python main.py treinar --pista classica     neuroevolução, com a janela aberta
    python main.py treinar --pista classica --sem-janela --geracoes 100
    python main.py ppo --pista classica --sem-janela --milhoes 5
    python main.py ppo --pistas classica,desafio,feijao,serpentina --sem-janela --milhoes 20
    python main.py assistir --pista oval        a IA treinada dirigindo
    python main.py correr --pista feijao        você contra a IA
    python main.py editor --pista serpentina    editor de pistas
"""

import argparse
import sys

from corrida import arquivos, config
from corrida.pista import Pista


def carregar_pista(nome):
    if nome is None:
        arquivos_pista = Pista.listar()
        if not arquivos_pista:
            sys.exit("Nenhuma pista encontrada na pasta 'pistas'.")
        return Pista.carregar(arquivos_pista[0])
    caminho = config.PASTA_PISTAS / f"{nome}.json"
    if not caminho.exists():
        disponiveis = ", ".join(c.stem for c in Pista.listar())
        sys.exit(f"Pista '{nome}' não encontrada. Disponíveis: {disponiveis}")
    return Pista.carregar(caminho)


def importar_ppo():
    try:
        from corrida.rl import ppo
    except ImportError:
        sys.exit("O PPO precisa do PyTorch. Instale com:\n"
                 "  .venv\\Scripts\\python -m pip install -r requirements-rl.txt")
    return ppo


def parametros_ppo(args, ppo):
    p = ppo.ParametrosPPO()
    if args.camadas:
        p.camadas = tuple(int(c) for c in args.camadas.split(","))
    if args.carros:
        p.carros = args.carros
    return p


def main():
    parser = argparse.ArgumentParser(description="Redes neurais aprendendo a pilotar numa pista de corrida.")
    sub = parser.add_subparsers(dest="modo")
    treinar = sub.add_parser("treinar", help="treinar a IA por neuroevolução")
    treinar.add_argument("--pista", help="nome do arquivo da pista, sem .json (ex.: classica)")
    treinar.add_argument("--continuar", action="store_true", help="continuar o treino salvo")
    treinar.add_argument("--sem-janela", action="store_true", help="treinar sem desenhar (bem mais rápido)")
    treinar.add_argument("--geracoes", type=int, default=100, help="gerações a treinar no modo sem janela")

    ppo = sub.add_parser("ppo", help="treinar a IA por Aprendizado por Reforço (PPO, com PyTorch)")
    ppo.add_argument("--pista", help="pista de treino")
    ppo.add_argument("--pistas", help="várias pistas ao mesmo tempo, separadas por vírgula")
    ppo.add_argument("--continuar", action="store_true", help="continuar o treino salvo")
    ppo.add_argument("--sem-janela", action="store_true", help="treinar sem desenhar (mais rápido)")
    ppo.add_argument("--milhoes", type=float, default=5, help="milhões de passos a treinar no modo sem janela")
    ppo.add_argument("--dispositivo", default="auto", choices=("auto", "cuda", "cpu"), help="GPU (cuda) ou CPU")
    ppo.add_argument("--camadas", help="camadas ocultas, ex.: 256,256 (padrão: 256,256 na GPU e 64,64 na CPU)")
    ppo.add_argument("--carros", type=int, help="carros em paralelo, divididos entre as pistas (padrão: 256)")

    for nome, ajuda in (("assistir", "ver a IA treinada dirigindo"), ("correr", "correr contra a IA"),
                        ("editor", "editor de pistas")):
        sub.add_parser(nome, help=ajuda).add_argument("--pista", help="nome do arquivo da pista, sem .json")
    args = parser.parse_args()

    if args.modo == "treinar" and args.sem_janela:
        from corrida.treino import treinar_sem_janela
        treinar_sem_janela(carregar_pista(args.pista), args.geracoes, args.continuar)
        return
    if args.modo == "ppo":
        modulo = importar_ppo()
        pistas = [carregar_pista(n) for n in args.pistas.split(",")] if args.pistas else [carregar_pista(args.pista)]
        if args.sem_janela:
            modulo.treinar_ppo_sem_janela(pistas, args.milhoes, parametros_ppo(args, modulo), args.dispositivo,
                                          args.continuar)
            return

    from corrida.app import App
    from corrida.telas.menu import TelaMenu

    app = App()
    if args.modo is None:
        tela = TelaMenu(app)
    elif args.modo == "ppo":
        from corrida.telas.ppo import TelaPPO
        tela = TelaPPO(app, pistas, parametros_ppo(args, modulo), args.dispositivo, args.continuar)
    else:
        pista = carregar_pista(args.pista)
        if args.modo == "treinar":
            from corrida.telas.treino import TelaTreino
            tela = TelaTreino(app, pista, continuar=args.continuar)
        elif args.modo == "editor":
            from corrida.telas.editor import TelaEditor
            tela = TelaEditor(app, pista)
        else:
            cerebros = arquivos.cerebros_disponiveis()
            if not cerebros:
                sys.exit("Ainda não há cérebro treinado. Rode primeiro: python main.py treinar")
            if args.modo == "assistir":
                from corrida.telas.assistir import TelaAssistir
                tela = TelaAssistir(app, pista)
            else:
                from corrida.telas.corrida import TelaCorrida
                from corrida.piloto import adversarios_padrao
                tela = TelaCorrida(app, pista, adversarios_padrao(pista))
    app.rodar(tela)


if __name__ == "__main__":
    main()

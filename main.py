"""
Corrida IA — redes neurais aprendendo a pilotar por neuroevolução.

    python main.py                              abre o menu
    python main.py treinar --pista classica     treina com a janela aberta
    python main.py treinar --pista classica --sem-janela --geracoes 100
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


def main():
    parser = argparse.ArgumentParser(description="Redes neurais aprendendo a pilotar por neuroevolução.")
    sub = parser.add_subparsers(dest="modo")
    treinar = sub.add_parser("treinar", help="treinar a IA numa pista")
    treinar.add_argument("--pista", help="nome do arquivo da pista, sem .json (ex.: classica)")
    treinar.add_argument("--continuar", action="store_true", help="continuar o treino salvo")
    treinar.add_argument("--sem-janela", action="store_true", help="treinar sem desenhar (bem mais rápido)")
    treinar.add_argument("--geracoes", type=int, default=100, help="gerações a treinar no modo sem janela")
    for nome, ajuda in (("assistir", "ver a IA treinada dirigindo"), ("correr", "correr contra a IA"),
                        ("editor", "editor de pistas")):
        sub.add_parser(nome, help=ajuda).add_argument("--pista", help="nome do arquivo da pista, sem .json")
    args = parser.parse_args()

    if args.modo == "treinar" and args.sem_janela:
        from corrida.treino import treinar_sem_janela
        treinar_sem_janela(carregar_pista(args.pista), args.geracoes, args.continuar)
        return

    from corrida.app import App
    from corrida.telas.menu import TelaMenu

    app = App()
    if args.modo is None:
        tela = TelaMenu(app)
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
                tela = TelaCorrida(app, pista, arquivos.cerebro_da_pista(pista) or cerebros[0])
    app.rodar(tela)


if __name__ == "__main__":
    main()

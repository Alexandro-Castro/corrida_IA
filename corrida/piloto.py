"""Piloto: um cérebro treinado (de qualquer técnica) pronto para dirigir."""

from . import arquivos


class Piloto:
    def __init__(self, cerebro, indice=0):
        self.cerebro = cerebro
        self.indice = indice
        self.rede = cerebro.rede
        self.genoma = cerebro.genomas[indice]
        self.camadas = self.rede.desempacotar(self.genoma)

    @property
    def nome(self):
        """Nome curto, para aparecer em cima do carro."""
        if self.cerebro.metodo == "ppo":
            return "PPO"
        return f"Neuro {self.indice + 1}"

    def agir(self, entradas):
        """O que os sensores veem (k, n_entradas) → comandos (k, 2): direção e acelerador."""
        return self.rede.avaliar(self.camadas, entradas)


def adversarios_padrao(pista, quantidade=3):
    """Os rivais de uma corrida: os melhores cérebros desta pista, alternando entre as
    técnicas quando houver mais de uma (neuroevolução contra PPO). Sem nenhum cérebro
    desta pista, usa o de outra."""
    cerebros = [c for c in (arquivos.cerebro_da_pista(pista, m) for m in arquivos.METODOS) if c is not None]
    if not cerebros:
        cerebros = arquivos.cerebros_disponiveis()[:1]
    pilotos = []
    for indice in range(quantidade):
        for cerebro in cerebros:
            if len(pilotos) < quantidade and indice < len(cerebro.genomas):
                pilotos.append(Piloto(cerebro, indice))
    return pilotos

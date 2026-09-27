"""Números e tempos no formato brasileiro (sem depender do pygame)."""


def numero(valor, casas=2):
    """1234.5 → '1.234,50'."""
    return f"{valor:,.{casas}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def tempo(segundos):
    """Tempo de volta ou de prova: '14,42 s' ou '1:05,32'."""
    if segundos is None or segundos != segundos:     # None ou nan
        return "—"
    if segundos < 60:
        return f"{numero(segundos)} s"
    minutos, resto = divmod(segundos, 60)
    return f"{int(minutos)}:{numero(resto).zfill(5)}"


def kmh(velocidade):
    """Converte px/s em km/h, considerando 1 px = 10 cm (o carro tem 2,4 m)."""
    return velocidade * 0.36

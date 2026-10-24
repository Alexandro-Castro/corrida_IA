"""
PPO (Proximal Policy Optimization) em PyTorch.

Como funciona, em uma iteração:

  1. Coleta: todos os carros do ambiente dirigem por alguns passos. A rede do *ator*
     dá a média de uma distribuição normal para cada comando; o comando usado é
     sorteado dessa distribuição — é assim que o carro explora.
  2. Vantagem: a rede do *crítico* estima quanto de recompensa o carro ainda vai
     ganhar a partir de cada situação. Comparando com o que de fato aconteceu
     (GAE), sabemos se cada ação foi melhor ou pior do que o esperado.
  3. Atualização: o ator aumenta a probabilidade das ações melhores que o esperado
     e diminui a das piores — mas o "clip" do PPO impede que ele mude demais de
     uma vez, o que deixa o treino estável. O crítico aprende a prever melhor.

O ator termina numa tangente hiperbólica, como a rede da neuroevolução: a ação
"sem sorteio" (a média) é exatamente uma `Rede` comum, salva no mesmo formato
de cérebro e usada pelas telas de assistir e de corrida.
"""

import math
import time
from dataclasses import asdict, dataclass
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .. import arquivos, config
from ..avaliacao import avaliar, resumo
from ..formato import numero, tempo
from ..rede_neural import Rede
from .ambiente import AmbienteCorrida


@dataclass
class ParametrosPPO:
    camadas: tuple = None            # camadas ocultas do ator e do crítico; None = (256, 256) na
                                     # GPU e (64, 64) na CPU (na CPU a rede grande fica 2× mais lenta)
    carros: int = 256                # carros dirigindo em paralelo (divididos entre as pistas);
                                     # lotes maiores deixam o treino mais estável e aproveitam a GPU
    passos_por_coleta: int = 128     # passos de cada carro entre duas atualizações
    epocas: int = 10                 # quantas vezes cada coleta é usada na atualização
    minilotes: int = 8
    taxa_aprendizado: float = 3e-4   # cai em linha reta até 10% disso ao longo de `passos_planejados`
    passos_planejados: int = 10_000_000
    gamma: float = 0.99              # desconto: quanto o futuro vale (0,99 ≈ olhar ~1,7 s à frente)
    lambda_gae: float = 0.95
    clip: float = 0.2                # quanto a política pode mudar por atualização
    kl_alvo: float = 0.02            # para a atualização mais cedo se a política mudar demais
    coef_valor: float = 0.5
    coef_entropia: float = 0.0
    max_norma_gradiente: float = 0.5
    log_std_inicial: float = -0.5    # exploração inicial: desvio padrão ≈ 0,6
    largada_aleatoria: bool = True
    avaliar_a_cada: int = 5          # atualizações entre duas avaliações justas


def escolher_dispositivo(nome="auto"):
    if nome == "auto":
        nome = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(nome)


def _camada(entrada, saida, ganho):
    linear = nn.Linear(entrada, saida)
    nn.init.orthogonal_(linear.weight, ganho)
    nn.init.zeros_(linear.bias)
    return linear


def _mlp(entrada, camadas, saida, ganho_saida):
    partes, anterior = [], entrada
    for tamanho in camadas:
        partes += [_camada(anterior, tamanho, math.sqrt(2)), nn.Tanh()]
        anterior = tamanho
    partes.append(_camada(anterior, saida, ganho_saida))
    return nn.Sequential(*partes)


class AtorCritico(nn.Module):
    def __init__(self, n_observacoes, n_acoes, camadas, log_std_inicial):
        super().__init__()
        self.tamanhos = (n_observacoes, *camadas, n_acoes)
        self.ator = _mlp(n_observacoes, camadas, n_acoes, 0.01)
        self.critico = _mlp(n_observacoes, camadas, 1, 1.0)
        self.log_std = nn.Parameter(torch.full((n_acoes,), float(log_std_inicial)))

    def distribuicao(self, obs):
        media = torch.tanh(self.ator(obs))
        return torch.distributions.Normal(media, self.log_std.exp().expand_as(media))

    def valor(self, obs):
        return self.critico(obs).squeeze(-1)

    def genoma(self):
        """Os pesos do ator no formato da `Rede` (W de cada camada, linha a linha, e depois o bias)."""
        partes = []
        for modulo in self.ator:
            if isinstance(modulo, nn.Linear):
                partes.append(modulo.weight.detach().cpu().numpy().T.ravel())
                partes.append(modulo.bias.detach().cpu().numpy())
        return np.concatenate(partes).astype(np.float64)


def destino_do_treino(pistas):
    """Onde o cérebro é salvo: a própria pista, ou 'varias' quando treina em várias ao mesmo tempo
    (aí o fitness salvo é a média de voltas, e o 'comprimento' vale 1)."""
    if len(pistas) == 1:
        return pistas[0]
    nomes = ", ".join(p.nome for p in pistas)
    return SimpleNamespace(id="varias", nome=f"Várias ({nomes})", comprimento=1.0)


def caminho_checkpoint(destino):
    return arquivos.caminho_treino_ppo(destino.id)


class TreinadorPPO:
    def __init__(self, pistas, parametros=None, dispositivo="auto", continuar=False, salvar=True, semente=None):
        self.pistas = [p.preparar() for p in pistas]
        self.destino = destino_do_treino(self.pistas)
        self.salvar = salvar
        self.dispositivo = escolher_dispositivo(dispositivo)
        if semente is not None:
            torch.manual_seed(semente)
        estado = None
        if continuar and caminho_checkpoint(self.destino).exists():
            estado = torch.load(caminho_checkpoint(self.destino), map_location=self.dispositivo, weights_only=False)
            parametros = ParametrosPPO(**{**estado["parametros"], "camadas": tuple(estado["parametros"]["camadas"])})
        self.p = parametros or ParametrosPPO()
        if self.p.camadas is None:
            self.p.camadas = (256, 256) if self.dispositivo.type == "cuda" else (64, 64)

        carros_por_pista = max(8, self.p.carros // len(self.pistas))
        self.ambiente = AmbienteCorrida(self.pistas, carros_por_pista, self.p.largada_aleatoria, semente=semente)
        self.modelo = AtorCritico(self.ambiente.n_observacoes, 2, self.p.camadas, self.p.log_std_inicial)
        self.modelo.to(self.dispositivo)
        self.otimizador = torch.optim.Adam(self.modelo.parameters(), lr=self.p.taxa_aprendizado, eps=1e-5)
        self.passos = 0                 # passos de carro (somando todos os carros)
        self.atualizacoes = 0
        self.historico = []             # uma entrada por avaliação justa
        self.recorde = -math.inf        # média de voltas da melhor avaliação
        self.ultima = {}                # estatísticas da última atualização
        self.tempo_treino = 0.0         # segundos de relógio gastos treinando
        if estado is None and salvar:
            # um treino novo só substitui o cérebro salvo se fizer melhor do que ele
            salvo = arquivos.cerebro_da_pista(self.destino, "ppo")
            if salvo is not None and abs(salvo.comprimento_pista - self.destino.comprimento) <= 1.0:
                self.recorde = salvo.voltas
        if estado is not None:
            self.modelo.load_state_dict(estado["modelo"])
            self.otimizador.load_state_dict(estado["otimizador"])
            self.passos = estado["passos"]
            self.atualizacoes = estado["atualizacoes"]
            self.historico = estado["historico"]
            self.recorde = estado["recorde"]
            self.tempo_treino = estado.get("tempo_treino", 0.0)
        self.obs = self._tensor(self.ambiente.observar())

    # ------------------------------------------------------------------ arquivos
    @property
    def caminho_checkpoint(self):
        return caminho_checkpoint(self.destino)

    def _salvar_checkpoint(self):
        config.PASTA_CEREBROS.mkdir(parents=True, exist_ok=True)
        temporario = self.caminho_checkpoint.with_name(self.caminho_checkpoint.name + ".tmp")
        torch.save({"modelo": self.modelo.state_dict(), "otimizador": self.otimizador.state_dict(),
                    "tamanhos": self.modelo.tamanhos, "parametros": asdict(self.p), "passos": self.passos,
                    "atualizacoes": self.atualizacoes, "historico": self.historico, "recorde": self.recorde,
                    "tempo_treino": self.tempo_treino, "pistas": [p.id for p in self.pistas]}, temporario)
        temporario.replace(self.caminho_checkpoint)

    # ------------------------------------------------------------------ treino
    @property
    def experiencia(self):
        """Segundos de direção simulados, somando todos os carros (comparável à neuroevolução)."""
        return self.passos * config.DT

    def _tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.dispositivo)

    def iteracao(self):
        """Uma coleta e uma atualização. A cada `avaliar_a_cada`, faz a avaliação justa
        (e salva o cérebro se for recorde). Devolve o resultado da avaliação, ou None."""
        inicio = time.perf_counter()
        lote = self._coletar()
        self._atualizar(lote)
        self.atualizacoes += 1
        self.tempo_treino += time.perf_counter() - inicio
        if self.atualizacoes % self.p.avaliar_a_cada == 0:
            return self.avaliar()
        return None

    @torch.no_grad()
    def _coletar(self):
        T, N, p = self.p.passos_por_coleta, self.ambiente.n, self.p
        obs = torch.zeros((T, N, self.ambiente.n_observacoes), device=self.dispositivo)
        acoes = torch.zeros((T, N, 2), device=self.dispositivo)
        log_probs = torch.zeros((T, N), device=self.dispositivo)
        valores = torch.zeros((T, N), device=self.dispositivo)
        recompensas = torch.zeros((T, N), device=self.dispositivo)
        fins = torch.zeros((T, N), device=self.dispositivo)
        for t in range(T):
            dist = self.modelo.distribuicao(self.obs)
            acao = dist.sample()
            obs[t], acoes[t] = self.obs, acao
            log_probs[t] = dist.log_prob(acao).sum(-1)
            valores[t] = self.modelo.valor(self.obs)
            proxima, recompensa, terminou, cortado, final = self.ambiente.passo(acao.cpu().numpy())
            if cortado.any():
                # episódio cortado pelo tempo não é um fim de verdade: o futuro ainda valeria algo
                recompensa[cortado] += p.gamma * self.modelo.valor(self._tensor(final[cortado])).cpu().numpy()
            recompensas[t] = self._tensor(recompensa)
            fins[t] = self._tensor(terminou | cortado)
            self.obs = self._tensor(proxima)
        self.passos += T * N

        # GAE: vantagem de cada ação = quanto ela rendeu acima do que o crítico esperava
        vantagens = torch.zeros_like(recompensas)
        proximo_valor = self.modelo.valor(self.obs)
        acumulado = torch.zeros(N, device=self.dispositivo)
        for t in reversed(range(T)):
            continua = 1.0 - fins[t]
            valor_seguinte = proximo_valor if t == T - 1 else valores[t + 1]
            delta = recompensas[t] + p.gamma * valor_seguinte * continua - valores[t]
            acumulado = delta + p.gamma * p.lambda_gae * continua * acumulado
            vantagens[t] = acumulado
        retornos = vantagens + valores
        achatar = lambda x: x.reshape(T * N, *x.shape[2:])
        return {"obs": achatar(obs), "acoes": achatar(acoes), "log_probs": achatar(log_probs),
                "vantagens": achatar(vantagens), "retornos": achatar(retornos), "valores": achatar(valores)}

    def _atualizar(self, lote):
        p = self.p
        # passos largos no começo, passos finos no fim: evita desaprender o que já sabe
        fracao = max(0.1, 1.0 - self.passos / p.passos_planejados)
        for grupo in self.otimizador.param_groups:
            grupo["lr"] = p.taxa_aprendizado * fracao
        total = lote["obs"].shape[0]
        tamanho = total // p.minilotes
        kls, perdas_valor, entropias = [], [], []
        for _ in range(p.epocas):
            ordem = torch.randperm(total, device=self.dispositivo)
            for inicio in range(0, total, tamanho):
                i = ordem[inicio:inicio + tamanho]
                dist = self.modelo.distribuicao(lote["obs"][i])
                log_prob = dist.log_prob(lote["acoes"][i]).sum(-1)
                razao_log = log_prob - lote["log_probs"][i]
                razao = razao_log.exp()
                vant = lote["vantagens"][i]
                vant = (vant - vant.mean()) / (vant.std() + 1e-8)
                # o "clip" do PPO: ganhar com uma ação só até a razão sair de [1 - clip, 1 + clip]
                perda_politica = torch.max(-vant * razao, -vant * razao.clamp(1 - p.clip, 1 + p.clip)).mean()
                perda_valor = 0.5 * (self.modelo.valor(lote["obs"][i]) - lote["retornos"][i]).pow(2).mean()
                entropia = dist.entropy().sum(-1).mean()
                perda = perda_politica + p.coef_valor * perda_valor - p.coef_entropia * entropia
                self.otimizador.zero_grad()
                perda.backward()
                nn.utils.clip_grad_norm_(self.modelo.parameters(), p.max_norma_gradiente)
                self.otimizador.step()
                with torch.no_grad():
                    kls.append(((razao - 1) - razao_log).mean().item())
                perdas_valor.append(perda_valor.item())
                entropias.append(entropia.item())
            if kls[-1] > p.kl_alvo:
                break
        episodios = list(self.ambiente.episodios)
        self.ultima = {
            "kl": float(np.mean(kls)), "perda_valor": float(np.mean(perdas_valor)),
            "entropia": float(np.mean(entropias)), "std": float(self.modelo.log_std.exp().mean().item()),
            "retorno": float(np.mean([e[0] for e in episodios])) if episodios else math.nan,
            "voltas_episodio": float(np.mean([e[1] for e in episodios])) if episodios else math.nan,
        }

    # ------------------------------------------------------------------ avaliação
    def rede(self):
        return Rede(self.modelo.tamanhos)

    def avaliar(self):
        """Avaliação justa em cada pista de treino (sem sorteio: o carro usa a média do ator)."""
        rede = self.rede()
        genoma = self.modelo.genoma()
        camadas = rede.desempacotar(genoma)
        resultados = [avaliar(pista, lambda e: rede.avaliar(camadas, e)) for pista in self.pistas]
        voltas, melhor_volta = resumo(resultados)
        novo_recorde = voltas > self.recorde + 1e-6
        registro = {"atualizacao": self.atualizacoes, "experiencia": self.experiencia, "voltas": voltas,
                    "melhor_volta": melhor_volta, "tempo_treino": self.tempo_treino, "novo_recorde": novo_recorde,
                    **self.ultima}
        self.historico.append(registro)
        if novo_recorde:
            self.recorde = voltas
            if self.salvar:
                d = self.destino
                fitness = voltas * d.comprimento
                arquivos.salvar_cerebro(d, genoma[None, :], np.array([fitness]), self.atualizacoes, melhor_volta,
                                        self.modelo.tamanhos, metodo="ppo", experiencia=self.experiencia)
        if self.salvar:
            self._salvar_checkpoint()
        return registro


def treinar_ppo_sem_janela(pistas, milhoes_de_passos, parametros=None, dispositivo="auto", continuar=False):
    """Treina sem desenhar nada e mostra o progresso no terminal."""
    treinador = TreinadorPPO(pistas, parametros, dispositivo, continuar)
    treinador.p.passos_planejados = treinador.passos + int(milhoes_de_passos * 1_000_000)
    nomes = ", ".join(pista.nome for pista in treinador.pistas)
    print(f"PPO em {nomes} | {treinador.ambiente.n} carros em paralelo | rede {treinador.modelo.tamanhos}"
          f" | {treinador.dispositivo.type.upper()}"
          + (f" ({torch.cuda.get_device_name(treinador.dispositivo)})" if treinador.dispositivo.type == "cuda" else ""))
    print("Ctrl+C para parar — o melhor cérebro e o treino ficam salvos a cada avaliação.")
    alvo = treinador.passos + milhoes_de_passos * 1_000_000
    try:
        while treinador.passos < alvo:
            registro = treinador.iteracao()
            if registro is None:
                continue
            ritmo = treinador.passos / max(treinador.tempo_treino, 1e-9)
            print(f"Atualização {registro['atualizacao']:4d} | {numero(treinador.passos / 1e6, 2):>5} mi passos"
                  f" ({numero(treinador.experiencia / 3600, 1)} h de direção) | avaliação {numero(registro['voltas']):>5}"
                  f" voltas | melhor volta {tempo(registro['melhor_volta']):>8} | episódios: retorno"
                  f" {numero(registro['retorno'], 1):>6} | std {numero(registro['std'])} | {numero(ritmo, 0)} passos/s"
                  f"{'  << recorde salvo' if registro['novo_recorde'] else ''}")
    except KeyboardInterrupt:
        print("\nTreino interrompido.")
    return treinador

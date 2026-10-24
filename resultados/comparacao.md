# Neuroevolução × PPO

Gerado em 24/10/2026 16:57 por `python main.py comparar`.

## Generalização: voltas em 45 s em cada pista

Mesma avaliação para todos: larga parado da linha de chegada e dirige até bater ou acabar o tempo. Em **negrito**, a pista (ou pistas) em que o cérebro treinou; `x` = bateu antes do tempo.

| Cérebro | Oval | Feijão | Serpentina | Clássica | Desafio | teste01 | Média nas pistas vistas | Média nas inéditas |
|---|---|---|---|---|---|---|---|---|
| Neuroevolução · Clássica | 5,76 | 5,07 | 3,99 | **2,60** | 0,24 x | 0,25 x | 2,60 | 3,06 |
| PPO · Clássica | 5,75 | 5,14 | 4,64 | **3,06** | 3,07 | 0,25 x | 3,06 | 3,77 |
| Neuroevolução · Desafio | 4,91 | 4,49 | 3,69 | 2,46 | **2,60** | 0,26 x | 2,60 | 3,16 |
| PPO · Desafio | 5,83 | 5,29 | 4,77 | 3,04 | **3,15** | 0,25 x | 3,15 | 3,84 |
| Neuroevolução · Feijão | 6,15 | **5,62** | 0,04 x | 0,27 x | 0,00 x | 0,08 x | 5,62 | 1,31 |
| PPO · Feijão | 6,16 | **5,64** | 0,04 x | 0,42 x | 0,16 x | 0,08 x | 5,64 | 1,37 |
| Neuroevolução · Serpentina | 5,90 | 5,16 | **4,55** | 0,42 x | 0,47 x | 0,25 x | 4,55 | 2,44 |
| PPO · Serpentina | 6,00 | 5,41 | **4,95** | 0,95 x | 0,16 x | 0,26 x | 4,95 | 2,56 |
| PPO · Várias (Clássica, Desafio, Feijão, Serpentina) | 6,05 | **5,48** | **4,96** | **3,14** | **3,17** | 0,31 x | 4,19 | 3,18 |

## Curvas de aprendizado (do zero, mesmo orçamento de experiência)

### Clássica — 30 h de direção para cada técnica

![curvas](curvas_classica.png)

| | Neuroevolução | PPO |
|---|---|---|
| 1ª volta completa — experiência | 0,9 h | 3,0 h |
| 1ª volta completa — relógio | 3 s | 15 s |
| Voltas em 45 s (melhor avaliação) | 3,07 | 3,08 |
| Melhor volta | 14,35 s | 14,35 s |
| Tempo total de relógio | 59 s | 146 s |

### Desafio — 30 h de direção para cada técnica

![curvas](curvas_desafio.png)

| | Neuroevolução | PPO |
|---|---|---|
| 1ª volta completa — experiência | 0,6 h | 3,0 h |
| 1ª volta completa — relógio | 2 s | 16 s |
| Voltas em 45 s (melhor avaliação) | 3,15 | 3,20 |
| Melhor volta | 14,03 s | 13,80 s |
| Tempo total de relógio | 67 s | 148 s |

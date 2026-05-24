# Rucoy_Computer_Vision_Bot

Bot de automação para Rucoy Online usando visão computacional com YOLOv11. Roda sobre o emulador LDPlayer.

> ⚠️ **Aviso de responsabilidade:**  
> Este projeto foi desenvolvido apenas para fins educacionais e de estudo em visão computacional, automação e inteligência artificial.  
> O uso de bots, macros ou qualquer tipo de automação pode violar os Termos de Serviço do jogo Rucoy Online e resultar em punições, incluindo suspensão temporária ou banimento permanente da conta.  
> O autor não se responsabiliza pelo uso indevido do software.

## O que faz

- Detecta inimigos (vampiros/drows) via modelo YOLO customizado
- Ataca automaticamente o alvo mais próximo dentro do range
- Coleta itens e gold do chão via template matching
- Monitora barras de HP/MP e usa poções automaticamente
- Detecta quando está preso e tenta escapar
- Patrulha a área em caso de ausência de mobs
- Sistema anti-loop e anti-stuck
- Movimentação inteligente baseada em regiões caminháveis
- Thread separada para inferência YOLO sem travar o loop principal

## Stack

| Componente | Tecnologia |
|---|---|
| Detecção de inimigos | YOLOv11s (ultralytics) |
| Captura de tela | dxcam + OpenCV |
| Automação de input | pyautogui + keyboard |
| Coleta de itens | Template matching (OpenCV) |
| Rastreamento | ByteTrack |
| Emulador | LDPlayer |

## Estrutura

```bash
├── main.py              # Loop principal do bot
├── captura.py           # Captura de tela via dxcam
├── detector.py          # Wrapper do modelo YOLO
├── coletor_itens.py     # Detecção de itens por template matching
├── get_px.py            # Utilitário para capturar coordenadas/cores
├── models/
│   ├── best_vampire.pt  # Modelo treinado para vampiros
│   └── best_drow.pt     # Modelo treinado para drows
└── sprites_itens/       # Templates dos itens para coleta
```

## Requisitos

```bash
pip install -r requirements.txt
```

### Dependências principais

- Python 3.10+
- OpenCV
- Ultralytics
- dxcam
- pyautogui
- keyboard
- numpy

### Ambiente

- Windows
- LDPlayer com Rucoy Online aberto

## Uso

```bash
python main.py
```

### Hotkeys

| Tecla | Ação |
|---|---|
| F8 | Pausa / Resume |
| ESC | Encerra |
| Mouse no canto superior esquerdo | Failsafe do pyautogui |

## Funcionamento

O bot utiliza captura de tela em tempo real do LDPlayer, enviando os frames para um modelo YOLOv11 customizado treinado especificamente para detectar mobs e drops do jogo.

Após detectar um alvo válido:

1. Calcula distância até o mob
2. Verifica se o alvo está em área caminhável
3. Move o personagem automaticamente
4. Usa ataques ou skills dependendo da distância
5. Monitora HP/MP constantemente
6. Coleta drops automaticamente após a morte do mob

Além disso, o sistema possui:

- Detecção de travamento
- Escape automático de obstáculos
- Anti-loop de movimentação
- Blacklist temporária de mobs inalcançáveis
- Cooldowns humanizados
- Randomização de cliques

## Treinamento

Os modelos foram treinados com dataset próprio usando YOLOv11s.

### Scripts auxiliares

- `coletar_frames.py` — captura frames para dataset
- `auto_label.py` — auxilia na rotulagem
- `dividir_dataset.py` — split treino/validação
- `treinar.py` — fine-tuning do modelo
- `dedup_labels.py` — remove labels duplicadas

## Objetivo

O projeto foi criado como estudo prático de:

- Visão computacional
- Inferência em tempo real
- Automação de interface
- Machine Learning aplicado
- Detecção de objetos
- Sistemas autônomos baseados em imagem

## Observações

- O projeto ainda está em desenvolvimento
- Algumas funcionalidades experimentais podem estar desativadas
- O sistema de coleta via template matching ainda está em validação
- O desempenho depende da qualidade do modelo treinado e da resolução utilizada

## Licença

Projeto para fins educacionais e de pesquisa.
Use por sua conta e risco.

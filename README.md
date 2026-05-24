# Rucoy_Computer_Vision_Bot

Bot de automação para Rucoy Online usando visão computacional com YOLOv11. Roda sobre o emulador Android LDPlayer.

## O que faz

- Detecta inimigos (vampiros/drows) via modelo YOLO customizado
- Ataca automaticamente o alvo mais próximo dentro do range
- Coleta itens e gold do chão via template matching
- Monitora barras de HP/MP e usa poções automaticamente
- Detecta quando está preso e tenta escapar
- Patrulha a área em caso de ausência de mobs

## Stack

| Componente | Tecnologia |
|---|---|
| Detecção de inimigos | YOLOv11s (ultralytics) |
| Captura de tela | dxcam + OpenCV |
| Automação de input | pyautogui + keyboard |
| Coleta de itens | Template matching (OpenCV) |
| Emulador | LDPlayer |

## Estrutura

```
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

- Python 3.10+
- LDPlayer com Rucoy Online aberto
- Windows (uso de dxcam e pywin32)

## Uso

```bash
python main.py
```

`F8` — pausa/resume o bot  
`ESC` — encerra (pyautogui failsafe: mova o mouse pro canto superior esquerdo)

## Treinamento

Os modelos foram treinados com dataset próprio usando YOLOv11s. Scripts auxiliares:

- `coletar_frames.py` — captura frames para dataset
- `auto_label.py` — auxilia na rotulagem
- `dividir_dataset.py` — split treino/validação
- `treinar.py` — fine-tuning do modelo
- `dedup_labels.py` — remove labels duplicadas

import os
import time
import keyboard
import cv2 as cv
from captura import Captura

INTERVALO = 1.5
PASTA = "dataset/images/raw"

os.makedirs(PASTA, exist_ok=True)

cap = Captura("LDPlayer")

print("Capturando frames a cada", INTERVALO, "segundos.")
print("Pressione F3 para parar.")

rodando = True
keyboard.add_hotkey("F3", lambda: globals().update(rodando=False))

total = len(os.listdir(PASTA))

while rodando:

    frame = cap.get_frame()

    if frame is None:
        time.sleep(0.1)
        continue

    nome = os.path.join(PASTA, f"frame_{total:05d}.jpg")
    cv.imwrite(nome, frame, [cv.IMWRITE_JPEG_QUALITY, 95])
    total += 1

    print(f"Salvo: {nome}")

    time.sleep(INTERVALO)

print(f"Coleta encerrada. Total: {total} frames em '{PASTA}'")

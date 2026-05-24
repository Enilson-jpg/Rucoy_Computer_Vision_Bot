import cv2 as cv
import os
import time
from captura import Captura

os.makedirs("dataset/images", exist_ok=True)

cap = Captura("LDPlayer")
contador = 0

print("Coletando imagens a cada 2 segundos...")
print("Ctrl+C para parar.")

while True:
    frame = cap.get_frame()

    if frame is not None:
        # Converte BGR para RGB
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        nome = f"dataset/images/frame_{contador:04d}.png"
        cv.imwrite(nome, frame)
        print(f"Salvo: {nome}")
        contador += 1

        time.sleep(2)
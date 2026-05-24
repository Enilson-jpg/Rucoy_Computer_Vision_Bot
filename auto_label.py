from ultralytics import YOLO
import os

MODEL_PATH = "models/best_vampire.pt"
RAW_DIR = "dataset/images/vampires"
CONF = 0.5

model = YOLO(MODEL_PATH)

todas_imgs = [f for f in os.listdir(RAW_DIR) if f.endswith((".jpg", ".png"))]
print(f"Total de imagens: {len(todas_imgs)}")

gerados = 0
vazios = 0

for img_name in todas_imgs:
    img_path = os.path.join(RAW_DIR, img_name)
    results = model(img_path, conf=CONF, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        vazios += 1
        continue

    label_path = os.path.join(RAW_DIR, img_name.rsplit(".", 1)[0] + ".txt")
    with open(label_path, "w") as f:
        for box in boxes:
            cls = int(box.cls)
            cx, cy, w, h = box.xywhn[0].tolist()
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    gerados += 1

print(f"Labels gerados : {gerados}")
print(f"Sem detecção   : {vazios}")

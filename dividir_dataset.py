import os
import shutil
import random

IMG_DIR          = "dataset/images/vampires"
LBL_DIR_SEPARADO = "dataset/labels/vampires"
CLASSES_TXT      = os.path.join(IMG_DIR, "classes.txt")
DATA_YAML        = "dataset/data.yaml"

TRAIN_IMG = "dataset/images/train"
TRAIN_LBL = "dataset/labels/train"
VAL_IMG   = "dataset/images/val"
VAL_LBL   = "dataset/labels/val"

VAL_RATIO = 0.15

# lê classes do raw
with open(CLASSES_TXT) as f:
    classes = [l.strip() for l in f if l.strip()]

# atualiza data.yaml
names_str = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
yaml = f"""path: ./dataset
train: images/train
val: images/val

nc: {len(classes)}
names:
{names_str}
"""
with open(DATA_YAML, "w") as f:
    f.write(yaml)
print(f"data.yaml atualizado: {len(classes)} classes -> {classes}")

def achar_label(stem):
    junto = os.path.join(IMG_DIR, stem + ".txt")
    if os.path.exists(junto):
        return junto
    separado = os.path.join(LBL_DIR_SEPARADO, stem + ".txt")
    if os.path.exists(separado):
        return separado
    return None

imagens = [f for f in os.listdir(IMG_DIR) if f.endswith((".jpg", ".png"))]

rotuladas = []
for f in imagens:
    stem = f.rsplit(".", 1)[0]
    lbl = achar_label(stem)
    if lbl:
        rotuladas.append((f, lbl))

print(f"Total rotuladas: {len(rotuladas)} / {len(imagens)}")

if not rotuladas:
    print("Nenhuma imagem rotulada encontrada.")
    exit(1)

random.shuffle(rotuladas)

corte  = int(len(rotuladas) * (1 - VAL_RATIO))
treino = rotuladas[:corte]
val    = rotuladas[corte:]

def copiar(lista, img_dst, lbl_dst):
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)
    for img_f, lbl_f in lista:
        stem = img_f.rsplit(".", 1)[0]
        shutil.copy(os.path.join(IMG_DIR, img_f), os.path.join(img_dst, img_f))
        shutil.copy(lbl_f, os.path.join(lbl_dst, stem + ".txt"))

copiar(treino, TRAIN_IMG, TRAIN_LBL)
copiar(val,    VAL_IMG,   VAL_LBL)

print(f"Treino: {len(treino)} | Val: {len(val)}")
print("Dataset dividido com sucesso!")

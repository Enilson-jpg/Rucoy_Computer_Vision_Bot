import os

RAW_DIR = "dataset/images/vampires"
IOU_THRESHOLD = 0.4

def compute_iou(a, b):
    ax1, ay1 = a[1] - a[3]/2, a[2] - a[4]/2
    ax2, ay2 = a[1] + a[3]/2, a[2] + a[4]/2
    bx1, by1 = b[1] - b[3]/2, b[2] - b[4]/2
    bx2, by2 = b[1] + b[3]/2, b[2] + b[4]/2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a[3]*a[4] + b[3]*b[4] - inter
    return inter / union if union > 0 else 0

total_removidos = 0
arquivos = 0

for fname in os.listdir(RAW_DIR):
    if not fname.endswith(".txt") or fname == "classes.txt":
        continue

    path = os.path.join(RAW_DIR, fname)
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    boxes = []
    for line in lines:
        parts = line.split()
        boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))

    kept = []
    for box in boxes:
        duplicate = False
        for k in kept:
            if k[0] == box[0] and compute_iou(box, k) > IOU_THRESHOLD:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)

    removidos = len(boxes) - len(kept)
    if removidos > 0:
        with open(path, "w") as f:
            for b in kept:
                f.write(f"{b[0]} {b[1]} {b[2]} {b[3]} {b[4]}\n")
        total_removidos += removidos
        arquivos += 1

print(f"Duplicatas removidas: {total_removidos} em {arquivos} arquivos")

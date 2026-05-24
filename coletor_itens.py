import os
import cv2
import numpy as np


class ColetorItens:
    def __init__(self, pasta="sprites_itens/itens", threshold=0.72):
        self.threshold = threshold
        self.sprites = []
        self._carregar_pasta(pasta)

    def _carregar_pasta(self, pasta):
        if not os.path.isdir(pasta):
            print(f"Pasta não encontrada: {pasta}")
            return

        for nome in os.listdir(pasta):
            if not nome.lower().endswith(".png"):
                continue

            caminho = os.path.join(pasta, nome)
            img = cv2.imread(caminho, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Falha ao carregar: {caminho}")
                continue

            if img.ndim == 3 and img.shape[2] == 4:
                mask = img[:, :, 3]
                template = img[:, :, :3]
            else:
                mask = None
                template = img

            h, w = template.shape[:2]
            self.sprites.append({
                "nome": os.path.splitext(nome)[0],
                "template": template,
                "template_gray": cv2.cvtColor(template, cv2.COLOR_BGR2GRAY),
                "mask": mask,
                "w": w,
                "h": h,
            })
            print(f"Item carregado: {nome} ({w}x{h}px)")

        print(f"{len(self.sprites)} sprite(s) de item carregado(s)")

    def detectar_todos(self, frame):
        resultados = []
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = gray_frame.shape[:2]

        for spr in self.sprites:
            if spr["h"] > fh or spr["w"] > fw:
                continue

            tmpl_gray = spr["template_gray"]

            if spr["mask"] is not None:
                res = cv2.matchTemplate(
                    gray_frame, tmpl_gray,
                    cv2.TM_CCOEFF_NORMED,
                    mask=spr["mask"]
                )
            else:
                res = cv2.matchTemplate(
                    gray_frame, tmpl_gray,
                    cv2.TM_CCOEFF_NORMED
                )

            # coleta todas as ocorrências acima do threshold
            locs = np.where(res >= self.threshold)
            for pt in zip(*locs[::-1]):
                cx = pt[0] + spr["w"] // 2
                cy = pt[1] + spr["h"] // 2
                score = float(res[pt[1], pt[0]])
                resultados.append({
                    "nome": spr["nome"],
                    "cx": cx,
                    "cy": cy,
                    "score": score,
                })

        # dedup por proximidade (raio 20px) — mantém o de maior score
        resultados.sort(key=lambda r: r["score"], reverse=True)
        dedup = []
        for r in resultados:
            muito_perto = any(
                ((r["cx"] - d["cx"])**2 + (r["cy"] - d["cy"])**2) ** 0.5 < 20
                for d in dedup
            )
            if not muito_perto:
                dedup.append(r)

        return dedup

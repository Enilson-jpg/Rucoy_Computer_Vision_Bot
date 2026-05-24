from ultralytics import YOLO
import cv2 as cv

class Detector:
    def __init__(self, model_path="models/best.pt", confidence=0.5):
        self.model = YOLO(model_path)
        self.confidence = confidence
        print(f"Modelo carregado: {model_path}")

    def detectar(self, frame):
        resultados = self.model(frame, conf=self.confidence, verbose=False)[0]
        deteccoes = []

        for box in resultados.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            classe = int(box.cls[0])
            nome = self.model.names[classe]

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            deteccoes.append({
                "classe": nome,
                "confianca": conf,
                "cx": cx,
                "cy": cy,
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
            })

        return deteccoes


# Teste
if __name__ == "__main__":
    from captura import Captura

    cap = Captura("LDPlayer")
    det = Detector()

    frame = cap.get_frame()
    if frame is not None:
        cv.imwrite("teste_captura.png", frame)
        deteccoes = det.detectar(frame)
        print(f"Detectados: {len(deteccoes)}")
        for d in deteccoes:
            print(f"  {d['classe']} | confiança={d['confianca']:.2f} | pos=({d['cx']},{d['cy']})")
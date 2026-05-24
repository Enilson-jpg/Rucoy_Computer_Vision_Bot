from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11s.pt")

    model.train(
        device="cpu",
        data="dataset/data.yaml",

        # treino
        epochs=60,
        imgsz=640,
        batch=16,

        # projeto
        name="vampiro_detector_640",

        # otimização
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=3,

        # estabilidade
        patience=30,
        cache=True,
        amp=False,

        # augmentations
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,

        translate=0.08,
        scale=0.2,

        fliplr=0.5,
        flipud=0.0,
        degrees=0.0,

        mosaic=1.0,
        close_mosaic=10,
        mixup=0.1,

        # performance
        workers=0,
        pretrained=True,

        # salvar melhor modelo
        save=True,
        save_period=10,
    )

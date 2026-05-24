import os
import time
import random
import logging
import logging.handlers
import threading
import queue as _queue
from collections import deque
import cv2 as cv
import numpy as np
import keyboard
import pyautogui

from captura import Captura
from detector import Detector
from coletor_itens import ColetorItens
12
# =========================
# CONFIG
# =========================

pyautogui.FAILSAFE = True
pyautogui.PAUSE    = 0.0
os.chdir(os.path.dirname(os.path.abspath(__file__)))

_log_fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
_file_handler = logging.handlers.RotatingFileHandler(
    "bot.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
_file_handler.setFormatter(_log_fmt)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_fmt)
logging.basicConfig(level=logging.INFO, handlers=[_file_handler, _console_handler])

TARGET_FPS        = 30
FRAME_TIME        = 1 / TARGET_FPS
DETECTION_INTERVAL = 3

RANGE         = 200
PLAYER_RADIUS = 100
PLAYER_X      = 800
PLAYER_Y      = 500

ATTACK_COOLDOWN   = 1.5
SKILL_CD          = 2.0
WALK_COOLDOWN     = 0.5
MOB_WALK_COOLDOWN = 0.5
LOCK_BONUS        = 150

CLICK_MIN_DIST_CENTRO = 80

# Regiões de UI onde o bot nunca pode clicar (coordenadas do frame capturado)
ZONAS_PROIBIDAS = [
    (0,    0,    1601, 133),   # menu superior
    (0,    0,    155,  928),   # painel esquerdo
    (1492, 608,  1615, 932),   # painel inferior direito
]

CORES_CAMINHAVEIS = [
    ((96, 80, 35), (114, 120, 105)),
]

CORES_NAO_CAMINHAVEIS = [
    ((10, 116, 54),  (30, 176, 114)),   # HSV ~(20,146, 84)
    (( 7, 225, 124), (27, 255, 184)),   # HSV ~(17,255,154)
    ((92,  32, 134), (112, 92, 194)),   # HSV ~(102,62,164)
    ((10, 225, 190), (30, 255, 250)),   # HSV ~(20,255,220)
]

MONEY_WALK_COOLDOWN  = 1.0
MONEY_MAX_DIST       = 600
MONEY_MAX_TENTATIVAS = 5

ITEM_WALK_COOLDOWN   = 1.0
ITEM_MAX_DIST        = 600
ITEM_MAX_TENTATIVAS  = 5
ITEM_DETECTION_INTERVAL = 5

MAO_ROI_X       = 1495
MAO_ROI_Y       = 188
MAO_ROI_W       = 103
MAO_ROI_H       = 105
MAO_HSV_LOW     = (8,  160, 220)
MAO_HSV_HIGH    = (22, 215, 255)
MAO2_HSV_LOW    = (0,   0, 120)
MAO2_HSV_HIGH   = (180, 30, 170)
MAO_AREA_MIN    = 120
MAO_KEY         = "space"
MAO_MAX_CLIQUES = 2
MAO_COOLDOWN    = 0.8
MAO_CONFIRM_FRAMES = 1

STUCK_CD             = 4.0
STUCK_DIFF_LIMIAR    = 8.0
STUCK_CHECK_INTERVAL = 3.0
STUCK_COUNT_ESCAPE   = 4
STUCK_ESCAPE_PASSOS  = 5
STUCK_ESCAPE_DIST_MIN = 220
STUCK_ESCAPE_DIST_MAX = 420
STUCK_ROI_X1 = 350;  STUCK_ROI_Y1 = 180
STUCK_ROI_X2 = 1250; STUCK_ROI_Y2 = 720

DIST_STUCK_JANELA    = 12
DIST_STUCK_DELTA     = 20
DIST_STUCK_PASSOS    = 3
DIST_STUCK_PASSO_DIST = 200

SEM_MOB_TIMEOUT  = 15.0
FORA_RANGE_MAX   = 20
MOB_MAX_TENTATIVAS = 8

MORTE_CHECK_DIST    = 80    # px — distância para considerar "chegou ao drop spot"
MORTE_CHECK_TIMEOUT = 6.0   # s  — desiste de ir ao drop spot após esse tempo
MORTE_RANGE_FACTOR  = 1.3   # mob deve ter estado a <= RANGE*fator quando sumiu

HP_BAR_X  = 8;   HP_BAR_Y = 40;  HP_BAR_W = 487; HP_BAR_H = 37
HP_MIN_PCT = 0.50; POCAO_HP_CD = 1.0; POCAO_HP_KEY = "1"

MP_BAR_X  = 8;   MP_BAR_Y = 85;  MP_BAR_W = 483; MP_BAR_H = 10
MP_MIN_PCT = 0.50; POCAO_MP_CD = 3.0; POCAO_MP_KEY = "2"
MP_SKILL_THRESHOLD = 0.20

SKILL_KEY = "3"

POPUP_X = 858; POPUP_Y = 304; POPUP_W = 586; POPUP_H = 85
POPUP_ESC_CD = 3.0

DIRECOES = [(1, 0), (0, 1), (-1, 0), (0, -1)]
PATROL_DIST_MIN = 250
PATROL_DIST_MAX = 420

PATROL_HISTORICO_MAX  = 15
PATROL_LOOP_RAIO      = 300
PATROL_DIR_COOLDOWN   = 5.0

direcao_atual = 0

# =========================
# INIT
# =========================

cap     = Captura("LDPlayer")
det     = Detector(model_path="models/best_vampire.pt", confidence=0.6)
coletor = ColetorItens(threshold=0.90)

# --- thread de inferência YOLO (não bloqueia o loop principal) ---
_det_queue  = _queue.Queue(maxsize=1)
_det_result = []
_det_lock   = threading.Lock()
_det_clahe  = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_det_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

def _det_worker():
    while True:
        frame = _det_queue.get()
        try:
            lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
            l, a, b = cv.split(lab)
            fp = cv.merge((_det_clahe.apply(l), a, b))
            fs = cv.filter2D(cv.cvtColor(fp, cv.COLOR_LAB2BGR), -1, _det_kernel)
            r = det.model.track(
                fs, conf=det.confidence, persist=True,
                tracker="bytetrack.yaml", imgsz=640, augment=False,
                iou=0.4, agnostic_nms=True, verbose=False
            )[0]
            dets = []
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    dets.append({
                        "classe":    det.model.names[int(box.cls[0])],
                        "confianca": float(box.conf[0]),
                        "cx": (x1 + x2) // 2,
                        "cy": (y1 + y2) // 2,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "id": int(box.id[0]) if box.id is not None else None,
                    })
            with _det_lock:
                _det_result[:] = dets
        except Exception as e:
            logging.error(f"Erro YOLO thread: {e}")

threading.Thread(target=_det_worker, daemon=True).start()

rodando = True
mp_atual = 1.0

keyboard.add_hotkey("F3", lambda: globals().update(rodando=False))

# =========================
# HELPERS
# =========================

def _jitter(x, y, raio=7):
    return (
        int(x + random.gauss(0, raio / 2)),
        int(y + random.gauss(0, raio / 2)),
    )

def click_humano(x, y, jitter=7):
    jx, jy = _jitter(x, y, jitter)
    pyautogui.click(jx, jy, duration=random.uniform(0.02, 0.05))
    time.sleep(random.uniform(0.02, 0.04))

def tecla_humana(key):
    time.sleep(random.uniform(0.02, 0.05))
    keyboard.send(key)
    time.sleep(random.uniform(0.02, 0.04))

# =========================
# DETECÇÃO DE ESTADO
# =========================

_hp_max_pixels = 1

def ler_hp_pct(frame):
    global _hp_max_pixels
    roi = frame[HP_BAR_Y:HP_BAR_Y + HP_BAR_H, HP_BAR_X:HP_BAR_X + HP_BAR_W]
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    m1 = cv.inRange(hsv, (0, 80, 80), (10, 255, 255))
    m2 = cv.inRange(hsv, (170, 80, 80), (180, 255, 255))
    pixels = cv.countNonZero(cv.bitwise_or(m1, m2))
    if pixels > _hp_max_pixels:
        _hp_max_pixels = pixels
    return pixels / _hp_max_pixels

def ler_mp_pct(frame):
    roi = frame[MP_BAR_Y:MP_BAR_Y + MP_BAR_H, MP_BAR_X:MP_BAR_X + MP_BAR_W]
    mask = cv.inRange(cv.cvtColor(roi, cv.COLOR_BGR2HSV), (95, 80, 80), (125, 255, 255))
    cols = np.any(mask > 0, axis=0)
    if not np.any(cols):
        return 1.0
    return min((int(np.where(cols)[0][-1]) + 1) / MP_BAR_W, 1.0)

def detectar_popup(frame):
    x2 = min(POPUP_X + POPUP_W, frame.shape[1])
    y2 = min(POPUP_Y + POPUP_H, frame.shape[0])
    roi = frame[POPUP_Y:y2, POPUP_X:x2]
    if roi.size == 0:
        return False
    mask = cv.inRange(cv.cvtColor(roi, cv.COLOR_BGR2HSV), (65, 200, 160), (76, 255, 210))
    return any(cv.contourArea(c) > 10000
               for c in cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0])

def detectar_mao(frame):
    x2 = min(MAO_ROI_X + MAO_ROI_W, frame.shape[1])
    y2 = min(MAO_ROI_Y + MAO_ROI_H, frame.shape[0])
    roi = frame[MAO_ROI_Y:y2, MAO_ROI_X:x2]
    if roi.size == 0:
        return None
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    mask = cv.bitwise_or(
        cv.inRange(hsv_roi, MAO_HSV_LOW,  MAO_HSV_HIGH),
        cv.inRange(hsv_roi, MAO2_HSV_LOW, MAO2_HSV_HIGH),
    )
    melhor = None
    melhor_area = 0
    for c in cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]:
        area = cv.contourArea(c)
        if area >= MAO_AREA_MIN and area > melhor_area:
            M = cv.moments(c)
            if M["m00"] > 0:
                melhor_area = area
                melhor = {
                    "cx": int(M["m10"] / M["m00"]) + MAO_ROI_X,
                    "cy": int(M["m01"] / M["m00"]) + MAO_ROI_Y,
                }
    return melhor

def usar_skill(x, y):
    if mp_atual < MP_SKILL_THRESHOLD:
        return False
    tecla_humana(SKILL_KEY)
    time.sleep(random.uniform(0.10, 0.20))
    click_humano(x, y)
    return True

def score_alvo(alvo, cx, cy, _lock_id=None):
    return ((alvo["cx"] - cx)**2 + (alvo["cy"] - cy)**2) ** 0.5

def pixel_valido(frame, screen_x, screen_y, centro_x, centro_y, raio=10):
    fx = screen_x - cap.win_x
    fy = screen_y - cap.win_y
    h, w = frame.shape[:2]
    if fx < 0 or fy < 0 or fx >= w or fy >= h:
        return False
    if ((fx - centro_x)**2 + (fy - centro_y)**2) ** 0.5 < CLICK_MIN_DIST_CENTRO:
        return False
    for zx1, zy1, zx2, zy2 in ZONAS_PROIBIDAS:
        if zx1 <= fx <= zx2 and zy1 <= fy <= zy2:
            return False

    # amostra janela raio×raio ao redor do alvo (passo 3 → ~49 pontos)
    ys = np.clip(np.arange(fy - raio, fy + raio + 1, 3), 0, h - 1)
    xs = np.clip(np.arange(fx - raio, fx + raio + 1, 3), 0, w - 1)
    patch = frame[np.ix_(ys, xs)]
    hsv   = cv.cvtColor(patch, cv.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.int16)
    S = hsv[:, :, 1].astype(np.int16)
    V = hsv[:, :, 2].astype(np.int16)

    nao_preto = V > 77

    nao_parede = np.ones(H.shape, dtype=bool)
    for lo, hi in CORES_NAO_CAMINHAVEIS:
        nao_parede &= ~((H >= lo[0]) & (H <= hi[0]) &
                        (S >= lo[1]) & (S <= hi[1]) &
                        (V >= lo[2]) & (V <= hi[2]))

    caminhavel = np.zeros(H.shape, dtype=bool)
    for lo, hi in CORES_CAMINHAVEIS:
        caminhavel |= ((H >= lo[0]) & (H <= hi[0]) &
                       (S >= lo[1]) & (S <= hi[1]) &
                       (V >= lo[2]) & (V <= hi[2]))

    valido = nao_preto & nao_parede & caminhavel
    return valido.sum() >= valido.size * 0.5

def detectar_stuck(f1, f2):
    if f1 is None or f2 is None:
        return False
    h, w = f1.shape[:2]
    y1, y2 = min(STUCK_ROI_Y1, h), min(STUCK_ROI_Y2, h)
    x1, x2 = min(STUCK_ROI_X1, w), min(STUCK_ROI_X2, w)
    r1, r2 = f1[y1:y2, x1:x2], f2[y1:y2, x1:x2]
    if r1.size == 0 or r1.shape != r2.shape:
        return False
    return cv.mean(cv.absdiff(r1, r2))[0] < STUCK_DIFF_LIMIAR

_ultimo_angulo_escape = None

def escapar_stuck(frame, centro_x, centro_y, img_w, img_h):
    global _ultimo_angulo_escape
    logging.warning("STUCK — escapando")
    angulo_base = (_ultimo_angulo_escape + np.pi
                   if _ultimo_angulo_escape is not None
                   else np.random.uniform(0, 2 * np.pi))
    clicou = 0
    for i in range(STUCK_ESCAPE_PASSOS * 3):
        ang  = angulo_base + (2 * np.pi * i / STUCK_ESCAPE_PASSOS) + np.random.uniform(-0.25, 0.25)
        dist = np.random.randint(STUCK_ESCAPE_DIST_MIN, STUCK_ESCAPE_DIST_MAX)
        ex = max(cap.win_x + 120, min(cap.win_x + centro_x + int(np.cos(ang) * dist), cap.win_x + img_w - 120))
        ey = max(cap.win_y + 120, min(cap.win_y + centro_y + int(np.sin(ang) * dist), cap.win_y + img_h - 120))
        if not pixel_valido(frame, ex, ey, centro_x, centro_y):
            continue
        click_humano(ex, ey, jitter=15)
        time.sleep(random.uniform(0.5, 0.8))
        if clicou % 2 == 0 and mp_atual >= MP_SKILL_THRESHOLD:
            usar_skill(cap.win_x + centro_x, cap.win_y + centro_y)
            time.sleep(random.uniform(0.3, 0.5))
        clicou += 1
        if clicou >= STUCK_ESCAPE_PASSOS:
            break
    _ultimo_angulo_escape = angulo_base

# =========================
# MAIN
# =========================

def main():
    global rodando, direcao_atual, mp_atual

    logging.info("Bot iniciando...")
    for i in range(3, 0, -1):
        logging.info(f"{i}..."); time.sleep(1)
    logging.info("Bot iniciado!")

    ultimo_ataque      = 0
    ultimo_skill       = 0
    ultimo_walk        = 0
    ultimo_money_click = 0
    ultimo_item_click  = 0
    ultimo_pocao_hp    = 0
    ultimo_pocao_mp    = 0
    ultimo_popup_esc   = 0
    ultimo_stuck       = 0
    ultimo_stuck_check = 0
    ultimo_log_bars    = 0
    ultimo_mob_visto   = time.time()
    mao_ultimo_click   = 0

    stuck_count      = 0
    frame_stuck_ref  = None
    fora_range_count = 0
    alvo_lock_id     = None
    deteccoes_cache  = []
    mao_confirm      = 0

    mob_tentativas  = {}
    mobs_bloqueados = set()
    money_tentativas = {}
    money_bloqueados = set()
    item_tentativas  = {}
    item_bloqueados  = set()
    itens_cache      = []

    ultimo_alvo_pos  = None
    ultimo_alvo_dist = float('inf')
    pos_morte        = None
    morte_walk_time  = 0.0

    dist_hist_id    = None
    dist_hist       = deque(maxlen=DIST_STUCK_JANELA)
    tinha_alvos_antes = False
    historico_clicks  = deque(maxlen=PATROL_HISTORICO_MAX)
    ultimo_dir_change = 0.0

    while rodando:
        inicio_loop = time.time()

        frame = cap.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        agora = time.time()

        # guard de memória
        if len(mob_tentativas) > 300:    mob_tentativas.clear()
        if len(mobs_bloqueados) > 300:   mobs_bloqueados.clear()
        if len(money_tentativas) > 300:  money_tentativas.clear()
        if len(money_bloqueados) > 300:  money_bloqueados.clear()
        if len(item_tentativas)  > 300:  item_tentativas.clear()
        if len(item_bloqueados)  > 300:  item_bloqueados.clear()

        img_h, img_w = frame.shape[:2]
        centro_x, centro_y = PLAYER_X, PLAYER_Y
        alvos = []

        # ---------- STUCK CHECK (só patrol — usa deteccoes_cache, alvos ainda não foi populado) ----------
        tem_mob = any(d["classe"] == "mob" for d in deteccoes_cache)
        if agora - ultimo_stuck_check >= STUCK_CHECK_INTERVAL:
            if frame_stuck_ref is not None and not tem_mob:
                if detectar_stuck(frame_stuck_ref, frame):
                    stuck_count += 1
                    logging.warning(f"Stuck ({stuck_count}/{STUCK_COUNT_ESCAPE})")
                else:
                    stuck_count = max(0, stuck_count - 1)
            elif tem_mob:
                stuck_count = 0
            frame_stuck_ref    = frame.copy()
            ultimo_stuck_check = agora

        if stuck_count >= STUCK_COUNT_ESCAPE and agora - ultimo_stuck >= STUCK_CD:
            escapar_stuck(frame, centro_x, centro_y, img_w, img_h)
            ultimo_stuck    = agora
            stuck_count     = 0
            frame_stuck_ref = None
            mobs_bloqueados.clear(); mob_tentativas.clear(); dist_hist.clear()
            alvo_lock_id    = None

        # ---------- POPUP ----------
        if detectar_popup(frame) and agora - ultimo_popup_esc >= POPUP_ESC_CD:
            logging.info("Popup detectado")
            tecla_humana("esc")
            ultimo_popup_esc = agora

        # ---------- HP / MP ----------
        hp = ler_hp_pct(frame)
        mp = ler_mp_pct(frame)
        mp_atual = mp

        if agora - ultimo_log_bars >= 5.0:
            logging.info(f"HP={hp:.2%} MP={mp:.2%}")
            ultimo_log_bars = agora

        if hp < HP_MIN_PCT and agora - ultimo_pocao_hp >= POCAO_HP_CD:
            tecla_humana(POCAO_HP_KEY)
            ultimo_pocao_hp = agora
            logging.info(f"Poção HP ({hp:.0%})")

        if mp < MP_MIN_PCT and agora - ultimo_pocao_mp >= POCAO_MP_CD:
            tecla_humana(POCAO_MP_KEY)
            ultimo_pocao_mp = agora
            logging.info(f"Poção MP ({mp:.0%})")

        # ---------- DETECÇÃO YOLO (thread) ----------
        try:
            _det_queue.put_nowait(frame.copy())
        except _queue.Full:
            pass
        with _det_lock:
            deteccoes_cache = list(_det_result)

        # ---------- ITENS (template matching) — desativado até validar templates ----------
        # if frame_count % ITEM_DETECTION_INTERVAL == 0:
        #     try:
        #         itens_cache = coletor.detectar_todos(frame)
        #     except Exception as e:
        #         logging.error(f"Erro coletor_itens: {e}")

        # ---------- FILTRO ----------
        alvos     = []
        dinheiros = []
        for d in deteccoes_cache:
            if d["classe"] == "money":
                if d["id"] not in money_bloqueados:
                    dist_m = ((d["cx"] - centro_x)**2 + (d["cy"] - centro_y)**2) ** 0.5
                    if dist_m <= MONEY_MAX_DIST:
                        dinheiros.append(d)
                continue
            if d["classe"] != "mob" or d["id"] in mobs_bloqueados:
                continue
            if d["confianca"] < 0.4:
                continue
            if (d["x2"] - d["x1"]) < 20 or (d["y2"] - d["y1"]) < 20:
                continue
            alvos.append(d)

        # mob morreu → limpa blacklist; registra drop spot se estávamos em range
        if tinha_alvos_antes and not alvos:
            if ultimo_alvo_pos is not None and ultimo_alvo_dist <= RANGE * MORTE_RANGE_FACTOR:
                pos_morte      = ultimo_alvo_pos
                morte_walk_time = agora
                logging.info(f"Mob morreu pos={pos_morte} dist={ultimo_alvo_dist:.0f} → indo checar drop")
            ultimo_alvo_pos  = None
            ultimo_alvo_dist = float('inf')
            mobs_bloqueados.clear()
            mob_tentativas.clear()
            dist_hist.clear()
            alvo_lock_id = None
        tinha_alvos_antes = bool(alvos)

        # ---------- MÃO (prioridade máxima) ----------
        mao = detectar_mao(frame)
        if mao:
            mao_confirm += 1
        else:
            mao_confirm = 0

        if mao_confirm >= MAO_CONFIRM_FRAMES and agora - mao_ultimo_click >= MAO_COOLDOWN:
            logging.info("Mão detectada — coletando")
            for _ in range(MAO_MAX_CLIQUES):
                tecla_humana(MAO_KEY)
                time.sleep(random.uniform(0.03, 0.06))
            mao_ultimo_click = agora
            mao_confirm = 0
            continue

        # ---------- MONEY ----------
        if dinheiros and agora - ultimo_money_click >= MONEY_WALK_COOLDOWN and mao is None:
            mais_perto = min(dinheiros,
                             key=lambda d: ((d["cx"] - centro_x)**2 + (d["cy"] - centro_y)**2))
            mid = mais_perto["id"]
            if mid is not None:
                money_tentativas[mid] = money_tentativas.get(mid, 0) + 1
                if money_tentativas[mid] >= MONEY_MAX_TENTATIVAS:
                    money_bloqueados.add(mid)
                    logging.warning(f"Money ID={mid} bloqueado")
                    ultimo_money_click = 0
                else:
                    click_humano(cap.win_x + mais_perto["cx"], cap.win_y + mais_perto["cy"], jitter=8)
                    ultimo_money_click = agora
                    dist_m = ((mais_perto["cx"]-centro_x)**2 + (mais_perto["cy"]-centro_y)**2) ** 0.5
                    logging.info(f"Money ID={mid} dist={dist_m:.0f}")
            else:
                click_humano(cap.win_x + mais_perto["cx"], cap.win_y + mais_perto["cy"], jitter=8)
                ultimo_money_click = agora

        # ---------- ITENS ----------
        itens_visiveis = [
            it for it in itens_cache
            if it["nome"] not in item_bloqueados
            and ((it["cx"] - centro_x)**2 + (it["cy"] - centro_y)**2) ** 0.5 <= ITEM_MAX_DIST
            and not any(zx1 <= it["cx"] <= zx2 and zy1 <= it["cy"] <= zy2
                        for zx1, zy1, zx2, zy2 in ZONAS_PROIBIDAS)
        ]
        if itens_visiveis and agora - ultimo_item_click >= ITEM_WALK_COOLDOWN and mao is None:
            mais_perto_item = min(itens_visiveis,
                                  key=lambda it: (it["cx"] - centro_x)**2 + (it["cy"] - centro_y)**2)
            nome_item = mais_perto_item["nome"]
            item_tentativas[nome_item] = item_tentativas.get(nome_item, 0) + 1
            if item_tentativas[nome_item] >= ITEM_MAX_TENTATIVAS:
                item_bloqueados.add(nome_item)
                logging.warning(f"Item '{nome_item}' bloqueado após {ITEM_MAX_TENTATIVAS} tentativas")
                ultimo_item_click = 0
            else:
                ix = cap.win_x + mais_perto_item["cx"]
                iy = cap.win_y + mais_perto_item["cy"]
                if not pixel_valido(frame, ix, iy, centro_x, centro_y):
                    logging.debug(f"Item '{nome_item}' em posição inválida (UI/preto), ignorando")
                    item_bloqueados.add(nome_item)
                else:
                    click_humano(ix, iy, jitter=8)
                    ultimo_item_click = agora
                    dist_it = ((mais_perto_item["cx"]-centro_x)**2 + (mais_perto_item["cy"]-centro_y)**2) ** 0.5
                    logging.info(f"Item '{nome_item}' tentativa={item_tentativas[nome_item]} dist={dist_it:.0f}")

        # ---------- COMBATE ----------
        indo_ao_money = agora - ultimo_money_click < MONEY_WALK_COOLDOWN
        indo_ao_item  = agora - ultimo_item_click  < ITEM_WALK_COOLDOWN

        if alvos and not indo_ao_money and not indo_ao_item:
            alvo = min(alvos, key=lambda d: score_alvo(d, centro_x, centro_y))

            ultimo_mob_visto = agora
            alvo_lock_id     = alvo["id"]

            cx, cy   = alvo["cx"], alvo["cy"]
            dx, dy   = cx - centro_x, cy - centro_y
            dist     = (dx**2 + dy**2) ** 0.5
            mob_sx   = cap.win_x + cx
            mob_sy   = cap.win_y + cy

            ultimo_alvo_pos  = (cx, cy)
            ultimo_alvo_dist = dist
            pos_morte        = None  # novo alvo ativo → descarta drop spot anterior

            if alvo["id"] != alvo_lock_id:
                logging.info(f'Alvo ID={alvo["id"]} dist={dist:.0f}')

            # dist stuck (atrás de parede)
            if alvo["id"] != dist_hist_id:
                dist_hist_id = alvo["id"]
                dist_hist.clear()
            dist_hist.append(dist)

            if (len(dist_hist) >= DIST_STUCK_JANELA
                    and max(dist_hist) - min(dist_hist) < DIST_STUCK_DELTA
                    and dist > RANGE):
                logging.warning(f"Mob ID={alvo['id']} atrás de parede (dist~{dist:.0f}), escapando")
                inv_x = -dx / dist if dist > 0 else 1.0
                inv_y = -dy / dist if dist > 0 else 0.0
                for _ in range(DIST_STUCK_PASSOS):
                    ex = max(cap.win_x + 150, min(cap.win_x + centro_x + int(inv_x * DIST_STUCK_PASSO_DIST), cap.win_x + img_w - 150))
                    ey = max(cap.win_y + 150, min(cap.win_y + centro_y + int(inv_y * DIST_STUCK_PASSO_DIST), cap.win_y + img_h - 150))
                    click_humano(ex, ey)
                    time.sleep(WALK_COOLDOWN + random.uniform(0.05, 0.15))
                mobs_bloqueados.add(alvo["id"])
                alvo_lock_id = None
                dist_hist.clear()
                continue

            # muito perto
            if dist < PLAYER_RADIUS:
                if agora - ultimo_skill >= SKILL_CD:
                    if usar_skill(mob_sx, mob_sy):
                        ultimo_skill = agora
                        logging.info("Skill (mob colado)")
                    elif agora - ultimo_ataque >= ATTACK_COOLDOWN:
                        click_humano(mob_sx, mob_sy, jitter=10)
                        ultimo_ataque = agora
                elif agora - ultimo_ataque >= ATTACK_COOLDOWN:
                    click_humano(mob_sx, mob_sy, jitter=10)
                    ultimo_ataque = agora
                continue

            # em range
            if dist <= RANGE:
                fora_range_count = 0
                if agora - ultimo_skill >= SKILL_CD:
                    if usar_skill(mob_sx, mob_sy):
                        ultimo_skill = agora
                        logging.info("Skill usada")
                    elif agora - ultimo_ataque >= ATTACK_COOLDOWN:
                        click_humano(mob_sx, mob_sy, jitter=10)
                        ultimo_ataque = agora
                elif agora - ultimo_ataque >= ATTACK_COOLDOWN:
                    click_humano(mob_sx, mob_sy, jitter=10)
                    ultimo_ataque = agora
                    logging.info("Atacando")

            # fora de range
            else:
                if agora - ultimo_walk >= MOB_WALK_COOLDOWN:
                    fora_range_count += 1
                    mob_id = alvo["id"]
                    if mob_id is not None:
                        mob_tentativas[mob_id] = mob_tentativas.get(mob_id, 0) + 1
                        if mob_tentativas[mob_id] >= MOB_MAX_TENTATIVAS:
                            mobs_bloqueados.add(mob_id)
                            alvo_lock_id = None
                            logging.warning(f"Mob ID={mob_id} bloqueado (inalcançável)")

                    if fora_range_count >= FORA_RANGE_MAX:
                        logging.warning(f"Mob inalcançável após {fora_range_count} tentativas")
                        escapar_stuck(frame, centro_x, centro_y, img_w, img_h)
                        alvo_lock_id     = None
                        fora_range_count = 0
                    else:
                        moveu = False
                        for desvio in [0, 0.4, -0.4, 0.8, -0.8]:
                            ang = np.arctan2(dy, dx) + desvio + np.random.uniform(-0.1, 0.1)
                            tx  = max(cap.win_x + 100, min(cap.win_x + int(cx - np.cos(ang) * (RANGE - 50)), cap.win_x + img_w - 100))
                            ty  = max(cap.win_y + 100, min(cap.win_y + int(cy - np.sin(ang) * (RANGE - 50)), cap.win_y + img_h - 100))
                            if pixel_valido(frame, tx, ty, centro_x, centro_y):
                                click_humano(tx, ty, jitter=12)
                                ultimo_walk = agora
                                moveu = True
                                logging.debug("Movendo ao mob")
                                break
                        if not moveu:
                            mobs_bloqueados.add(alvo["id"])
                            alvo_lock_id = None
                            logging.warning(f"Mob ID={alvo['id']} inacessível")

        # ---------- SEM ALVOS — patrol ----------
        else:
            alvo_lock_id  = None
            sem_mob_secs  = agora - ultimo_mob_visto

            # ---------- DROP SPOT ----------
            indo_ao_money = agora - ultimo_money_click < MONEY_WALK_COOLDOWN
            if pos_morte is not None and not indo_ao_money:
                if agora - morte_walk_time > MORTE_CHECK_TIMEOUT:
                    logging.debug("Drop spot timeout")
                    pos_morte = None
                else:
                    mx, my = pos_morte
                    dist_morte = ((mx - centro_x)**2 + (my - centro_y)**2) ** 0.5
                    if dist_morte < MORTE_CHECK_DIST:
                        logging.debug(f"Chegou ao drop spot dist={dist_morte:.0f}")
                        pos_morte = None
                    elif agora - ultimo_walk >= WALK_COOLDOWN:
                        tx = cap.win_x + mx
                        ty = cap.win_y + my
                        if pixel_valido(frame, tx, ty, centro_x, centro_y):
                            click_humano(tx, ty, jitter=10)
                            ultimo_walk = agora
                            logging.info(f"Indo ao drop spot dist={dist_morte:.0f}")

            if sem_mob_secs >= SEM_MOB_TIMEOUT:
                ultimo_mob_visto = agora
                mobs_bloqueados.clear();  mob_tentativas.clear()
                money_bloqueados.clear(); money_tentativas.clear()
                item_bloqueados.clear();  item_tentativas.clear()
                dist_hist.clear(); historico_clicks.clear()
                pos_morte = None
                logging.warning(f"Sem mob por {sem_mob_secs:.0f}s → limpando blacklists")

            if pos_morte is None and not indo_ao_money and agora - ultimo_walk >= WALK_COOLDOWN:
                dist_patrol = random.randint(PATROL_DIST_MIN, PATROL_DIST_MAX)
                ddx, ddy   = DIRECOES[direcao_atual]
                click_x = max(cap.win_x + 200, min(cap.win_x + centro_x + ddx * dist_patrol, cap.win_x + img_w - 200))
                click_y = max(cap.win_y + 200, min(cap.win_y + centro_y + ddy * dist_patrol, cap.win_y + img_h - 200))

                if pixel_valido(frame, click_x, click_y, centro_x, centro_y):
                    click_humano(click_x, click_y, jitter=15)
                    historico_clicks.append((click_x - cap.win_x, click_y - cap.win_y))
                    if len(historico_clicks) == PATROL_HISTORICO_MAX:
                        xs = [p[0] for p in historico_clicks]
                        ys = [p[1] for p in historico_clicks]
                        spread = ((max(xs) - min(xs))**2 + (max(ys) - min(ys))**2) ** 0.5
                        if spread < PATROL_LOOP_RAIO and agora - ultimo_dir_change >= PATROL_DIR_COOLDOWN:
                            direcao_atual = (direcao_atual + 1) % len(DIRECOES)
                            historico_clicks.clear()
                            ultimo_dir_change = agora
                            logging.warning(f"Loop detectado → dir={direcao_atual}")
                    logging.info(f"Patrol dir={direcao_atual} dist={dist_patrol}")
                else:
                    if agora - ultimo_dir_change >= PATROL_DIR_COOLDOWN:
                        direcao_atual = (direcao_atual + 1) % len(DIRECOES)
                        ultimo_dir_change = agora
                        logging.debug(f"Patrol bloqueado → dir={direcao_atual}")
                    ultimo_walk = agora

        # ---------- FPS LIMITER ----------
        elapsed = time.time() - inicio_loop
        if elapsed < FRAME_TIME:
            time.sleep(FRAME_TIME - elapsed)

    logging.info("Bot encerrado")

# =========================
# START
# =========================

if __name__ == "__main__":
    try:
        main()
    except pyautogui.FailSafeException:
        logging.warning("Failsafe ativado")
    except KeyboardInterrupt:
        logging.warning("Interrompido")
    except Exception as e:
        logging.critical(f"Crash: {e}", exc_info=True)

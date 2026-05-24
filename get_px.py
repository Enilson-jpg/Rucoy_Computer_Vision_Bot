import ctypes
import cv2 as cv
import numpy as np
import win32gui
import win32ui
import win32api


def capturar_janela(nome):
    hwnd = win32gui.FindWindow(None, nome)
    if not hwnd:
        raise Exception(f"Janela '{nome}' não encontrada!")

    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    w = right - left
    h = bottom - top

    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc  = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()

    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(mfc_dc, w, h)
    save_dc.SelectObject(bmp)

    ctypes.windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 1)

    info = bmp.GetInfo()
    bits = bmp.GetBitmapBits(True)
    img  = np.frombuffer(bits, dtype=np.uint8).reshape(
        info["bmHeight"], info["bmWidth"], 4
    )

    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv.cvtColor(img, cv.COLOR_BGRA2BGR)


frame = capturar_janela("LDPlayer")
h, w  = frame.shape[:2]
centro_x, centro_y = 800, 500

overlay = frame.copy()

for gx in range(0, w, 100):
    cv.line(overlay, (gx, 0), (gx, h), (40, 40, 40), 1)
    cv.putText(overlay, str(gx), (gx + 2, 12), cv.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
for gy in range(0, h, 100):
    cv.line(overlay, (0, gy), (w, gy), (40, 40, 40), 1)
    cv.putText(overlay, str(gy), (2, gy + 12), cv.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
cv.drawMarker(overlay, (centro_x, centro_y), (0, 255, 0), cv.MARKER_CROSS, 20, 2)


def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_MOUSEMOVE:
        tmp = overlay.copy()
        if 0 <= y < h and 0 <= x < w:
            bgr    = frame[y, x].tolist()
            hsv_px = cv.cvtColor(np.uint8([[bgr]]), cv.COLOR_BGR2HSV)[0][0].tolist()
            cv.putText(tmp,
                f"({x},{y})  BGR=({bgr[0]},{bgr[1]},{bgr[2]})  "
                f"HSV=({hsv_px[0]},{hsv_px[1]},{hsv_px[2]})",
                (10, h - 10), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv.rectangle(tmp, (w - 80, h - 80), (w - 10, h - 10), bgr, -1)
        cv.imshow("get_px", tmp)

    elif event == cv.EVENT_LBUTTONDOWN:
        if 0 <= y < h and 0 <= x < w:
            bgr    = frame[y, x].tolist()
            hsv_px = cv.cvtColor(np.uint8([[bgr]]), cv.COLOR_BGR2HSV)[0][0].tolist()
            dist   = int(((x - centro_x)**2 + (y - centro_y)**2) ** 0.5)
            print(
                f"x={x:4d}  y={y:4d}  dist={dist:4d}  "
                f"BGR=({bgr[0]:3d},{bgr[1]:3d},{bgr[2]:3d})  "
                f"HSV=({hsv_px[0]:3d},{hsv_px[1]:3d},{hsv_px[2]:3d})"
            )


print(f"LDPlayer capturado: {w}x{h}")
print("Clique esquerdo: cor do pixel  |  Q: sair")

cv.namedWindow("get_px", cv.WINDOW_NORMAL)
cv.setMouseCallback("get_px", on_mouse)
cv.imshow("get_px", overlay)
cv.waitKey(0)
cv.destroyAllWindows()

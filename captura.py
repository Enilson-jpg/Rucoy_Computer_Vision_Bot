import dxcam
import cv2 as cv
import win32gui
import win32api
import time

_REGIAO_CACHE_INTERVAL = 5.0  # segundos entre recálculos de posição da janela

class Captura:
    def __init__(self, window_name):
        self.hwnd = win32gui.FindWindow(None, window_name)
        if not self.hwnd:
            raise Exception(f"Janela '{window_name}' não encontrada!")

        self.win_x = self.win_y = self.win_x2 = self.win_y2 = 0
        self._ultimo_update_regiao = 0.0
        self._atualizar_regiao()

        self.camera = dxcam.create()
        print(f"Captura iniciada! Região: ({self.win_x}, {self.win_y}, {self.win_x2}, {self.win_y2})")

    def _atualizar_regiao(self):
        while win32gui.IsIconic(self.hwnd):
            print("LDPlayer minimizado, aguardando...")
            time.sleep(1)

        client_rect = win32gui.GetClientRect(self.hwnd)
        origem = win32gui.ClientToScreen(self.hwnd, (0, 0))

        screen_w = win32api.GetSystemMetrics(0)
        screen_h = win32api.GetSystemMetrics(1)

        self.win_x  = max(0, origem[0])
        self.win_y  = max(0, origem[1])
        self.win_x2 = min(screen_w, origem[0] + client_rect[2])
        self.win_y2 = min(screen_h, origem[1] + client_rect[3])
        self._ultimo_update_regiao = time.time()

    def get_frame(self):
        agora = time.time()
        if agora - self._ultimo_update_regiao >= _REGIAO_CACHE_INTERVAL:
            self._atualizar_regiao()

        region = (self.win_x, self.win_y, self.win_x2, self.win_y2)
        frame = self.camera.grab(region=region)

        if frame is None:
            # janela pode ter movido; força recálculo no próximo tick
            self._ultimo_update_regiao = 0.0
            return None

        return cv.cvtColor(frame, cv.COLOR_RGB2BGR)


# Teste
if __name__ == "__main__":
    cap = Captura("LDPlayer")
    frame = cap.get_frame()
    if frame is not None:
        cv.imwrite("teste_captura.png", frame)
        print(f"Frame capturado! Tamanho: {frame.shape}")
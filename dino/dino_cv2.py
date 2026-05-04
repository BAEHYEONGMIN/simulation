import ctypes

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import threading
import time
import tkinter as tk
from tkinter import messagebox

import cv2
import mss
import numpy as np
import pyautogui


class RegionSelector:
    def __init__(self, root):
        self.root = root
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.region = None

    def select(self):
        self.top = tk.Toplevel(self.root)
        self.top.attributes("-fullscreen", True)
        self.top.attributes("-alpha", 0.25)
        self.top.attributes("-topmost", True)
        self.top.configure(bg="black")

        self.canvas = tk.Canvas(self.top, cursor="cross", bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.top.bind("<Escape>", lambda e: self.cancel())

        self.root.wait_window(self.top)
        return self.region

    def on_press(self, event):
        self.start_x = event.x_root
        self.start_y = event.y_root

        self.rect = self.canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline="lime",
            width=3,
        )

    def on_drag(self, event):
        self.canvas.coords(
            self.rect,
            self.start_x,
            self.start_y,
            event.x_root,
            event.y_root,
        )

    def on_release(self, event):
        x1 = min(self.start_x, event.x_root)
        y1 = min(self.start_y, event.y_root)
        x2 = max(self.start_x, event.x_root)
        y2 = max(self.start_y, event.y_root)

        width = x2 - x1
        height = y2 - y1

        if width < 10 or height < 10:
            self.region = None
        else:
            self.region = {
                "left": int(x1),
                "top": int(y1),
                "width": int(width),
                "height": int(height),
            }

        self.top.destroy()

    def cancel(self):
        self.region = None
        self.top.destroy()


class DinoBotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dino Auto Jump Bot")
        self.root.geometry("360x260")
        self.root.resizable(False, False)

        self.region = None
        self.running = False
        self.worker = None

        self.pixel_threshold = tk.IntVar(value=180)
        self.obstacle_pixels = tk.IntVar(value=100)
        self.jump_cooldown = tk.DoubleVar(value=0.35)

        self.status_var = tk.StringVar(value="대기중")
        self.region_var = tk.StringVar(value="감지 영역: 미설정")

        self.build_ui()
        
        # 전역 단축키 감시 스레드 시작 (Windows 전용)
        self.start_global_listener()

    def build_ui(self):
        tk.Label(
            self.root,
            text="Dino Auto Jump Bot",
            font=("Arial", 16, "bold"),
        ).pack(pady=10)

        tk.Label(self.root, textvariable=self.status_var).pack()
        tk.Label(self.root, textvariable=self.region_var).pack(pady=5)

        tk.Button(
            self.root,
            text="감지 영역 선택",
            command=self.select_region,
            width=25,
        ).pack(pady=8)

        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=5)

        tk.Button(
            control_frame,
            text="시작",
            command=self.start_bot,
            width=12,
        ).grid(row=0, column=0, padx=5)

        tk.Button(
            control_frame,
            text="정지",
            command=self.stop_bot,
            width=12,
        ).grid(row=0, column=1, padx=5)

        option_frame = tk.Frame(self.root)
        option_frame.pack(pady=10)

        tk.Label(option_frame, text="밝기 기준").grid(row=0, column=0, sticky="w")
        tk.Entry(option_frame, textvariable=self.pixel_threshold, width=8).grid(row=0, column=1)

        tk.Label(option_frame, text="감지 픽셀 수").grid(row=1, column=0, sticky="w")
        tk.Entry(option_frame, textvariable=self.obstacle_pixels, width=8).grid(row=1, column=1)

        tk.Label(option_frame, text="점프 딜레이").grid(row=2, column=0, sticky="w")
        tk.Entry(option_frame, textvariable=self.jump_cooldown, width=8).grid(row=2, column=1)

        tk.Label(
            self.root,
            text="ESC: 봇 정지 / Shift+ESC: 전체 종료",
            fg="gray",
        ).pack(pady=5)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def start_global_listener(self):
        """백그라운드에서 전역 단축키를 감시하는 스레드 실행"""
        thread = threading.Thread(target=self._global_key_monitor, daemon=True)
        thread.start()

    def _global_key_monitor(self):
        """프로그램이 켜져 있는 동안 전역 키 입력을 감시"""
        while True:
            try:
                # Shift (0x10) + ESC (0x1B) 체크
                shift_down = ctypes.windll.user32.GetAsyncKeyState(0x10) & 0x8000
                esc_down = ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000

                if shift_down and esc_down:
                    self.root.after(0, self.on_close)
                    break
                elif esc_down:
                    if self.running:
                        self.root.after(0, self.stop_bot)
                
                time.sleep(0.1)  # CPU 점유율 방지
            except Exception:
                break

    def select_region(self):
        self.stop_bot()

        selector = RegionSelector(self.root)
        region = selector.select()

        if not region:
            self.status_var.set("영역 선택 취소됨")
            return

        self.region = region
        self.region_var.set(
            f"감지 영역: x={region['left']}, y={region['top']}, "
            f"w={region['width']}, h={region['height']}"
        )
        self.status_var.set("영역 설정 완료")

    def start_bot(self):
        if self.region is None:
            messagebox.showwarning("알림", "먼저 감지 영역을 선택하세요.")
            return

        if self.running:
            return

        self.running = True
        self.status_var.set("3초 후 실행 - 크롬 공룡게임 화면 클릭 준비")

        # UI가 포커스를 뺏지 않게 최소화
        self.root.iconify()

        self.worker = threading.Thread(target=self.bot_loop, daemon=True)
        self.worker.start()

    def stop_bot(self):
        self.running = False
        self.status_var.set("정지됨")

    def bot_loop(self):
        pyautogui.PAUSE = 0

        # 시작 버튼 누른 뒤 크롬으로 포커스 넘길 시간
        time.sleep(1)

        # 선택한 감지 영역 중앙을 클릭해서 크롬/게임 캔버스에 포커스 주기
        click_x = self.region["left"] + self.region["width"] // 2
        click_y = self.region["top"] + self.region["height"] // 2
        pyautogui.click(click_x, click_y)

        time.sleep(0.2)

        # 게임 시작 보장
        pyautogui.keyDown("space")
        time.sleep(0.08)
        pyautogui.keyUp("space")

        time.sleep(0.5)

        with mss.MSS() as sct:
            while self.running:
                # ESC 키가 눌렸는지 글로벌하게 확인 (Windows 전용 전역 단축키)
                if ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000:
                    self.root.after(0, self.stop_bot)
                    break

                try:
                    img = np.array(sct.grab(self.region))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

                    dark_pixels = np.sum(gray < self.pixel_threshold.get())

                    print(
                        "dark_pixels:",
                        dark_pixels,
                        "threshold:",
                        self.obstacle_pixels.get()
                    )

                    if dark_pixels > self.obstacle_pixels.get():
                        print("JUMP!")

                        click_x = self.region["left"] + self.region["width"] // 2
                        click_y = self.region["top"] + self.region["height"] // 2
                        pyautogui.click(click_x, click_y)

                        time.sleep(self.jump_cooldown.get())

                    time.sleep(0.005)

                except Exception as e:
                    self.running = False
                    print("ERROR:", e)
                    self.status_var.set(f"에러: {e}")
                    break
    def on_close(self):
        self.stop_bot()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DinoBotApp(root)
    root.mainloop()
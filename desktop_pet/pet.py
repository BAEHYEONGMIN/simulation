import sys
import os
import random
import ctypes
import ctypes.wintypes
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QSystemTrayIcon, QMenu
from PyQt6.QtCore import Qt, QTimer, QPoint
from PyQt6.QtGui import QPixmap, QMovie, QPainter, QColor, QIcon, QTransform


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# ── Win32: 열려 있는 창 Rect 목록 ──────────────────────────────────────────
def get_window_rects(exclude_hwnd=None):
    """현재 화면에 보이는 창들의 (left, top, right, bottom) 목록 반환."""
    rects = []
    IsWindowVisible  = ctypes.windll.user32.IsWindowVisible
    GetWindowRect    = ctypes.windll.user32.GetWindowRect
    GetWindowTextLen = ctypes.windll.user32.GetWindowTextLengthW
    WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)

    def _cb(hwnd, _):
        if hwnd == exclude_hwnd:
            return True
        if not IsWindowVisible(hwnd):
            return True
        if GetWindowTextLen(hwnd) == 0:
            return True
        r = ctypes.wintypes.RECT()
        GetWindowRect(hwnd, ctypes.byref(r))
        w, h = r.right - r.left, r.bottom - r.top
        if w > 100 and h > 50:   # 아주 작은 팝업 제외
            rects.append((r.left, r.top, r.right, r.bottom))
        return True

    ctypes.windll.user32.EnumWindows(WNDENUMPROC(_cb), 0)
    return rects


# ── 메인 위젯 ────────────────────────────────────────────────────────────────
class DesktopPet(QWidget):
    MODE_WALK = "walk"
    MODE_FALL = "fall"   # 자유낙하 (클릭 해제 후)

    GRAVITY   = 1.5      # 낙하 가속도 (px/frame²)
    MAX_FALL  = 20       # 최대 낙하 속도

    def __init__(self):
        super().__init__()

        # ── 창 설정 ──
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # ── 크기 ──
        self.pet_width  = 100
        self.pet_height = 100
        self.resize(self.pet_width, self.pet_height)

        # ── 화면 좌표 ──
        virt = QApplication.primaryScreen().virtualGeometry()
        self.min_x = virt.left()
        self.max_x = virt.right()
        primary    = QApplication.primaryScreen().geometry()
        self.screen_height = primary.height()
        self.floor_y = self.screen_height - self.pet_height - 2  # 바닥 Y 기준

        # 모니터 객체와 경계선 목록 (x 좌위 기준 정렬)
        self._screens = sorted(QApplication.screens(), key=lambda s: s.geometry().left())
        self._boundaries = []
        for i in range(len(self._screens) - 1):
            right_of_left = self._screens[i].geometry().right()
            left_of_right = self._screens[i + 1].geometry().left()
            self._boundaries.append((right_of_left, left_of_right))

        # 창 충돌 체크에서 제외할 경계선 x 좌표 세트 (모니터 경계 작업표시줄 등은 벽이 아님)
        self._boundary_xs = set()
        for (rb, lr) in self._boundaries:
            self._boundary_xs.add(rb)
            self._boundary_xs.add(lr)

        # 초기 위치: 주 모니터 내부 중앙 (경계선에 걸치면 양쪽 모니터에 동시 렌더링되는 버그 발생)
        self.x = primary.left() + (primary.width() - self.pet_width) // 2
        self.y = self.floor_y
        self.move(self.x, self.y)

        # ── 상태 ──
        self.speed_x   = random.choice([-2, 2])
        self.speed_y   = 0.0
        self.direction = "right" if self.speed_x > 0 else "left"
        self.mode      = self.MODE_WALK

        # ── 기능 플래그 ──
        self.window_enabled = True   # 창 테두리 인식
        self.hang_enabled   = True   # 좌클릭 집어들기 (매달리기)

        # ── 드래그 ──
        self._drag_offset = None
        self._is_dragging = False

        # ── 이미지 라벨 (load_image 보다 먼저) ──
        self.label = QLabel(self)
        self.label.resize(self.pet_width, self.pet_height)
        self.label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # 이미지 경로 - doro 폴더는 프로젝트 루트(desktop_pet 상위)에 있음
        self.walk_image_path = resource_path("doro.gif")
        _project_root = os.path.dirname(os.path.abspath(__file__))
        self.hang_image_path = os.path.join(_project_root, "..", "doro", "cola_doro.gif")
        self._movie = None
        self._load_image(self.walk_image_path)

        # ── 창 목록 캐시 (자신의 hwnd 는 제외) ──
        self._window_rects = []
        self._my_hwnd = None   # show() 이후 채움

        self._win_scan_timer = QTimer(self)
        self._win_scan_timer.timeout.connect(self._scan_windows)
        self._win_scan_timer.start(400)

        # ── 메인 타이머 (~60FPS) ──
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)

        # ── 트레이 ──
        self.is_always_on_top = True
        self._setup_tray()

    # ── show() 후 hwnd 확보 ─────────────────────────────────────────────────
    def showEvent(self, event):
        super().showEvent(event)
        if self._my_hwnd is None:
            self._my_hwnd = int(self.winId())

    # ── 트레이 ──────────────────────────────────────────────────────────────
    def _setup_tray(self):
        px = QPixmap(16, 16)
        px.fill(QColor(Qt.GlobalColor.transparent))
        p = QPainter(px)
        p.setBrush(QColor("#fbbf24"))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(0, 0, 16, 16)
        p.end()

        self.tray = QSystemTrayIcon(QIcon(px), self)
        self.tray.setToolTip("Desktop Pet")

        menu = QMenu()

        a = menu.addAction("보이기 / 숨기기")
        a.triggered.connect(self._toggle_visibility)

        menu.addSeparator()

        a_top = menu.addAction("항상 위")
        a_top.setCheckable(True); a_top.setChecked(True)
        a_top.triggered.connect(self._toggle_ontop)

        menu.addSeparator()

        self._a_hang = menu.addAction("집어들기 (ON)")
        self._a_hang.setCheckable(True); self._a_hang.setChecked(True)
        self._a_hang.triggered.connect(self._toggle_hang)

        self._a_win = menu.addAction("창 테두리 인식 (ON)")
        self._a_win.setCheckable(True); self._a_win.setChecked(True)
        self._a_win.triggered.connect(self._toggle_window)

        menu.addSeparator()

        menu.addAction("종료").triggered.connect(QApplication.instance().quit)

        self.tray.setContextMenu(menu)
        self.tray.show()

    def _toggle_visibility(self):
        if self.isVisible():
            self.hide()
        else:
            self.show(); self.raise_()

    def _toggle_ontop(self, checked):
        self.is_always_on_top = checked
        flags = self.windowFlags()
        flags = (flags | Qt.WindowType.WindowStaysOnTopHint) if checked \
            else (flags & ~Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowFlags(flags)
        if self.isVisible():
            self.show()

    def _toggle_hang(self, checked):
        self.hang_enabled = checked
        self._a_hang.setText(f"집어들기 ({'ON' if checked else 'OFF'})")
        if not checked and self._is_dragging:
            self._release_pet()

    def _toggle_window(self, checked):
        self.window_enabled = checked
        self._a_win.setText(f"창 테두리 인식 ({'ON' if checked else 'OFF'})")
        if not checked:
            self._window_rects = []

    # ── 이미지 ────────────────────────────────────────────────────────────
    def _load_image(self, path):
        """QMovie를 새로 교체. 이전 movie는 완전히 정리."""
        if not os.path.exists(path):
            # 파일 없으면 걷기 이미지로 폴백
            if path != self.walk_image_path:
                self._load_image(self.walk_image_path)
            return

        # 이전 movie 정리
        if self._movie is not None:
            try:
                self._movie.frameChanged.disconnect()
            except Exception:
                pass
            self._movie.stop()
            self._movie = None

        movie = QMovie(path)
        movie.setScaledSize(self.label.size())
        movie.frameChanged.connect(self._on_frame)
        movie.start()
        self._movie = movie

    def _on_frame(self, _):
        if self._movie is None:
            return
        px = self._movie.currentPixmap()
        # 드래그 중이 아닐 때만 방향 뒤집기
        if not self._is_dragging and self.direction == "left":
            px = px.transformed(QTransform().scale(-1, 1))
        self.label.setPixmap(px)

    # ── 창 스캔 ──────────────────────────────────────────────────────────
    def _scan_windows(self):
        if self.window_enabled:
            self._window_rects = get_window_rects(exclude_hwnd=self._my_hwnd)

    # ── 마우스 이벤트 ────────────────────────────────────────────────────
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.hang_enabled:
            self._is_dragging = True
            self._drag_offset = event.globalPosition().toPoint() - self.pos()
            self._load_image(self.hang_image_path)

        elif event.button() == Qt.MouseButton.RightButton:
            QApplication.instance().quit()

    def mouseMoveEvent(self, event):
        if self._is_dragging and self._drag_offset is not None:
            new_pos = event.globalPosition().toPoint() - self._drag_offset
            self.x = new_pos.x()
            self.y = float(new_pos.y())
            self.move(new_pos.x(), new_pos.y())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._is_dragging:
            self._release_pet()

    def _release_pet(self):
        """드래그 해제 → 자유낙하 모드"""
        self._is_dragging = False
        self._drag_offset  = None
        self.speed_y       = 0.0
        self.mode          = self.MODE_FALL
        self._load_image(self.walk_image_path)

    # ── 메인 틱 ──────────────────────────────────────────────────────────
    def _tick(self):
        if self._is_dragging:
            return  # 드래그 중엔 위치를 mouseMoveEvent가 담당

        if self.mode == self.MODE_FALL:
            self._tick_fall()
        else:
            self._tick_walk()

    # ── 자유낙하 ─────────────────────────────────────────────────────────
    def _tick_fall(self):
        self.speed_y = min(self.speed_y + self.GRAVITY, self.MAX_FALL)
        self.y += self.speed_y

        # 바닥 또는 창 상단에 닿으면 착지
        landed = False

        # 창 테두리 착지 체크
        if self.window_enabled:
            for (wl, wt, wr, wb) in self._window_rects:
                if self.x + self.pet_width > wl and self.x < wr:
                    # 펫이 이 창의 x 범위 안에 있고, 창 상단을 통과하려 할 때
                    if self.y + self.pet_height >= wt and self.y < wt:
                        self.y = wt - self.pet_height
                        landed = True
                        break

        # 바닥 착지
        if self.y >= self.floor_y:
            self.y = self.floor_y
            landed = True

        self.move(int(self.x), int(self.y))

        if landed:
            self.speed_y = 0.0
            self.mode    = self.MODE_WALK

    # ── 걷기 ──────────────────────────────────────────────────────────────
    def _tick_walk(self):
        next_x = self.x + self.speed_x

        # ── 창 테두리 좌우 충돌 (모니터 경계에 있는 창은 벽로 취급 안 함) ──
        if self.window_enabled:
            pet_l = next_x
            pet_r = next_x + self.pet_width
            pet_t = self.y
            pet_b = self.y + self.pet_height

            for (wl, wt, wr, wb) in self._window_rects:
                # Y축이 갹치는 창만 대상
                if pet_b <= wt or pet_t >= wb:
                    continue
                # 모니터 경계에서 시작하는 창(보조 모니터 작업표시줄 등)만 벽 제외
                # wr 기준으로도 필터하면 주 모니터 최대화 창 오른쪽 경계를 통과하는 비대칭 버그 발생
                if wl in self._boundary_xs:
                    continue
                # 오른쪽으로 가다가 창 왼쪽 면 충돌
                if self.speed_x > 0 and self.x + self.pet_width <= wl and pet_r > wl:
                    next_x = wl - self.pet_width
                    self.speed_x = -abs(self.speed_x)
                    self._on_direction_change()
                    break
                # 왼쪽으로 가다가 창 오른쪽 면 충돌
                if self.speed_x < 0 and self.x >= wr and pet_l < wr:
                    next_x = wr
                    self.speed_x = abs(self.speed_x)
                    self._on_direction_change()
                    break

        # ── 모니터 경계 교차 시 텔레포트 (창 충돌보다 우선, 마지막에 적용) ──
        next_x = self._snap_across_boundary(self.x, next_x)

        # ── 화면 경계 충돌 ──
        if next_x <= self.min_x:
            next_x = self.min_x
            self.speed_x = abs(self.speed_x)
            self._on_direction_change()
        elif next_x >= self.max_x - self.pet_width:
            next_x = self.max_x - self.pet_width
            self.speed_x = -abs(self.speed_x)
            self._on_direction_change()

        self.x = next_x
        self.move(self.x, self.y)

    def _snap_across_boundary(self, cur_x: int, next_x: int) -> int:
        """다음 프레임 x가 모니터 경계를 과통하면 다음 모니터 시작점으로 텔레포트.
        이렇게 하리에 줄쳐' 친 첩하는 순간이 없어져 DWM 이중 렌더링을 방지합니다."""
        for (rb, lr) in self._boundaries:
            if self.speed_x > 0:
                # 오른쪽 이동: 폴 오른쪽이 rb를 넘으려 할 때
                if cur_x + self.pet_width <= rb < next_x + self.pet_width:
                    return lr          # 다음 모니터 시작점으로 텔레포트
            else:
                # 왼쪽 이동: 폴 왼쪽이 lr을 넘으려 할 때
                if cur_x >= lr > next_x:
                    return rb - self.pet_width  # 이전 모니터 끝점으로 텔레포트
        return next_x

    def _on_direction_change(self):
        self.direction = "right" if self.speed_x > 0 else "left"
        self._load_image(self.walk_image_path)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    pet = DesktopPet()
    pet.show()
    sys.exit(app.exec())

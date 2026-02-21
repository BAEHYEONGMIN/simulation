import sys
import os
import threading
import http.server
import socketserver
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSystemTrayIcon, QMenu
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtCore import Qt, QUrl, QRect, QTimer
import ctypes
from PyQt6.QtGui import QColor, QIcon, QPixmap

def make_icon(color="#38bdf8", size=16):
    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(Qt.GlobalColor.transparent))
    from PyQt6.QtGui import QPainter
    painter = QPainter(pixmap)
    painter.setBrush(QColor(color))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(0, 0, size, size)
    painter.end()
    return QIcon(pixmap)

class SidePanel(QWidget):
    """드래그와 높이 확장을 담당하는 좌측 패널"""
    def __init__(self, parent_win):
        super().__init__()
        self.parent_win = parent_win
        self.setFixedWidth(24)
        self.setStyleSheet("""
            QWidget {
                background: rgba(15, 15, 20, 0.4); /* 투명도 0.9 -> 0.4 대폭 증가 */
                color: #cbd5e1;
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-right: none;
                border-radius: 0px; /* 라운드 사각형 요소 완전히 파괴 */
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0) # 버튼이 튀어나오지 않게 여백 완전히 없앰
        layout.setSpacing(0)
        
        layout.addStretch() # 위쪽(확장 영역)은 빈 공간으로 둠
        
        # 하단 48px 고정 영역 (버튼과 라벨을 하나로 묶음)
        bottom_widget = QWidget()
        bottom_widget.setFixedHeight(45)
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)
        
        # 확장/축소 토글 버튼
        self.btn_expand = QPushButton("🔼")
        self.btn_expand.setFixedSize(24, 20)
        self.btn_expand.setStyleSheet("background: transparent; border: none; font-size: 11px; margin-top: 2px; color: white;")
        self.btn_expand.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_expand.clicked.connect(self.parent_win.toggle_expand)
        
        # 드래그 손잡이용 라벨 (1개로 축소)
        self.lbl_drag = QLabel("⋮")
        self.lbl_drag.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_drag.setStyleSheet("font-size: 16px; color: #64748b; background: transparent; border: none; margin-bottom: 2px;")
        self.lbl_drag.setCursor(Qt.CursorShape.OpenHandCursor)
        
        bottom_layout.addWidget(self.btn_expand)
        bottom_layout.addWidget(self.lbl_drag)
        
        layout.addWidget(bottom_widget)
        
        self._drag_pos = None

    def mousePressEvent(self, event):
        if self.parent_win.is_locked:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.lbl_drag.setCursor(Qt.CursorShape.ClosedHandCursor)
            self._drag_pos = event.globalPosition().toPoint() - self.parent_win.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.parent_win.is_locked:
            return
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            # 창 드래그 이동 (자유롭게)
            self.parent_win.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        self.lbl_drag.setCursor(Qt.CursorShape.OpenHandCursor)
        event.accept()


class TaskbarWidget(QMainWindow):
    def __init__(self, app: QApplication, server_port: int):
        super().__init__()
        self.app = app
        self.server_port = server_port
        
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.screen_geo = QApplication.primaryScreen().geometry()
        
        self.w = 560
        self.h_min = 48
        self.h_max = 150
        self.x_pos = (self.screen_geo.width() - self.w) // 2
        # 초기 위치: 화면 맨 밑바닥에서 약간 띄움
        self.y_min = self.screen_geo.height() - self.h_min - 40 
        
        self.setGeometry(self.x_pos, self.y_min, self.w, self.h_min)
        # 옵션 플래그
        self.expanded = False
        self.is_locked = False
        self.is_always_on_top = True
        self.is_force_topmost = True  # 새롭게 추가된 강력한 최상단 모드 플래그
        
        central = QWidget()
        central.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCentralWidget(central)
        
        # 가로 레이아웃 (좌측 패널 + 우측 HTML 위젯)
        hbox = QHBoxLayout(central)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(0)
        
        self.side_panel = SidePanel(self)
        hbox.addWidget(self.side_panel)
        
        self.view = QWebEngineView()
        self.view.page().setBackgroundColor(QColor(Qt.GlobalColor.transparent))
        
        settings = self.view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False)
        settings.setAttribute(QWebEngineSettings.WebAttribute.AllowRunningInsecureContent, True)
        
        self.view.load(QUrl(f"http://localhost:{self.server_port}/taskbar.html"))
        hbox.addWidget(self.view)
        
        self._setup_tray()

        # 강제 최상단 유지 타이머 (윈도우 작업 표시줄이나 게임 등에 가려지지 않게 방어, 50ms로 단축)
        self.topmost_timer = QTimer(self)
        self.topmost_timer.timeout.connect(self._enforce_topmost)
        self.topmost_timer.start(50)

    def changeEvent(self, event):
        if event.type() == event.Type.WindowStateChange:
            if self.isMinimized() and self.is_always_on_top:
                # 윈도우 작업표시줄이나 바탕화면 보기에 의해 위젯이 최소화/숨겨지면 즉시 복구
                self.setWindowState(self.windowState() & ~Qt.WindowState.WindowMinimized)
                self.showNormal()
                self.raise_()
        super().changeEvent(event)

    def hideEvent(self, event):
        # 만약 시스템에 의해 강제로 숨겨진다면 (사용자가 숨기기 누른게 아니라면)
        # 방어 로직 추가 가능하지만, 일단은 기본 유지
        super().hideEvent(event)

    def _enforce_topmost(self):
        # '강제 맨 위 유지 모드'가 켜져있을 때만 동작
        if self.is_always_on_top and self.is_force_topmost:
            # 창이 숨겨져버렸다면 다시 끌어올림 (단, 트레이에서 수동으로 숨긴 경우는 제외하도록 isVisible 체크는 유지하되, 약간의 강제성 부여)
            if not self.isVisible():
                return
                
            hwnd = int(self.winId())
            # HWND_TOPMOST = -1, SWP_NOMOVE = 0x0002, SWP_NOSIZE = 0x0001, SWP_NOACTIVATE = 0x0010
            # 윈도우 작업 표시줄이나 게임 창이 계속 TOPMOST를 가져가려고 할 때, 우리 위젯을 그들보다 더 위로 쑤셔넣음
            # BringWindowToTop으로 OS 수준에서 제일 위로 강제 호출
            ctypes.windll.user32.BringWindowToTop(hwnd)
            # 확실하게 TOPMOST 속성을 0.5초마다 캐시 우회하여 재박제
            ctypes.windll.user32.SetWindowPos(hwnd, -2, 0, 0, 0, 0, 0x0013) # NOTOPMOST
            ctypes.windll.user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, 0x0013) # TOPMOST

    def _setup_tray(self):
        icon = make_icon("#38bdf8")
        self.tray = QSystemTrayIcon(icon, self)
        self.tray.setToolTip("Taskbar YouTube Widget")
        
        menu = QMenu()
        action_toggle = menu.addAction("보이기 / 숨기기")
        action_toggle.triggered.connect(self._toggle_visibility)
        
        menu.addSeparator()
        
        action_ontop = menu.addAction("항상 위")
        action_ontop.setCheckable(True)
        action_ontop.setChecked(True)
        action_ontop.triggered.connect(self._toggle_ontop)
        
        action_force_topmost = menu.addAction("강제 맨 위 유지 모드 (게임/작업표시줄 뚫기)")
        action_force_topmost.setCheckable(True)
        action_force_topmost.setChecked(False)
        action_force_topmost.triggered.connect(self._toggle_force_topmost)
        
        action_lock = menu.addAction("위치 고정")
        action_lock.setCheckable(True)
        action_lock.setChecked(False)
        action_lock.triggered.connect(self._toggle_lock)
        
        menu.addSeparator()
        action_quit = menu.addAction("종료")
        action_quit.triggered.connect(self.app.quit)
        self.tray.setContextMenu(menu)
        self.tray.show()

    def _toggle_ontop(self, checked):
        self.is_always_on_top = checked
        flags = self.windowFlags()
        if checked:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        
        if self.isVisible():
            self.show() # 플래그 변경 후 화면에서 사라지는 현상 방지용 재호출

    def _toggle_force_topmost(self, checked):
        self.is_force_topmost = checked

    def _toggle_lock(self, checked):
        self.is_locked = checked

    def _toggle_visibility(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self.activateWindow()

    def toggle_expand(self):
        """설정 패널(높이 150px)과 닫힌 상태(높이 48px)를 일순간에 켜고 끔 (흔들림 방지용)"""
        cx = self.x()
        cy = self.y()
        
        if self.expanded:
            # 축소
            self.setGeometry(QRect(cx, cy + (self.h_max - self.h_min), self.w, self.h_min))
            self.side_panel.btn_expand.setText("🔼")
            self.expanded = False
        else:
            # 확장
            self.setGeometry(QRect(cx, cy - (self.h_max - self.h_min), self.w, self.h_max))
            self.side_panel.btn_expand.setText("🔽")
            self.expanded = True

class LocalHTTPServer:
    def __init__(self, port=0):
        self.port = port
        self.server = None
        self.thread = None

    def start(self):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        try:
            Handler = http.server.SimpleHTTPRequestHandler
            class QuietHandler(Handler):
                def log_message(self, format, *args): pass
            self.server = socketserver.TCPServer(("", self.port), QuietHandler)
            self.port = self.server.server_address[1]
            self.thread = threading.Thread(target=self.server.serve_forever)
            self.thread.daemon = True
            self.thread.start()
        except OSError:
            pass 

if __name__ == "__main__":
    local_server = LocalHTTPServer(port=0)
    local_server.start()
    
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--autoplay-policy=no-user-gesture-required --disable-features=HardwareMediaKeyHandling"
    
    app = QApplication(sys.argv)
    app.setApplicationName("Taskbar YouTube Widget")
    app.setQuitOnLastWindowClosed(False)
    
    window = TaskbarWidget(app, local_server.port)
    window.show()
    sys.exit(app.exec())

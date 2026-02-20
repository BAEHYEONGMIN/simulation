"""
YouTube Music Player - PyQt6 투명 데스크탑 위젯
실행: python widget.py
종료: 시스템 트레이 아이콘 우클릭 → 종료
"""

import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QSystemTrayIcon, QMenu, QGraphicsOpacityEffect
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings, QWebEngineProfile
from PyQt6.QtCore import Qt, QUrl, QPoint, QPropertyAnimation
from PyQt6.QtGui import QColor, QIcon, QPixmap
import threading
import http.server
import socketserver


def make_icon(color="#a78bfa", size=16):
    """icon.png 파일이 있으면 로드, 없으면 기본 단색 아이콘 생성"""
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
    if os.path.exists(icon_path):
        return QIcon(icon_path)
    pix = QPixmap(size, size)
    pix.fill(QColor(color))
    return QIcon(pix)


class TitleBar(QWidget):
    """드래그 가능한 상단 컨트롤 바"""

    def __init__(self, parent_win):
        super().__init__(parent_win)
        self.parent_win = parent_win
        self._drag_pos = None

        self.setFixedHeight(30)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setStyleSheet("""
            QWidget {
                background: rgba(18, 18, 28, 0.9);
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 6, 0)
        layout.setSpacing(4)

        # 드래그 핸들 라벨 (마우스 이벤트 투명 처리)
        label = QWidget()
        label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        label.setStyleSheet("background: transparent;")
        layout.addWidget(label)
        layout.addStretch()

        # 📌 항상 위 토글
        self.pin_btn = QPushButton("📌")
        self.pin_btn.setFixedSize(24, 24)
        self.pin_btn.setCheckable(True)
        self.pin_btn.setChecked(True)
        self.pin_btn.setToolTip("항상 위 ON/OFF")
        self.pin_btn.setStyleSheet(self._btn_style())
        self.pin_btn.clicked.connect(self._toggle_pin)
        self.pin_btn.setAttribute(Qt.WidgetAttribute.WA_NoMousePropagation)
        layout.addWidget(self.pin_btn)

        # ✕ 닫기
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(24, 24)
        close_btn.setToolTip("숨기기 (트레이에 남음)")
        close_btn.setStyleSheet(self._btn_style(close=True))
        close_btn.clicked.connect(parent_win.hide)
        close_btn.setAttribute(Qt.WidgetAttribute.WA_NoMousePropagation)
        layout.addWidget(close_btn)

    def _btn_style(self, close=False):
        hover = "#c0392b" if close else "rgba(255,255,255,0.15)"
        return f"""
            QPushButton {{
                background: transparent;
                color: rgba(255,255,255,0.65);
                border: none;
                border-radius: 4px;
                font-size: 12px;
            }}
            QPushButton:hover {{ background: {hover}; color: white; }}
            QPushButton:checked {{ color: #a78bfa; }}
        """

    def _toggle_pin(self, checked):
        flags = self.parent_win.windowFlags()
        if checked:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        self.parent_win.setWindowFlags(flags)
        self.parent_win.show()

    # ─── 드래그 이동 ───────────────────────────────────────────
    def mousePressEvent(self, event):
        if self.parent_win.is_locked:
            return
        if event.button() == Qt.MouseButton.LeftButton:
            # 글로벌 클릭 위치 — 창 위치 = 오프셋
            self._drag_pos = event.globalPosition().toPoint() - self.parent_win.pos()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.parent_win.is_locked:
            return
        if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.parent_win.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        event.accept()


class YouTubeWidget(QMainWindow):
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
        self.is_locked = False

        # ─── 초기 위치: 화면 우하단 ──────────────────────────────
        screen = QApplication.primaryScreen().availableGeometry()
        w, h = 360, 530
        self.setGeometry(screen.width() - w - 24, screen.height() - h - 48, w, h)

        # ─── 레이아웃 ─────────────────────────────────────────────
        central = QWidget()
        central.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCentralWidget(central)

        vbox = QVBoxLayout(central)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        self.title_bar = TitleBar(self)
        vbox.addWidget(self.title_bar)

        # 타이틀바 투명도 효과 (마우스 오버 시에만 보이게)
        self.title_opacity = QGraphicsOpacityEffect(self.title_bar)
        self.title_opacity.setOpacity(0.0)
        self.title_bar.setGraphicsEffect(self.title_opacity)

        self.view = QWebEngineView()
        self.view.page().setBackgroundColor(QColor(Qt.GlobalColor.transparent))

        # YouTube IFrame API 작동에 필요한 WebEngine 설정
        settings = self.view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False)
        settings.setAttribute(QWebEngineSettings.WebAttribute.AllowRunningInsecureContent, True)

        # 로컬 서버에서 호스팅된 URL로 로드 (동적 포트 할당)
        self.view.load(QUrl(f"http://localhost:{self.server_port}/player.html"))
        vbox.addWidget(self.view)

        # ─── 시스템 트레이 ────────────────────────────────────────
        self._setup_tray()

    def _setup_tray(self):
        icon = make_icon("#a78bfa")
        self.tray = QSystemTrayIcon(icon, self)
        self.tray.setToolTip("YouTube Widget")

        menu = QMenu()

        # 재생 / 정지
        self.action_playpause = menu.addAction("⏸ 일시정지")
        self.action_playpause.triggered.connect(self._toggle_playback)

        menu.addSeparator()
        
        # 위치 고정 (드래그 잠금)
        self.action_lock = menu.addAction("위치 고정")
        self.action_lock.setCheckable(True)
        self.action_lock.setChecked(False)
        self.action_lock.triggered.connect(self._toggle_lock)
        
        # 항상 위 기능
        self.action_ontop = menu.addAction("항상 위")
        self.action_ontop.setCheckable(True)
        self.action_ontop.setChecked(True)
        self.action_ontop.triggered.connect(self._toggle_ontop_from_tray)

        menu.addSeparator()
        action_show = menu.addAction("보이기 / 숨기기")
        action_show.triggered.connect(self._toggle_visibility)
        menu.addSeparator()
        action_quit = menu.addAction("종료")
        action_quit.triggered.connect(self.app.quit)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(
            lambda reason: self._toggle_visibility()
            if reason == QSystemTrayIcon.ActivationReason.Trigger else None
        )
        self.tray.show()

    def _toggle_playback(self):
        """트레이에서 재생/정지 토글 — JS로 플레이어 상태 확인 후 제어"""
        js = """
        (function() {
            if (!player || !player.getPlayerState) return 'no_player';
            var state = player.getPlayerState();
            if (state === YT.PlayerState.PLAYING) {
                player.pauseVideo();
                return 'paused';
            } else {
                player.playVideo();
                return 'playing';
            }
        })();
        """
        def on_result(result):
            if result == 'paused':
                self.action_playpause.setText("▶ 재생")
            elif result == 'playing':
                self.action_playpause.setText("⏸ 일시정지")

        self.view.page().runJavaScript(js, on_result)


    def _toggle_visibility(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self.activateWindow()

    def _toggle_lock(self, checked):
        """트레이 메뉴: 위치 고정 (드래그 불가)"""
        self.is_locked = checked

    def _toggle_ontop_from_tray(self, checked):
        """트레이 메뉴: 항상 위 (상단바 버튼과 상태 동기화)"""
        self.title_bar.pin_btn.setChecked(checked)
        self.title_bar._toggle_pin(checked)

    def enterEvent(self, event):
        """마우스가 창 안으로 들어오면 타이틀바 표시"""
        self.title_opacity.setOpacity(1.0)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """마우스가 창 밖으로 나가면 타이틀바 숨김"""
        self.title_opacity.setOpacity(0.0)
        super().leaveEvent(event)

    def closeEvent(self, event):
        # X 버튼은 숨기기만 (트레이에 남음), 종료는 트레이에서
        event.ignore()
        self.hide()


class LocalHTTPServer:
    """widget.py 실행 시 내부적으로만 돌아가는 백그라운드 웹 서버. 
    YouTube IFrame 정책 우회를 위함"""
    def __init__(self, port=0):
        self.port = port
        self.server = None
        self.thread = None

    def start(self):
        # 현재 파일이 있는 디렉토리를 루트로 설정
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        try:
            Handler = http.server.SimpleHTTPRequestHandler
            # 로그 출력 끄기
            class QuietHandler(Handler):
                def log_message(self, format, *args):
                    pass
                    
            self.server = socketserver.TCPServer(("", self.port), QuietHandler)
            self.port = self.server.server_address[1] # OS가 할당한 실제 동적 포트 번호 저장
            self.thread = threading.Thread(target=self.server.serve_forever)
            self.thread.daemon = True # 메인 프로그램 종료 시 같이 죽음
            self.thread.start()
        except OSError:
            pass 


if __name__ == "__main__":
    # 내장 로컬 웹 서버 시작 (포트 0 = 남는 포트 자동 할당)
    local_server = LocalHTTPServer(port=0)
    local_server.start()

    # WebEngine 미디어 자동재생 & 오디오 제약 해제
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--autoplay-policy=no-user-gesture-required --disable-features=HardwareMediaKeyHandling"

    app = QApplication(sys.argv)
    app.setApplicationName("YouTube Widget")
    app.setQuitOnLastWindowClosed(False)  # 창 닫아도 앱 유지 (트레이 남음)

    window = YouTubeWidget(app, local_server.port)
    window.show()

    sys.exit(app.exec())

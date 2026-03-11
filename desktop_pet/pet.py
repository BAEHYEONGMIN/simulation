import sys
import os
import random
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QSystemTrayIcon, QMenu
from PyQt6.QtCore import Qt, QTimer, QPoint
from PyQt6.QtGui import QPixmap, QMovie, QPainter, QColor, QIcon

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class DesktopPet(QWidget):
    def __init__(self):
        super().__init__()

        # --- 창 설정 (투명, 테두리 없음, 항상 위) ---
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool 
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # 펫의 기본 크기 및 변수
        self.pet_width = 100
        self.pet_height = 100
        self.resize(self.pet_width, self.pet_height)

        # 시작 위치 (전체 모니터 가상 바탕화면 기준 중앙쯤)
        virtual_screen = QApplication.primaryScreen().virtualGeometry()
        self.min_x = virtual_screen.left()
        self.max_x = virtual_screen.right()
        
        # Y축 바닥은 주 모니터 해상도 기준으로 통일 (모니터마다 높이가 다를 수 있으므로)
        primary_screen = QApplication.primaryScreen().geometry()
        self.screen_height = primary_screen.height()
        
        self.x = (self.min_x + self.max_x) // 2
        self.y = self.screen_height - self.pet_height - 50 # 바닥 근처
        self.move(self.x, self.y)

        # 이동 속도
        self.speed_x = random.choice([-2, 2])
        self.speed_y = 0 # 일단 좌우로만 걷게

        # 이미지 라벨 설정
        self.label = QLabel(self)
        self.label.resize(self.pet_width, self.pet_height)
        
        # 임시 이미지 또는 GIF 로드
        self.image_path = resource_path("doro.gif")
        self.direction = "right" if self.speed_x > 0 else "left"
        self.load_image()

        # 이동을 제어하는 타이머 (약 60FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_position)
        self.timer.start(16)
        
        # 상태 변수 및 트레이 설정
        self.is_always_on_top = True
        self._setup_tray()

    def _setup_tray(self):
        # 땜빵 아이콘 생성
        pixmap = QPixmap(16, 16)
        pixmap.fill(QColor(Qt.GlobalColor.transparent))
        painter = QPainter(pixmap)
        painter.setBrush(QColor("#fbbf24")) # 노란색 펫 아이콘
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(0, 0, 16, 16)
        painter.end()
        icon = QIcon(pixmap)

        self.tray = QSystemTrayIcon(icon, self)
        self.tray.setToolTip("Desktop Pet")
        
        menu = QMenu()
        action_toggle = menu.addAction("보이기 / 숨기기")
        action_toggle.triggered.connect(self._toggle_visibility)
        
        menu.addSeparator()
        
        action_ontop = menu.addAction("항상 위")
        action_ontop.setCheckable(True)
        action_ontop.setChecked(True)
        action_ontop.triggered.connect(self._toggle_ontop)
        
        menu.addSeparator()
        
        action_quit = menu.addAction("종료")
        # app에 접근하기 위해 parent나 전역 app 객체 사용
        action_quit.triggered.connect(QApplication.instance().quit)
        
        self.tray.setContextMenu(menu)
        self.tray.show()

    def _toggle_visibility(self):
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self.activateWindow()

    def _toggle_ontop(self, checked):
        self.is_always_on_top = checked
        flags = self.windowFlags()
        if checked:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        if self.isVisible():
            self.show()

    def load_image(self):
        """이미지 로드 (없으면 임시 네모 그리기)"""
        if os.path.exists(self.image_path):
            self.movie = QMovie(self.image_path)
            self.movie.setScaledSize(self.label.size())
            # movie.start()는 하되 label에 직접 setMovie를 하지 않고 수동으로 프레임 렌더링
            self.movie.frameChanged.connect(self._update_frame)
            self.movie.start()

    def _update_frame(self, frame_number):
        """방향에 따라 프레임을 뒤집어서 label에 그림"""
        pixmap = self.movie.currentPixmap()
        if self.direction == "left": # 왼쪽으로 갈 때 뒤집기 (doro 원본이 오른쪽을 보고 있다고 가정)
            from PyQt6.QtGui import QTransform
            transform = QTransform().scale(-1, 1)
            pixmap = pixmap.transformed(transform)
        self.label.setPixmap(pixmap)
        #     # 파일이 없으면 그냥 파란색 네모에 눈 코 입 그려진 느낌으로 땜빵
        #     pixmap = QPixmap(self.pet_width, self.pet_height)
        #     pixmap.fill(QColor(Qt.GlobalColor.transparent))
        #     painter = QPainter(pixmap)
        #     painter.setBrush(QColor(100, 150, 250, 200)) # 반투명 파란색
        #     painter.setPen(Qt.PenStyle.NoPen)
        #     painter.drawRoundedRect(0, 0, self.pet_width, self.pet_height, 20, 20)
        #     
        #     # 눈 그리기 (방향에 따라)
        #     painter.setBrush(QColor(Qt.GlobalColor.black))
        #     if self.direction == "right":
        #         painter.drawEllipse(60, 30, 10, 10)
        #         painter.drawEllipse(80, 30, 10, 10)
        #     else:
        #         painter.drawEllipse(20, 30, 10, 10)
        #         painter.drawEllipse(40, 30, 10, 10)
        #         
        #     painter.end()
        #     self.label.setPixmap(pixmap)

    def update_position(self):
        """매 프레임마다 위치 업데이트"""
        self.x += self.speed_x
        self.y += self.speed_y

        # 화면(멀티모니터 전체 가상영역) 밖으로 나가면 방향 전환 (반전)
        changed_direction = False
        if self.x <= self.min_x:
            self.x = self.min_x
            self.speed_x = abs(self.speed_x)
            changed_direction = True
        elif self.x >= self.max_x - self.pet_width:
            self.x = self.max_x - self.pet_width
            self.speed_x = -abs(self.speed_x)
            changed_direction = True

        # 방향이 바뀌면 이미지 새로고침
        if changed_direction:
            self.direction = "right" if self.speed_x > 0 else "left"
            self.load_image()

        self.move(self.x, self.y)

    # 우클릭으로 종료 기능 (원상 복구)
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            QApplication.instance().quit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False) # 창 닫혀도 트레이 살리기
    pet = DesktopPet()
    pet.show()
    sys.exit(app.exec())

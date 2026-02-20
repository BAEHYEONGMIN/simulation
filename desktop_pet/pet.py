import sys
import os
import random
from PyQt6.QtWidgets import QApplication, QLabel, QWidget
from PyQt6.QtCore import Qt, QTimer, QPoint
from PyQt6.QtGui import QPixmap, QMovie, QPainter, QColor

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

        # 시작 위치 (화면 중앙쯤)
        screen = QApplication.primaryScreen().geometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height()
        self.x = self.screen_width // 2
        self.y = self.screen_height - self.pet_height - 50 # 바닥 근처
        self.move(self.x, self.y)

        # 이동 속도
        self.speed_x = random.choice([-2, 2])
        self.speed_y = 0 # 일단 좌우로만 걷게

        # 이미지 라벨 설정
        self.label = QLabel(self)
        self.label.resize(self.pet_width, self.pet_height)
        
        # 임시 이미지 또는 GIF 로드
        self.image_path = "pet.gif" # 나중에 여기에 구하신 이미지 파일 이름 넣기
        self.direction = "right" if self.speed_x > 0 else "left"
        self.load_image()

        # 이동을 제어하는 타이머 (약 60FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_position)
        self.timer.start(16)

    def load_image(self):
        """이미지 로드 (없으면 임시 네모 그리기)"""
        if os.path.exists(self.image_path):
            self.movie = QMovie(self.image_path)
            self.movie.setScaledSize(self.label.size())
            self.label.setMovie(self.movie)
            self.movie.start()
            # ※ 주의: 방향 전환은 여기서 이미지(QPixmap) 자체를 뒤집는 처리가 필요할 수 있음
        else:
            # 파일이 없으면 그냥 파란색 네모에 눈 코 입 그려진 느낌으로 땜빵
            pixmap = QPixmap(self.pet_width, self.pet_height)
            pixmap.fill(QColor(Qt.GlobalColor.transparent))
            painter = QPainter(pixmap)
            painter.setBrush(QColor(100, 150, 250, 200)) # 반투명 파란색
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(0, 0, self.pet_width, self.pet_height, 20, 20)
            
            # 눈 그리기 (방향에 따라)
            painter.setBrush(QColor(Qt.GlobalColor.black))
            if self.direction == "right":
                painter.drawEllipse(60, 30, 10, 10)
                painter.drawEllipse(80, 30, 10, 10)
            else:
                painter.drawEllipse(20, 30, 10, 10)
                painter.drawEllipse(40, 30, 10, 10)
                
            painter.end()
            self.label.setPixmap(pixmap)

    def update_position(self):
        """매 프레임마다 위치 업데이트"""
        self.x += self.speed_x
        self.y += self.speed_y

        # 화면 밖으로 나가면 방향 전환 (반전)
        changed_direction = False
        if self.x <= 0:
            self.x = 0
            self.speed_x = abs(self.speed_x)
            changed_direction = True
        elif self.x >= self.screen_width - self.pet_width:
            self.x = self.screen_width - self.pet_width
            self.speed_x = -abs(self.speed_x)
            changed_direction = True

        # 방향이 바뀌면 이미지 새로고침
        if changed_direction:
            self.direction = "right" if self.speed_x > 0 else "left"
            self.load_image()

        self.move(self.x, self.y)

    # 우클릭으로 종료 기능
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    pet = DesktopPet()
    pet.show()
    sys.exit(app.exec())

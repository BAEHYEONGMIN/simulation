# 위젯 공통 기능 및 아키텍처 분석

`baemin_project` 디렉토리 내의 위젯들(`youtube_widget`, `taskbar_widget`, `desktop_pet`)을 순회하며 분석한 결과, 데스크톱 오버레이 위젯으로서 **모든 위젯에 공통으로 적용할 수 있는(또는 이미 적용된) 핵심 기능과 패턴**들을 추출했습니다.

추후 새로운 위젯을 개발할 때 아래의 패턴들을 보일러플레이트(기본 틀)로 삼으면 빠르고 안정적으로 위젯을 제작할 수 있습니다.

---

## 1. 윈도우 OS 데스크톱 오버레이 속성 (필수 공통)
모든 위젯은 바탕화면이나 다른 창 위에 자연스럽게 녹아들기 위해 다음 `PyQt6` 플래그와 속성을 공통으로 사용합니다.

```python
# 1. 프레임(테두리/타이틀바) 제거
# 2. 항상 최상단 위 노출
# 3. 작업표시줄 아이콘 숨김 (Tool 속성)
self.setWindowFlags(
    Qt.WindowType.FramelessWindowHint |
    Qt.WindowType.WindowStaysOnTopHint |
    Qt.WindowType.Tool
)

# 4. 앱 배경 스크린 투명화 (HTML/이미지의 투명 영역이 윈도우 바탕화면을 투과하게 함)
self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
```

## 2. 마우스 드래그를 통한 창 이동 (Drag to Move)
테두리(타이틀바)가 없기 때문에 사용자가 창을 이동시킬 수 있도록 마우스 이벤트를 오버라이딩하는 기능이 공통으로 필요합니다. (`desktop_pet`은 자동 이동하지만, 수동 드래그가 추가되면 좋습니다.)

```python
def mousePressEvent(self, event):
    if event.button() == Qt.MouseButton.LeftButton:
        self._drag_pos = event.globalPosition().toPoint() - self.pos()
        event.accept()

def mouseMoveEvent(self, event):
    if self._drag_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
        self.move(event.globalPosition().toPoint() - self._drag_pos)
        event.accept()

def mouseReleaseEvent(self, event):
    self._drag_pos = None
    event.accept()
```

## 3. 시스템 트레이(System Tray) 연동 방식
작업표시줄(Taskbar)에서 앱 아이콘이 숨겨져 있으므로(`Tool` 플래그), 백그라운드 제어나 종료를 위해 시스템 트레이 아이콘이 필수적입니다. (`youtube_widget`, `taskbar_widget` 적용됨)

* **보이기 / 숨기기 토글**: 위젯을 잠시 가리거나 다시 띄우는 기능
* **항상 위 표출 토글**: 일반 창 모드와 최상단 모드를 전환
* **위치 고정 토글**: 마우스 드래그 기능을 일시적으로 잠그는 기능
* **종료**: 앱 강제 종료 탈출구

## 4. 항상 위(Always On Top) 플래그 안전 토글 로직
"항상 위" 기능을 껐다 켤 때, 기존 윈도우 속성(Frameless 등)이 날아가는 버그를 막기 위해 비트 연산자로 안전하게 토글하는 로직입니다.

```python
def toggle_ontop(self, is_checked):
    flags = self.windowFlags()
    if is_checked:
        flags |= Qt.WindowType.WindowStaysOnTopHint
    else:
        flags &= ~Qt.WindowType.WindowStaysOnTopHint
    self.setWindowFlags(flags)
    self.show() # Windows 환경에서 플래그 변경 시 창 숨김 현상 방지용
```

## 5. (선택형 공통) HTML/WebEngine 기반 UI 렌더링
`youtube_widget`과 `taskbar_widget`처럼 화려하고 세밀한 UI(CSS/애니메이션)나 외부 웹 API(Youtube IFrame 등)가 필요한 경우 공통으로 사용되는 아키텍처입니다.

1. **QWebEngineView 사용**: PyQt의 기본 위젯 대신 웹 엔진 뷰를 얹어 투명하게 만듦.
2. **Local HTTP Server 모듈화**: `file://` 로드 시 발생하는 CORS 및 보안 제약을 우회하기 위해, 파이썬 내장 `http.server`를 백그라운드 데몬 스레드로 띄움 (자동 포트 `0` 할당).
3. **Chromium 플래그**: 미디어 자동 재생 등을 위해 `QTWEBENGINE_CHROMIUM_FLAGS` 환경 변수에 `--autoplay-policy=no-user-gesture-required` 옵션 부여.

---

### 💡 요약: "위젯 코어(Widget Core)" 클래스 분리 제안
현재 3개의 위젯이 각자 폴더에서 비슷한 기능(투명, 시스템 트레이, 드래그 이동)을 중복으로 구현하고 있습니다. 추후 확장을 고려한다면, `BaseWidget(QMainWindow)` 같은 부모 클래스를 하나 만들어 **플래그 설정, 드래그 로직, 트레이 아이콘 세팅**을 몰아두고 자식 위젯들이 이를 상속받아 사용하는 구조로 리팩토링할 수 있습니다.

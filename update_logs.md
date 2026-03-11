# 위젯 애플리케이션 트러블슈팅 및 해결 레퍼런스

## 1. PyInstaller 빌드 시 `pkgutil` 등 모듈 누락 에러 (ModuleNotFoundError)
* **발생 문제**: 
  PyInstaller로 exe 파일을 만들고 실행했을 때 `Failed to execute script taskbar due to unhandled exception: No module named 'pkgutil'` 같은 에러가 나면서 켜지지 않음. 파이썬 환경을 옮기거나 다른 컴퓨터에서 빌드할 때 종종 발생.
* **해결 방법**: 
  PyInstaller가 의도치 않게 기본 모듈들을 불필요하다고 판단해 빼버리는 캐싱 동작 때문입니다. 빌드 시 `--hidden-import <모듈명>` 인자를 명시적으로 추가하여 강제로 패키징에 포함시키고, 찌꺼기 방지를 위해 `--clean` 옵션을 함께 사용하면 해결됩니다.
  * *적용 예시*: `pyinstaller --clean --noconsole --onefile --name ... --hidden-import pkgutil script.py`

## 2. PyQt6 웹엔진 DLL 관련 ImportError 에러
* **발생 문제**: 
  파이썬 코드를 실행하거나 빌드하려 할 때 `ImportError: DLL load failed while importing QtCore: 지정된 프로시저를 찾을 수 없습니다.` 에러가 나오면서 스크립트 자체가 죽어버림.
* **해결 방법**: 
  윈도우 OS의 C++ 배포 패키지 버전이 최신 PyQt6 라이브러리(ex. 6.10.x 대역)가 요구하는 빌드 환경과 맞지 않아서 일어나는 문제입니다. 이를 해결하기 위해 양쪽 PC에서 모두 호환성이 가장 뛰어난 LTS/안정화 버전 대역(6.4.x)으로 의존성 라이브러리를 일괄 다운그레이드 및 통일하여 완전히 해결했습니다.
  * *통일한 버전*: `PyQt6==6.4.2`, `PyQt6-WebEngine==6.4.0`, `PyQt6-Qt6==6.4.3` 등 (`requirements.txt`에 명시)

## 3. "항상 위(WindowStaysOnTopHint)" 옵션이 윈도우 작업표시줄이나 전체화면 게임에 먹히고 밀려버리는 현상
* **발생 문제**: 
  위젯을 윈도우 하단 작업 표시줄 근처에 두고 쓰려고 하는데, PyQt6에서 기본적으로 제공하는 "항상 위" 옵션을 사용했음에도 불구하고 윈도우(OS) 작업표시줄 쪽을 클릭하거나 전체화면(창 모드 등) 게임을 실행하면 위젯이 뒤쪽으로 밀려가서 아예 사라져버림.
* **해결 방법**: 
  윈도우 OS 자체적으로 Taskbar와 전체화면 게임 등에 Z-Order 최상위(TOPMOST) 권한을 독점적으로 강제로 새로고침해버리는 정책이 있어서 발생합니다. 이를 뚫어버리기 위해 파이썬 내에서 C++의 Windows API(Win32 `ctypes`)를 직접 호출하는 방식의 고성능 폭격 로직 세 가지를 융합 적용했습니다.
  1. **Z-Order 폭격**: 0.05초(50ms)마다 OS에 "내 위젯을 최상위 캐시에서 빼라(NOTOPMOST) -> 다시 즉시 가장 위로 올려라(TOPMOST)" 명령을 쉬지 않고 내리도록 타이머 적용. 이렇게 OS의 최상위 스케줄러를 무조건 덮어씌워버림.
  2. **시스템 호출 강탈**: `ctypes.windll.user32.BringWindowToTop(hwnd)` API를 호출해서 OS 커널에 제일 앞단 화면으로 강제 소환해버림.
  3. **Win+D최소화 방어**: 윈도우 하단 바의 맨 우측 바탕화면 보기를 눌렀을 때 억지로 최소화 당하는 이벤트를 낚아채서 0초 만에 `showNormal()`로 복구시킴.

## 4. PyQt6 QWebEngineView에서 YouTube 오디오가 볼륨 믹서에 전혀 안 나타나는 현상

* **발생 문제**:
  유튜브 썸네일과 재생 UI는 정상적으로 불러오는데, 실제 오디오가 전혀 재생되지 않으며 윈도우 볼륨 믹서에도 앱이 표시되지 않음.

* **원인**:
  1. **Chromium 보안 정책**: PyQt6이 내부적으로 사용하는 Chromium 엔진은 최신 YouTube IFrame API가 `file://` 프로토콜(로컬 파일) 기반에서 실행될 때 CORS(Cross-Origin) 보안 정책을 이유로 오디오 스트림 자체를 원천 차단합니다.
  2. **자동재생(Autoplay) 차단**: Chromium 기본 설정상 사용자 직접 상호작용 없이 오디오가 자동으로 재생되는 것을 막습니다.

* **해결 방법**:
  1. **내장 HTTP 로컬 서버 탑재**: `file://` 대신 `http://localhost:{port}/player.html`로 파일을 서빙하도록 파이썬 표준 라이브러리(`http.server`, `socketserver`)를 이용한 `LocalHTTPServer` 클래스를 위젯 코드 (`widget.py`, `taskbar.py`) 내부에 직접 탑재했습니다. 앱 실행 시 자동으로 백그라운드(데몬) 스레드로 켜지고, 앱 종료 시 같이 꺼집니다.
  2. **동적 포트 할당**: 서버 포트를 `8080`같이 고정하면 다른 프로그램과 충돌할 우려가 있어, 파이썬 소켓에 포트 `0`을 전달하여 OS가 빈 포트를 자동으로 할당하게 했습니다.
  3. **Chromium 자동재생 정책 해제**: `QTWEBENGINE_CHROMIUM_FLAGS` 환경변수에 `--autoplay-policy=no-user-gesture-required` 플래그를 추가하여 미디어 자동재생 제약을 제거했습니다.
  4. **iframe 표시 방식 조정**: `#yt-player` CSS를 `opacity: 0.01`, `z-index: -1`, `pointer-events: none`으로 설정하여 화면에는 사실상 보이지 않지만 Chromium 엔진이 완전히 숨겨진 요소로 판단하지 않도록(throttling 방지) 했습니다.

  * *적용 예시* (`widget.py` / `taskbar.py` 공통 진입점):
    ```python
    local_server = LocalHTTPServer(port=0)
    local_server.start()
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--autoplay-policy=no-user-gesture-required"
    # ...
    self.view.load(QUrl(f"http://localhost:{local_server.port}/player.html"))
    ```

---

## 5. Taskbar 특화 위젯 (`taskbar_widget`) 개발 중 발생한 UI 이슈

### 5-1. 라운드 테두리(border-radius)가 의도치 않게 생기는 현상
* **발생 문제**: PyQt6 `QWidget`의 `setStyleSheet()` 속성이 부모로 자식 위젯에 상속되어, 명시적으로 지정하지 않은 자식 위젯에 둥근 테두리가 생기는 현상 반복 발생.
* **해결 방법**: 의도한 위젯에 `border-radius: 0px`를 명시적으로 지정하여 상속을 강제로 차단합니다.

### 5-2. 창 확장 애니메이션 시 하단 본체 부분이 같이 움직이는 현상
* **발생 문제**: 🔼 버튼을 눌러 설정 패널이 위로 열릴 때, `QPropertyAnimation`으로 창 geometry를 변경하는 과정에서 OS 레벨의 렌더링 지연으로 인해 창 전체가 들썩이는 현상 발생.
* **해결 방법**: `QPropertyAnimation` 방식을 완전히 제거하고, `toggle_expand()` 함수 내에서 `self.setGeometry()`를 즉시 호출하도록 변경하여 애니메이션 없이 순간적으로 변환합니다. (들썩임의 원인인 중간 프레임 렌더링 자체를 없앰)

### 5-3. 패널이 닫혔을 때 숨겨진 영역이 클릭을 가로채는 현상
* **발생 문제**: 창 높이가 48px로 줄어들어도 내부 WebEngine 뷰는 전체 영역을 그대로 유지하고 있어, 상단 설정 패널(`expanded-panel`) 영역이 투명하지만 클릭을 가로채는 현상 발생.
* **해결 방법**: CSS 미디어 쿼리(`@media (max-height: 50px)`)를 이용하여 창 높이가 50px 이하로 작아질 때 `.expanded-panel`에 `display: none !important`를 적용하여 DOM에서 완전히 사라지게 처리합니다.

### 5-4. "항상 위" 기능을 토글할 때 기존 창 설정이 날아가면서 먹통이 되는 현상
* **발생 문제**: 시스템 트레이에서 "항상 위" 기능을 껐다 켤 때 `setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, checked)` 방식으로 설정하면 윈도우 내부적인 Z-order와 기존 속성(Frameless 등)이 리셋되면서 창이 윈도우 바 밑으로 숨거나 클릭이 안 되는 버그 발생.
* **해결 방법**: 기존 플래그 값을 가져와 비트 연산(`|=`, `&= ~`)으로 `WindowStaysOnTopHint` 속성만 덮어씌운 뒤, 화면 일시 사라짐 방지를 위해 `show()` 메서드를 재호출하도록 보강했습니다.

### 5-5. 창 확장 시 UI 고정 버튼들이 위로 딸려 올라가는 현상
* **발생 문제**: 창 높이가 늘어날 때 내부 레이아웃의 기준점이 명확하지 않아 상단으로 치우쳐지면서 하단에 있어야 할 버튼(🔼)과 그립(⋮) 텍스트가 위로 빨려 올라가는 문제.
* **해결 방법**: 하단 고정 UI 요소들(버튼, 그립)을 별도의 `bottom_widget(고정 높이 45px)`으로 묶고, 상단 확장 영역은 `addStretch()`를 통해 여유 공간으로 처리하여 전체 높이가 변하더라도 하단 컨트롤 버튼들의 위치가 절대로 이탈하지 않께 Layout을 분리 설계했습니다.

### 5-6. 듀얼(멀티) 모니터 환경에서 옆 모니터로 이동 불가능한 현상
* **발생 문제**: `QApplication.primaryScreen().geometry()`만 사용하면 주 모니터 해상도 밖으로 나가는 좌표를 '화면 밖'으로 인식하여 펫이 옆 모니터로 건너가지 못하고 튕겨 나오는 현상.
* **해결 방법**: `QApplication.primaryScreen().virtualGeometry()`를 사용하여 모든 모니터가 연결된 전체 가상 캔버스 좌표를 기준으로 벽(Boundary)을 설정하여 해결했습니다.

### 5-7. PyInstaller 빌드 후 이미지/리소스 로드 실패 (MEIPASS)
* **발생 문제**: `--onefile`로 패키징 시 외부에 노출되지 않은 이미지(`doro.gif` 등)를 찾지 못해 앱이 죽거나 이미지가 안 나옴.
* **해결 방법**: PyInstaller가 실행 시 임시로 푸는 경로(`sys._MEIPASS`)를 우선적으로 탐색하는 `resource_path()` 헬퍼 함수를 구현하여 해결했습니다.

---

## 6. 외부 리소스 연결 (Google Drive 등) 및 권한 이슈
* **발생 문제**: 구글 드라이브 같은 외부 폴더를 프로젝트 내부에 심볼릭 링크(`mklink /D`)로 연결하려 할 때 '권한 없음' 에러 발생.
* **해결 방법**: 관리자 권한 없이도 생성이 가능하고 호환성이 높은 디렉토리 정션(`mklink /J`) 명령어를 사용하여 해결했습니다.

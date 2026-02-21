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

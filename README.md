# X Bookmark Backup Tool

X(트위터) 북마크의 이미지/동영상 URL과 메타데이터를 수집하고, 원본 미디어를 로컬에 백업하는 Python 도구입니다.

이 프로젝트는 2가지 수집 방식을 지원합니다.
- `LocalVer` (주력): Chrome attach + CDP/DOM 기반 수집
- `OAuth2Ver`: X API(OAuth2) 기반 수집

## 프로젝트 목적

- 북마크 이미지의 로컬 백업 자동화
- 북마크 동영상(video.twimg.com / m3u8)의 로컬 백업 자동화
- 재실행 시 중복 저장 최소화
- 파일명에 메타데이터(작성자/시간/미디어키/트윗ID) 반영

## 주요 기능

1. LocalVer (`X_bookmark_backuptool.py`)
- Chrome 원격 디버깅 attach 방식
- `FULL` / `PERIODIC` 백업 전략
- `IMAGE_ONLY` / `ALL (IMAGE + VIDEO)` 미디어 범위 선택
- `CDP_ONLY` / `SAFE` / `NDJSON_ONLY` / `DEDUPE_ONLY` / `NDJSON_CLEANUP` / `VIDEO_META_REPAIR` 모드
- 중복 키 감지 기반 조기 중단, NDJSON 누적 관리
- 이미지 중복 파일 격리 및 메타 정리 유틸
- `video.twimg.com` URL 수집, m3u8 → mp4 다운로드, 비디오 메타 repair 지원

2. OAuth2Ver (`X_bookmark_backuptool_oAuch2.py`)
- X API Bookmarks 엔드포인트 기반 수집
- `FULL` / `INCREMENTAL` 모드
- `state.json` 기반 재개 토큰 관리
- 별도 Downloader(`Downloader_oAuch2ver.py`)로 이미지 저장

## 기술 스택

- Language: Python 3
- Runtime/OS: Windows 중심
- Browser Automation: Selenium + Chrome DevTools Protocol(performance log)
- Networking: requests + urllib3 Retry
- Data Format: NDJSON, JSON
- 주요 라이브러리: `selenium`, `webdriver-manager`, `requests`, `Pillow`, `tqdm`

## 개발 환경

- 권장 OS: Windows 10/11
- 권장 Python: 3.9.x ~ 3.10.x
- 브라우저: Google Chrome

## 저장소 구성

- `X_bookmark_backuptool.py`: LocalVer 메인 스크립트
- `run_bookmark_tool_local.ps1`, `run_bookmark_tool_local.bat`: LocalVer 실행 보조
- `X_bookmark_backuptool_oAuch2.py`: OAuth2Ver 메타 수집기
- `Downloader_oAuch2ver.py`: OAuth2Ver 메타 기반 다운로드기
- `bookmark_meta_local/`: LocalVer 메타 저장
- `downloaded_images_local/`: LocalVer 이미지/동영상 저장
- `bookmark_meta/`: OAuth2Ver 메타 저장
- `downloaded_images/`: OAuth2Ver 이미지 저장

## 빠른 시작

### 1) LocalVer (권장)

1. Chrome를 원격 디버깅 포트로 실행
```powershell
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="%LOCALAPPDATA%\Google\Chrome\User Data" --profile-directory="Default"
```
2. 위 Chrome에서 `https://x.com/i/bookmarks` 열기
3. 실행
```powershell
python X_bookmark_backuptool.py
```
또는
```powershell
.\run_bookmark_tool_local.ps1
```

동영상까지 백업하려면 실행 후 `Media scope`에서 `ALL (IMAGE + VIDEO)`을 선택합니다. m3u8 동영상 다운로드에는 `ffmpeg.exe`가 필요하며, 보조 실행 스크립트는 아래 경로를 자동으로 PATH에 추가합니다.

```text
.venv\tools\ffmpeg\bin\ffmpeg.exe
```

`ffmpeg.exe`가 시스템 PATH에 이미 있으면 별도 배치 없이 사용할 수 있습니다.

### 2) OAuth2Ver

1. `x_oauth_token.json` 준비
2. `X_bookmark_backuptool_oAuch2.py`의 `CLIENT_ID` 설정
3. 수집/다운로드 실행
```powershell
python X_bookmark_backuptool_oAuch2.py
python Downloader_oAuch2ver.py
```

## 출력 데이터

- LocalVer 메타: `bookmark_meta_local/items.ndjson`
- LocalVer 이미지/동영상: `downloaded_images_local/`
- OAuth2Ver 메타: `bookmark_meta/items.ndjson`, `bookmark_meta/state.json`
- OAuth2Ver 이미지: `downloaded_images/`

## 주의 사항

- OAuth2Ver는 API 플랜/권한/요금 정책 영향을 받을 수 있습니다.
- LocalVer attach 모드는 Chrome 실행 인자(`--remote-debugging-port`)가 필수입니다.
- LocalVer 동영상 백업은 X 페이지의 GraphQL/CDP 응답과 DOM 스캔을 함께 사용하므로, 일시적인 X 로딩 오류가 있으면 repair를 다시 실행하는 것이 유효할 수 있습니다.

## 추가 문서

- LocalVer 상세: [README_localver.md](README_localver.md)
- OAuth2Ver 상세: [README_oauth2ver.md](README_oauth2ver.md)


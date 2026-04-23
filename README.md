# X Bookmark Backup Tool

X(트위터) 북마크의 이미지 URL/메타데이터를 수집하고, 원본 이미지를 로컬에 백업하는 Python 도구입니다.

이 프로젝트는 2가지 수집 방식을 지원합니다.
- `LocalVer` (주력): Chrome attach + CDP/DOM 기반 수집
- `OAuth2Ver`: X API(OAuth2) 기반 수집

## 프로젝트 목적

- 북마크 이미지의 로컬 백업 자동화
- 재실행 시 중복 저장 최소화
- 파일명에 메타데이터(작성자/시간/미디어키/트윗ID) 반영

## 주요 기능

1. LocalVer (`X_bookmark_backuptool.py`)
- Chrome 원격 디버깅 attach 방식
- `FULL` / `PERIODIC` 백업 전략
- `CDP_ONLY` / `SAFE` / `NDJSON_ONLY` / `DEDUPE_ONLY` / `NDJSON_CLEANUP` 모드
- 중복 키 감지 기반 조기 중단, NDJSON 누적 관리
- 이미지 중복 파일 격리 및 메타 정리 유틸

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
- `run_bookmark_tool_oldver.ps1`, `run_bookmark_tool_oldver.bat`: LocalVer 실행 보조
- `X_bookmark_backuptool_oAuch2.py`: OAuth2Ver 메타 수집기
- `Downloader_oAuch2ver.py`: OAuth2Ver 메타 기반 다운로드기
- `bookmark_meta_oldver/`: LocalVer 메타 저장
- `downloaded_images_oldver/`: LocalVer 이미지 저장
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
.\run_bookmark_tool_oldver.ps1
```

### 2) OAuth2Ver

1. `x_oauth_token.json` 준비
2. `X_bookmark_backuptool_oAuch2.py`의 `CLIENT_ID` 설정
3. 수집/다운로드 실행
```powershell
python X_bookmark_backuptool_oAuch2.py
python Downloader_oAuch2ver.py
```

## 출력 데이터

- LocalVer 메타: `bookmark_meta_oldver/items.ndjson`
- LocalVer 이미지: `downloaded_images_oldver/`
- OAuth2Ver 메타: `bookmark_meta/items.ndjson`, `bookmark_meta/state.json`
- OAuth2Ver 이미지: `downloaded_images/`

## 주의 사항

- OAuth2Ver는 API 플랜/권한/요금 정책 영향을 받을 수 있습니다.
- LocalVer attach 모드는 Chrome 실행 인자(`--remote-debugging-port`)가 필수입니다.

## 추가 문서

- LocalVer 상세: [README_localver.md](README_localver.md)
- OAuth2Ver 상세: [README_oauth2ver.md](README_oauth2ver.md)

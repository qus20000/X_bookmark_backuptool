# LocalVer 가이드 (기존 `oldver`)

`X_bookmark_backuptool.py` 기반의 로컬 브라우저 attach 방식입니다.  
현재 주력 사용 버전입니다.

## 파일

- 실행 스크립트: `X_bookmark_backuptool.py`
- 실행 보조: `run_bookmark_tool_oldver.ps1`, `run_bookmark_tool_oldver.bat`
- 메타 출력: `bookmark_meta_oldver/items.ndjson`
- 이미지 출력: `downloaded_images_oldver/`

## 실행 전 준비

1. Windows + Python 3.9~3.10 권장
2. Chrome 실행 (원격 디버깅 포트)
3. X 로그인 상태에서 `https://x.com/i/bookmarks` 접근 가능

Chrome 예시 실행:

```powershell
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="%LOCALAPPDATA%\Google\Chrome\User Data" --profile-directory="Default"
```

## 실행

```powershell
python X_bookmark_backuptool.py
```

또는:

```powershell
.\run_bookmark_tool_oldver.ps1
```

## 모드

- Backup strategy
1. `FULL`: 가능한 전체 키 수집
2. `PERIODIC`: 기존 키를 기준으로 조기 중단

- Select mode
1. `CDP_ONLY`: 빠른 수집/다운로드
2. `SAFE`: 업스크롤로 메타 확정 강화 (중복 환경에 권장)
3. `NDJSON_ONLY`: ndjson 기준 다운로드만 수행
4. `DEDUPE_ONLY`: 이미지 중복 파일 격리
5. `NDJSON_CLEANUP`: ndjson 정규화/중복 정리

## 최근 SAFE/PERIODIC 보수화 포인트

- `PERIODIC_STOP_HIT_STREAK` 도입
- 기존 키 hit가 연속 N회일 때만 하강 조기중단
- 같은 burst에 `new key`가 보이면 hit 연속 카운트 초기화

## 트러블슈팅 (attach 실패)

- `debuggerAddress 127.0.0.1:9222 is not reachable`가 나오면:
1. Chrome가 실제로 `--remote-debugging-port=9222`로 실행됐는지 확인
2. `netstat -ano | findstr :9222` 확인
3. `chrome://version`의 Command Line에 디버그 인자 포함 여부 확인

# LocalVer 가이드

`X_bookmark_backuptool.py` 기반의 로컬 브라우저 attach 방식입니다.  
현재 주력 사용 버전입니다.

## 파일

- 실행 스크립트: `X_bookmark_backuptool.py`
- 실행 보조: `run_bookmark_tool_local.ps1`, `run_bookmark_tool_local.bat`
- 메타 출력: `bookmark_meta_local/items.ndjson`
- 이미지/동영상 출력: `downloaded_images_local/`

## 실행 전 준비

1. Windows + Python 3.9~3.10 권장
2. Chrome 실행 (원격 디버깅 포트)
3. X 로그인 상태에서 `https://x.com/i/bookmarks` 접근 가능
4. 동영상 다운로드를 사용할 경우 `ffmpeg.exe` 준비

Chrome 예시 실행:

```powershell
"C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="%LOCALAPPDATA%\Google\Chrome\User Data" --profile-directory="Default"
```

ffmpeg 권장 배치 경로:

```text
.venv\tools\ffmpeg\bin\ffmpeg.exe
```

`run_bookmark_tool_local.ps1`와 `run_bookmark_tool_local.bat`는 위 경로에 `ffmpeg.exe`가 있으면 실행 시 자동으로 PATH에 추가합니다. 시스템 PATH에 `ffmpeg.exe`가 이미 등록되어 있어도 됩니다.

## 실행

```powershell
python X_bookmark_backuptool.py
```

또는:

```powershell
.\run_bookmark_tool_local.ps1
```

## 모드

- Media scope
1. `IMAGE_ONLY`: 이미지 백업만 수행
2. `ALL`: 이미지 + 동영상(video.twimg.com) 백업 수행

- Backup strategy
1. `FULL`: 가능한 전체 키 수집
2. `PERIODIC`: 기존 키를 기준으로 조기 중단

- Select mode
1. `CDP_ONLY`: 빠른 수집/다운로드
2. `SAFE`: 업스크롤로 메타 확정 강화 (중복 환경에 권장)
3. `NDJSON_ONLY`: ndjson 기준 다운로드만 수행
4. `DEDUPE_ONLY`: 이미지 중복 파일 격리
5. `NDJSON_CLEANUP`: ndjson 정규화/중복 정리
6. `VIDEO_META_REPAIR`: 기존 `items.ndjson`의 동영상 메타 보강

## 동영상 백업

`Media scope`에서 `ALL`을 선택하면 이미지와 함께 동영상 URL도 수집합니다.

- 수집 대상: `video.twimg.com`의 `m3u8` / mp4 계열 URL
- 저장 메타: `bookmark_meta_local/items.ndjson`
- 다운로드 위치: `downloaded_images_local/`
- 파일명: 작성자, 작성시간, media_key, tweet_id 기반 결정적 이름
- m3u8 다운로드: 가능한 variant를 선택한 뒤 ffmpeg로 mp4 파일 생성

동영상 CDN URL만으로는 원 트윗 작성자/작성시간을 알 수 없는 경우가 많습니다. 이 도구는 CDP 성능 로그와 GraphQL 응답을 파싱해 `video media_key -> tweet_id/author/created_at`을 복구합니다.

## VIDEO_META_REPAIR

`VIDEO_META_REPAIR`는 이미 저장된 `bookmark_meta_local/items.ndjson`의 동영상 행을 다시 보강합니다.

주요 동작:

- 전체 북마크 타임라인을 다시 내려가며 CDP/GraphQL 응답 수집
- `ext_tw_video_<id>`와 `amplify_video_<id>` alias 매칭
- 누락된 `tweet_id` 보강
- 기록된 최하단 yOffset/scrollHeight 기준으로 author/date 전체 업스크롤 backfill
- X 가상 스크롤이 중간에서 멈출 때 descent rescue 시도

`ALL` 모드 수집 완료 후에도 다운로드 직전에 repair 여부를 선택할 수 있습니다. 이 경우 이미 로드한 타임라인/CDP/GraphQL 캐시를 재사용하므로 별도 full descent 없이 동영상 메타를 보강합니다.

## 최근 SAFE/PERIODIC 보수화 포인트

- `PERIODIC_STOP_HIT_STREAK` 도입
- 기존 키 hit가 연속 N회일 때만 하강 조기중단
- 같은 burst에 `new key`가 보이면 hit 연속 카운트 초기화

## 최근 동영상 백업/repair 보강 포인트

- `ALL (IMAGE + VIDEO)` 미디어 범위 추가
- `video.twimg.com` media_key 정규화
- GraphQL/CDP 기반 동영상 `tweet_id` / `author` / `created_at` 복구
- m3u8 → mp4 다운로드 지원
- 다운로드 직전 repair 분기 추가
- author/date backfill을 실제 descent 높이에 맞춰 동적으로 수행

## 트러블슈팅 (attach 실패)

- `debuggerAddress 127.0.0.1:9222 is not reachable`가 나오면:
1. Chrome가 실제로 `--remote-debugging-port=9222`로 실행됐는지 확인
2. `netstat -ano | findstr :9222` 확인
3. `chrome://version`의 Command Line에 디버그 인자 포함 여부 확인

## 트러블슈팅 (동영상)

- `ffmpeg_not_found`가 나오면:
1. `.venv\tools\ffmpeg\bin\ffmpeg.exe`가 있는지 확인
2. 또는 `ffmpeg.exe`를 시스템 PATH에 등록
3. 보조 실행 스크립트를 다시 실행해 `[INFO] ffmpeg path added` 메시지 확인

- 동영상 `author` / `created_at`이 비어 있으면:
1. 6번 `VIDEO_META_REPAIR` 실행
2. X 페이지에 `알 수 없는 오류가 발생했습니다. 새로고침 하시겠습니까?` 상태가 보이면 새로고침 후 재실행
3. `AuthorBackfill ... found=..., remain=...` 로그로 보강 결과 확인



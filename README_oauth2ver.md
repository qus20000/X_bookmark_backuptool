# OAuth2Ver 가이드

`X_bookmark_backuptool_oAuch2.py` + `Downloader_oAuch2ver.py` 기반의 X API 방식입니다.

주의:
- API 플랜/권한에 따라 호출 제한이 있을 수 있습니다.
- 사용량에 따른 과금 가능성이 있으므로 비용 정책을 먼저 확인하세요.

## 파일

- 수집기: `X_bookmark_backuptool_oAuch2.py`
- 다운로드기: `Downloader_oAuch2ver.py`
- 토큰: `x_oauth_token.json`
- 메타 출력: `bookmark_meta/items.ndjson`, `bookmark_meta/state.json`
- 이미지 출력: `downloaded_images/`

## 실행 순서

1. OAuth2 토큰 준비 (`x_oauth_token.json`)
2. `X_bookmark_backuptool_oAuch2.py`의 `CLIENT_ID` 설정
3. 수집 실행
4. 다운로드 실행

```powershell
python X_bookmark_backuptool_oAuch2.py
python Downloader_oAuch2ver.py
```

## 수집 모드

1. `FULL`
- `state.json`의 `full_next_token`을 유지하며 이어받기

2. `INCREMENTAL`
- 항상 최신부터 시작
- `full_next_token`은 보호(갱신/삭제 안 함)
- 연속 N페이지 신규 0개면 종료

## 중복 기준

- 기본 중복 제거는 `media_key` 기준입니다.

## 권장 사용 시나리오

- 비용/권한 이슈 없이 API 중심 자동화를 하려는 경우: `OAuth2Ver`
- 비용 없이 브라우저 세션 기반으로 안정 운용하려는 경우: `LocalVer` (권장 주력)

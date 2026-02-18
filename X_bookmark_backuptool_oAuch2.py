# collect_bookmarks_meta.py
# X Bookmarks -> media(meta) collector (FULL / INCREMENTAL)
#
# 저장 포맷
# - items.ndjson : 한 줄에 JSON 1개 (photo 1장 = 1레코드)
# - state.json   : FULL 재개 토큰(full_next_token) + 누적 통계
#
# 모드 정책 (중요)
# - FULL:
#   - pagination(next_token)을 state.json의 "full_next_token"으로 저장/유지
#   - 중단(402/강제종료) 후 재실행 시 full_next_token부터 이어받기
# - INCREMENTAL:
#   - state.json에 남아있는 어떤 next_token도 무시하고 "항상 최신부터" 시작
#   - 실행 중에도 full_next_token을 절대 갱신/삭제하지 않음 (FULL 재개 정보 보호)
#   - 연속 N페이지 신규 0개면 종료
#
# 중복 제거 기준
# - media_key 기준
#
# 주의
# - user.fields=username 가 플랜/권한에 따라 안 올 수 있음
#   이 경우 author는 "@uid_<author_id>" 로 저장하며,
#   트윗 링크는 https://x.com/i/web/status/<tweet_id> 형태로 항상 생성 가능

import os
import re
import json
import time
from typing import Dict, List, Optional, Tuple

import requests


# ---------------------------------------------------------------------------
# 사용자 설정
# ---------------------------------------------------------------------------
CLIENT_ID = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # 여기에 너의 OAuth2 Client ID 넣기

TOKEN_PATH = "x_oauth_token.json"

OUT_DIR = "bookmark_meta"
ITEMS_PATH = os.path.join(OUT_DIR, "items.ndjson")
STATE_PATH = os.path.join(OUT_DIR, "state.json")

# 안정성 때문에 50 권장(너 환경에서 100은 next_token이 끊기는 케이스가 있었음)
PAGE_SIZE = 50

FORCE_NAME_ORIG = True
STOP_IF_NO_NEW_PAGES = 3          # INCREMENTAL: 연속 N페이지 신규 0개면 종료
PAGE_DELAY_S = 0.15               # API 호출 간 딜레이

INCLUDE_VIDEOS = False            # False면 photo만 저장(권장). True면 video도 저장


# ---------------------------------------------------------------------------
# X API endpoints
# ---------------------------------------------------------------------------
API_BASE = "https://api.x.com/2"
TOKEN_URL = "https://api.x.com/2/oauth2/token"


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------
def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Token refresh
# ---------------------------------------------------------------------------
def refresh_access_token(token_path: str) -> Dict:
    tok = _load_json(token_path)
    refresh_token = tok.get("refresh_token")
    if not refresh_token:
        raise RuntimeError("refresh_token이 없습니다. offline.access 스코프로 PKCE를 다시 수행해야 합니다.")

    data = {
        "grant_type": "refresh_token",
        "client_id": CLIENT_ID,
        "refresh_token": refresh_token,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    r = requests.post(TOKEN_URL, data=data, headers=headers, timeout=30)
    if r.status_code >= 400:
        raise RuntimeError(f"refresh failed: HTTP {r.status_code} {r.text}")

    new_tok = r.json()

    merged = dict(tok)
    for k, v in new_tok.items():
        merged[k] = v

    _save_json(token_path, merged)
    return merged


# ---------------------------------------------------------------------------
# API call helpers
# ---------------------------------------------------------------------------
def x_get(url: str, access_token: str, params: Optional[Dict] = None) -> Dict:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "bookmark-meta-collector/1.0",
    }
    r = requests.get(url, headers=headers, params=params, timeout=30)
    if r.status_code >= 400:
        raise requests.HTTPError(f"HTTP {r.status_code} {r.text}", response=r)
    return r.json()

def x_get_with_refresh(url: str, token_path: str, params: Optional[Dict] = None) -> Dict:
    tok = _load_json(token_path)
    access_token = tok.get("access_token", "")
    if not access_token:
        raise RuntimeError("x_oauth_token.json에 access_token이 없습니다. PKCE로 토큰을 먼저 발급하세요.")

    try:
        return x_get(url, access_token, params=params)
    except requests.HTTPError as e:
        resp = e.response
        if resp is not None and resp.status_code == 401:
            new_tok = refresh_access_token(token_path)
            return x_get(url, new_tok["access_token"], params=params)
        raise


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def ensure_out_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

def canon_media_url(url: str) -> str:
    if not url:
        return url
    if "name=" in url:
        return re.sub(r"name=[^&]+", "name=orig", url)
    if "?" in url:
        return url + "&name=orig"
    return url + "?name=orig"

def normalize_time(ts: str) -> str:
    return (ts or "").replace(":", "").replace("Z", "")

def build_tweet_url(tweet_id: str) -> str:
    return f"https://x.com/i/web/status/{tweet_id}"


# ---------------------------------------------------------------------------
# State / items load
# ---------------------------------------------------------------------------
def load_state() -> Dict:
    """
    state.json 스키마(현재):
      - full_next_token: FULL 재개용 pagination_token
      - full_pages_done, full_items_written
      - last_full_run_ts
      - last_incremental_run_ts
    과거 버전 호환:
      - next_token 이 있으면 full_next_token으로 1회 마이그레이션
    """
    if not os.path.exists(STATE_PATH):
        return {
            "full_next_token": None,
            "full_pages_done": 0,
            "full_items_written": 0,
            "last_full_run_ts": None,
            "last_incremental_run_ts": None,
        }

    state = _load_json(STATE_PATH)

    # 과거 키(next_token) 마이그레이션: FULL 재개 토큰으로만 취급
    if "full_next_token" not in state and "next_token" in state:
        state["full_next_token"] = state.get("next_token")
        # 기존 next_token은 남겨도 되고 지워도 되지만, 혼란 방지를 위해 제거
        try:
            del state["next_token"]
        except Exception:
            pass
        _save_json(STATE_PATH, state)

    # 누락 키 보강
    state.setdefault("full_next_token", None)
    state.setdefault("full_pages_done", 0)
    state.setdefault("full_items_written", 0)
    state.setdefault("last_full_run_ts", None)
    state.setdefault("last_incremental_run_ts", None)
    return state

def save_state(state: Dict) -> None:
    _save_json(STATE_PATH, state)

def load_seen_media_keys(items_path: str) -> set:
    if not os.path.exists(items_path):
        return set()

    seen = set()
    with open(items_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                mk = obj.get("media_key")
                if mk:
                    seen.add(mk)
            except Exception:
                continue
    return seen

def append_items(items: List[Dict]) -> None:
    if not items:
        return
    with open(ITEMS_PATH, "a", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Mode selector
# ---------------------------------------------------------------------------
def select_mode() -> str:
    print("\n============================================================")
    print("Select mode:")
    print("  1) FULL        : 전체 북마크 끝까지 수집 (next_token 끝날 때까지)")
    print(f"  2) INCREMENTAL : 최근 추가분만 수집 (연속 {STOP_IF_NO_NEW_PAGES}페이지 신규 0개면 종료)")
    print("============================================================")

    while True:
        sel = input("Enter 1 or 2: ").strip()
        if sel == "1":
            return "FULL"
        if sel == "2":
            return "INCREMENTAL"


# ---------------------------------------------------------------------------
# Bookmarks page fetch + join
# ---------------------------------------------------------------------------
def get_me_user_id() -> str:
    me = x_get_with_refresh(f"{API_BASE}/users/me", TOKEN_PATH)
    return me["data"]["id"]

def fetch_bookmarks_page(user_id: str, next_token: Optional[str]) -> Tuple[Dict, Optional[str]]:
    params = {
        "max_results": PAGE_SIZE,
        "expansions": "author_id,attachments.media_keys",
        "tweet.fields": "created_at,author_id,attachments",
        "media.fields": "media_key,type,url,preview_image_url,variants",
        "user.fields": "username",
    }
    if next_token:
        params["pagination_token"] = next_token

    page = x_get_with_refresh(f"{API_BASE}/users/{user_id}/bookmarks", TOKEN_PATH, params=params)

    meta = page.get("meta", {}) or {}
    nt = meta.get("next_token")
    return page, nt

def extract_items_from_page(page: Dict) -> List[Dict]:
    tweets = page.get("data", []) or []
    includes = page.get("includes", {}) or {}
    media_list = includes.get("media", []) or []
    users_list = includes.get("users", []) or []

    users_by_id: Dict[str, str] = {}
    for u in users_list:
        uid = u.get("id")
        username = u.get("username")
        if uid and username:
            users_by_id[uid] = username

    media_by_key: Dict[str, Dict] = {}
    for m in media_list:
        mk = m.get("media_key")
        if mk:
            media_by_key[mk] = m

    out: List[Dict] = []

    for tw in tweets:
        tweet_id = tw.get("id") or ""
        if not tweet_id:
            continue

        created_at = tw.get("created_at") or ""
        created_norm = normalize_time(created_at) if created_at else ""

        author_id = tw.get("author_id") or ""
        username = users_by_id.get(author_id)
        if username:
            author = f"@{username}"
        else:
            author = f"@uid_{author_id}" if author_id else "@unknown"

        tw_url = build_tweet_url(tweet_id)

        attachments = tw.get("attachments", {}) or {}
        media_keys = attachments.get("media_keys", []) or []

        for mk in media_keys:
            m = media_by_key.get(mk)
            if not m:
                continue

            mtype = m.get("type")
            if mtype == "photo":
                url = m.get("url") or ""
                if not url:
                    continue
                if FORCE_NAME_ORIG:
                    url = canon_media_url(url)

                out.append({
                    "tweet_id": tweet_id,
                    "tweet_url": tw_url,
                    "author": author,
                    "created_at": created_at,
                    "created_at_norm": created_norm,
                    "media_key": mk,
                    "media_type": "photo",
                    "url": url,
                })

            elif INCLUDE_VIDEOS and mtype in ("video", "animated_gif"):
                variants = m.get("variants", []) or []
                mp4s = [v for v in variants if (v.get("content_type") == "video/mp4" and v.get("url"))]
                if mp4s:
                    mp4s.sort(key=lambda v: int(v.get("bit_rate", 0)), reverse=True)
                    vurl = mp4s[0]["url"]
                    out.append({
                        "tweet_id": tweet_id,
                        "tweet_url": tw_url,
                        "author": author,
                        "created_at": created_at,
                        "created_at_norm": created_norm,
                        "media_key": mk,
                        "media_type": mtype,
                        "url": vurl,
                    })

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not CLIENT_ID or CLIENT_ID.strip() == "여기에_너의_OAuth2_Client_ID":
        raise RuntimeError("코드 상단 CLIENT_ID를 네 OAuth2 Client ID로 바꿔야 합니다.")
    if not os.path.exists(TOKEN_PATH):
        raise RuntimeError(f"토큰 파일이 없습니다: {TOKEN_PATH}")

    ensure_out_dir()

    mode = select_mode()
    state = load_state()

    user_id = get_me_user_id()
    seen_media_keys = load_seen_media_keys(ITEMS_PATH)

    # FULL/INCREMENTAL 시작 토큰 결정
    if mode == "FULL":
        # FULL만 재개 토큰을 사용한다.
        next_token = state.get("full_next_token")
    else:
        # INCREMENTAL은 어떤 상황에서도 state 토큰을 무시하고 최신부터 시작한다.
        next_token = None

    full_pages_done = int(state.get("full_pages_done") or 0)
    full_items_written = int(state.get("full_items_written") or 0)

    print(f"message: MODE={mode}, my_id={user_id}")
    print(f"message: loaded_seen_media_keys={len(seen_media_keys)}")
    print(f"message: resume_full_next_token={'YES' if (mode == 'FULL' and next_token) else 'NO'}")

    no_new_pages_seq = 0
    page_idx = 0

    while True:
        page_idx += 1

        try:
            page, nt = fetch_bookmarks_page(user_id=user_id, next_token=next_token)
        except requests.HTTPError as e:
            resp = e.response
            if resp is not None and resp.status_code == 402:
                # FULL은 이미 "성공한 페이지 단위"로 state가 저장되어 있어 이어받기 가능.
                # INCREMENTAL은 next_token을 저장하지 않으므로, 충전 후 재실행하면 최신부터 다시 훑는다.
                print(f"message: HTTP 402 Payment Required. stop now. detail={resp.text[:200]}")
                break
            raise

        items = extract_items_from_page(page)

        new_items: List[Dict] = []
        for it in items:
            mk = it.get("media_key")
            if not mk:
                continue
            if mk in seen_media_keys:
                continue
            seen_media_keys.add(mk)
            new_items.append(it)

        if new_items:
            append_items(new_items)
            if mode == "FULL":
                full_items_written += len(new_items)
            no_new_pages_seq = 0
        else:
            no_new_pages_seq += 1

        if mode == "FULL":
            full_pages_done += 1

            # FULL만 full_next_token을 갱신한다.
            state["full_next_token"] = nt
            state["full_pages_done"] = full_pages_done
            state["full_items_written"] = full_items_written
            state["last_full_run_ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_state(state)

            print(
                f"message: page={page_idx}, new_items={len(new_items)}, full_items_written={full_items_written}, "
                f"next_token={'YES' if nt else 'NO'}, no_new_seq={no_new_pages_seq}"
            )

            if not nt:
                print("message: reached end (no next_token).")
                break

        else:
            # INCREMENTAL은 full_next_token을 절대 건드리지 않는다.
            state["last_incremental_run_ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
            save_state(state)

            print(
                f"message: page={page_idx}, new_items={len(new_items)}, "
                f"next_token={'YES' if nt else 'NO'}, no_new_seq={no_new_pages_seq}"
            )

            if no_new_pages_seq >= STOP_IF_NO_NEW_PAGES:
                print("message: incremental stop (no new items for consecutive pages).")
                break

            # INCREMENTAL은 nt를 저장하지 않지만, 다음 페이지 조회를 위해 런타임 변수로는 사용 가능.
            # 단, 조기 종료 조건이 있기 때문에 굳이 깊게 내려가지 않게 된다.
            next_token = nt
            if not next_token:
                break

        next_token = nt
        if not next_token:
            break

        time.sleep(PAGE_DELAY_S)

    print("message: done.")
    print(f"message: out_dir={OUT_DIR}")
    print(f"message: items_path={ITEMS_PATH}")
    print(f"message: state_path={STATE_PATH}")


if __name__ == "__main__":
    main()

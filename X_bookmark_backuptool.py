# X bookmark backuptool by qus20000
# Windows / Python 3.9.x ~ 3.10.x 권장
#
# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------
# 본 스크립트는 X(트위터) 북마크 페이지에서 이미지/동영상 URL을 수집하고
# 로컬 메타(ndjson)와 다운로드 파일을 생성하는 도구입니다.
#
# 공통 동작:
# - 1) 완전 하강(북마크 페이지에서 스크롤 최하단까지 탐지)"방식을 통해 모든 트윗이 로드된 상태를 확보 
#   2) CDP(Network.* 성능 로그)에서 pbs.twimg.com/media/... 및 video.twimg.com/... 요청을 추출 → Target 집합 구성
#   3) 모드에 따라:
#      - CDP_ONLY: 하강 직후 바로 다운로드(업스크롤 없음). 하강 중 IO(IntersectionObserver)로
#                  수집해둔 메타(업로더/시간)가 있으면 파일명에 반영, 없으면 MEDIA_KEY로 저장.
#      - SAFE:    업스크롤 하며 DOM+IO로 Target을 하나씩 “확정”하고 메타를 최대한 채움.
#                  확정되지 않은(Target - Current) 키는 “메타 없이”라도 URL로 저장.
#      - NDJSON_ONLY: 기존 bookmark_meta_local/items.ndjson 기준으로 다운로드만 수행.
#      - VIDEO_META_REPAIR: 이미 수집된 video.twimg.com 항목의 tweet_id/author/created_at 보강만 수행.
#
# 미디어 범위:
#   - IMAGE_ONLY: 기존 이미지 백업 흐름만 수행.
#   - ALL: 이미지 + 동영상(video.twimg.com)을 수집/저장/다운로드 대상에 포함.
#   - 동영상은 m3u8 master/variant URL을 media_key(ext_tw_video/amplify_video + id)로 정규화한다.
#
# 동영상 메타 복구 정책:
#   - X 동영상 CDN URL은 원 트윗 URL/작성자/작성시간을 직접 포함하지 않는 경우가 많다.
#   - CDP GraphQL 응답 body를 스크롤 중/수집 후 파싱하여 video media_key → tweet_id/author/created_at 매핑을 만든다.
#   - ext_tw_video_<id>와 amplify_video_<id>는 같은 numeric id alias로 간주하여 매칭 누락을 줄인다.
#   - GraphQL/CDP로 부족한 author/date는 기록된 최하단 yOffset/scrollHeight 기준으로 전체 업스크롤 backfill을 수행한다.
#   - 다운로드 직전 repair 선택 시, 이미 로드한 timeline/CDP/GraphQL 캐시를 재사용하여 별도 full descent 없이 보강한다.
#
# 동영상 다운로드 정책:
#   - video.twimg.com m3u8은 가능한 최고 variant를 선택하고 ffmpeg로 mp4 컨테이너에 저장한다.
#   - ffmpeg는 PATH 또는 .venv/tools/ffmpeg/bin/ffmpeg.exe 위치를 우선 탐색한다.
#   - 파일명 규칙은 이미지와 동일하게 author/time/media_key/tweet_id 기반 결정적 이름을 사용한다.
#
# 로그 정책:
#   - 하강 단계: 매 Burst 진행 상황과 CDP/IO 누적치 로그
#   - 업스크롤 단계: 단일 라인 포맷만 출력
#   - Save시 디버그타입 : debug: [MODE=SAFE] scrollstep=..., newURL=..., dupURL=..., batchSize=..., jsCalls=..., yOffset=..., TargetTotalSeen=..., CurrentTotalSeen=...
#   - backup 종료시 TargetTotalSeen, CurrentTotalSeen, Missing 수를 출력하고,
#                Missing 키/URL 상세를 log.txt에 기록(다운로드는 “키만”으로 진행)
#   - Descent 중 GraphQL 백필은 tqdm postfix의 gql 누적값으로 표시하여 진행바 줄깨짐을 줄인다.
#
# 파일명 정책(결정적):
#   - 기본은 META_if_available 모드: 메타가 있으면 uploader_time_key.ext, 없으면 key.ext
#   - 항상 MEDIA_KEY를 포함하여 재실행/병합 시에도 파일명이 결정적
#   - 이미 같은 파일명이 있으면 스킵(SKIP_IF_EXISTS=True 권장) 
#
# 배포/튜닝:
#   - CONFIG 섹션만 수정해서 환경/성능 튜닝 가능
#   - CDP_ONLY 변동성을 줄이기 위한 옵션을 제공:
#     * 캐시 비활성화(CDP_DISABLE_CACHE), 북마크 진입 후 하드 리로드,
#     * 하강 중 CDP drain 주기(Burst마다), 자동 SAFE 폴백(Threshold 미달 시)
# -----------------------------------------------------------------------------

import sys
import os
import subprocess
import importlib.util
import time
import re
import hashlib
import shutil
import json
from typing import Dict, List, Tuple
import msvcrt
import threading
import socket
import urllib.request
import urllib.parse
import tempfile

# -----------------------------------------------------------------------------
# CONFIG: 사용자 튜닝 파라미터
# -----------------------------------------------------------------------------
CONFIG = {
    # 연결/프로필
    "DEBUGGER_ADDRESS": "127.0.0.1:9222",
    "ATTACH_ONLY": True,  # True면 디버거 attach 성공 전까지 새 세션을 띄우지 않음
    "USER_DATA_DIR": os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "User Data"),
    "PROFILE_DIR_NAME": "Default",
    "AUTO_LAUNCH_DEBUG_CHROME": True,  # attach 전에 디버그 포트 Chrome 자동 실행 시도
    "AUTO_LAUNCH_WAIT_S": 2.5,         # 자동 실행 후 포트 확인까지 대기
    "AUTO_KILL_CHROME_BEFORE_ATTACH": True,  # auto-launch 전에 기존 chrome.exe 정리

    # 하강(북마크 페이지에서 스크롤 최하단까지 탐지할 때 사용되는 파라미터입니다.)
    "DOWN_SCROLL_BURST": 40,         # Burst당 스크롤 횟수
    
    "DOWN_STEP_PX": 1100,            # 스크롤 1회 픽셀 (기본), 실제론 VH 기반 DOWN_STEP_PX_EFF가 사용됨.  VH 기반 클램프로 자동 조정됨.
    "DOWN_DELAY_S": 0.035,            # 스크롤 호출 사이 대기
    "DOWN_BUFFER_CHECKS": 6,         # Burst 후 scrollHeight 증가 체크 횟수
    "DOWN_BUFFER_SLEEP_S": 0.18,     # 각 체크 간 대기
    "DOWN_STALL_TOLERANCE": 10,       # scrollHeight 증가 정지 연속 허용 횟수
    "YOFFSET_STALL_BURSTS": 10,       # 연속 Burst 동안 yOffset 변화 없음 허용 횟수
    "YOFFSET_EPS": 8,                # yOffset 변화 유효성 오차(px)
    "DESCENT_CDP_LOG_INTERVAL": 1,   # 하강 중 CDP/IO drain 및 로그 주기(Burst 단위, 1=매 Burst)

    # CDP 안정화 옵션
    "CDP_DISABLE_CACHE": True,               # Network.setCacheDisabled(true)로 캐시 무효화
    "HARD_RELOAD_ON_BOOKMARKS": True,        # 북마크 진입 직후 Page.reload(ignoreCache=true)
    "CDP_ONLY_AUTOFALLBACK": True,           # CDP_ONLY 타겟 수 낮으면 SAFE로 자동 폴백
    "CDP_ONLY_MIN_KEYS": 500,                # 최소 허용 키 수(환경에 맞게 조정)
    "CDP_ONLY_MIN_RATIO_OF_PEAK": 0.90,      # 하강 중 관측된 피크 대비 허용 최소 비율
    "PERIODIC_STOP_HIT_STREAK": 5,           # PERIODIC: 기존 키 hit만 연속 N회 관측되면 하강 조기중단

    # 업스크롤(SAFE 모드에서만 사용)
    "UP_STEP_PX": 3500,              # 의도 스텝 상한(실제 stepPxEff는 뷰포트 기반으로 클램프)
    "UP_DELAY_S": 0.04,              # 배치 폴링/스크롤 settle 대기 최소단위
    "VIEWPORT_PAD": 300,             # 수집 패딩(px)
    "SAFE_OVERLAP_RATIO": 0.35,      # 커버리지 대비 최소 겹침 비율(0.4~0.6 권장)

    # 다운로드
    "MAX_WORKERS": 10,               # 이미지 병렬 다운로드 스레드 수

    # 파일명 정책(결정적 파일명 + 존재시 스킵)
    # - "meta_if_available": 메타 있으면 uploader_time_key.ext, 없으면 key.ext
    # - "key_only":          항상 key.ext
    "FILENAME_MODE": "meta_if_available",
    "SKIP_IF_EXISTS": True,          # 같은 파일명 있으면 다운로드 스킵
}

# 매크로 변수로 CONFIG 값 할당
DEBUGGER_ADDRESS     = CONFIG["DEBUGGER_ADDRESS"]
ATTACH_ONLY          = CONFIG["ATTACH_ONLY"]
USER_DATA_DIR        = CONFIG["USER_DATA_DIR"]
PROFILE_DIR_NAME     = CONFIG["PROFILE_DIR_NAME"]
AUTO_LAUNCH_DEBUG_CHROME = CONFIG["AUTO_LAUNCH_DEBUG_CHROME"]
AUTO_LAUNCH_WAIT_S   = CONFIG["AUTO_LAUNCH_WAIT_S"]
AUTO_KILL_CHROME_BEFORE_ATTACH = CONFIG["AUTO_KILL_CHROME_BEFORE_ATTACH"]

DOWN_SCROLL_BURST    = CONFIG["DOWN_SCROLL_BURST"]
DOWN_STEP_PX         = CONFIG["DOWN_STEP_PX"]
DOWN_DELAY_S         = CONFIG["DOWN_DELAY_S"]
DOWN_BUFFER_CHECKS   = CONFIG["DOWN_BUFFER_CHECKS"]
DOWN_BUFFER_SLEEP_S  = CONFIG["DOWN_BUFFER_SLEEP_S"]
DOWN_STALL_TOLERANCE = CONFIG["DOWN_STALL_TOLERANCE"]
YOFFSET_STALL_BURSTS = CONFIG["YOFFSET_STALL_BURSTS"]
YOFFSET_EPS          = CONFIG["YOFFSET_EPS"]
DESCENT_CDP_LOG_INTERVAL = CONFIG["DESCENT_CDP_LOG_INTERVAL"]

CDP_DISABLE_CACHE          = CONFIG["CDP_DISABLE_CACHE"]
HARD_RELOAD_ON_BOOKMARKS   = CONFIG["HARD_RELOAD_ON_BOOKMARKS"]
CDP_ONLY_AUTOFALLBACK      = CONFIG["CDP_ONLY_AUTOFALLBACK"]
CDP_ONLY_MIN_KEYS          = CONFIG["CDP_ONLY_MIN_KEYS"]
CDP_ONLY_MIN_RATIO_OF_PEAK = CONFIG["CDP_ONLY_MIN_RATIO_OF_PEAK"]
PERIODIC_STOP_HIT_STREAK   = CONFIG["PERIODIC_STOP_HIT_STREAK"]

UP_STEP_PX           = CONFIG["UP_STEP_PX"]
UP_DELAY_S           = CONFIG["UP_DELAY_S"]
VIEWPORT_PAD         = CONFIG["VIEWPORT_PAD"]
SAFE_OVERLAP_RATIO   = CONFIG["SAFE_OVERLAP_RATIO"]

MAX_WORKERS          = CONFIG["MAX_WORKERS"]

FILENAME_MODE        = CONFIG["FILENAME_MODE"]
SKIP_IF_EXISTS       = CONFIG["SKIP_IF_EXISTS"]
TID_TAG = "_tid_"

BOOKMARK_META_LOCAL_DIR = "bookmark_meta_local"
BOOKMARK_META_LOCAL_ITEMS_PATH = os.path.join(BOOKMARK_META_LOCAL_DIR, "items.ndjson")
BOOKMARK_META_LOCAL_VIDEO_ITEMS_PATH = os.path.join(BOOKMARK_META_LOCAL_DIR, "items_vid.ndjson")
DOWNLOADED_LOCAL_DIR = "downloaded_images_local"
LEGACY_BOOKMARK_META_OLDVER_DIR = "bookmark_meta_oldver"
LEGACY_BOOKMARK_META_OLDVER_ITEMS_PATH = os.path.join(LEGACY_BOOKMARK_META_OLDVER_DIR, "items.ndjson")
LEGACY_DOWNLOADED_OLDVER_DIR = "downloaded_images_oldver"

def migrate_legacy_local_paths() -> None:
    if (not os.path.exists(BOOKMARK_META_LOCAL_DIR)) and os.path.exists(LEGACY_BOOKMARK_META_OLDVER_DIR):
        try:
            shutil.move(LEGACY_BOOKMARK_META_OLDVER_DIR, BOOKMARK_META_LOCAL_DIR)
            print(f"message: migrated legacy folder {LEGACY_BOOKMARK_META_OLDVER_DIR} -> {BOOKMARK_META_LOCAL_DIR}")
        except Exception as e:
            print(f"message: legacy meta folder migration skipped: {e}")
    if (not os.path.exists(DOWNLOADED_LOCAL_DIR)) and os.path.exists(LEGACY_DOWNLOADED_OLDVER_DIR):
        try:
            shutil.move(LEGACY_DOWNLOADED_OLDVER_DIR, DOWNLOADED_LOCAL_DIR)
            print(f"message: migrated legacy folder {LEGACY_DOWNLOADED_OLDVER_DIR} -> {DOWNLOADED_LOCAL_DIR}")
        except Exception as e:
            print(f"message: legacy download folder migration skipped: {e}")

# -----------------------------------------------------------------------------
# Dependencies: pip auto-installer (최초 실행 시 필요한 패키지 자동 설치)
# -----------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

REQUIRED_PKGS = {
    "requests": "requests",
    "selenium": "selenium",
    "webdriver_manager": "webdriver-manager",
    "PIL": "Pillow",
    "tqdm": "tqdm",
}

def _have_module(module_name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(module_name) is not None

def _pip_install(pip_name: str, retries: int = 2) -> None:
    last = None
    try:
        import ensurepip
        subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=False,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        pass
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
                   check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for i in range(1, retries + 2):
        try:
            print(f"message: installing '{pip_name}' (attempt {i})")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            return
        except subprocess.CalledProcessError as e:
            last = e
            time.sleep(1.0)
    raise RuntimeError(f"Failed to install '{pip_name}': {last}")

def ensure_packages(pkgs: Dict[str, str]) -> None:
    missing = [(k, v) for k, v in pkgs.items() if not _have_module(k)]
    if not missing:
        return
    print("message: preparing required packages...")
    for import_name, pip_name in missing:
        _pip_install(pip_name)
        if not _have_module(import_name):
            raise ImportError(f"Installed but cannot import '{import_name}'")

ensure_packages(REQUIRED_PKGS)

# ----------------------------------------------------------------------------- 
# Selenium / Chrome bootstrap 
# -----------------------------------------------------------------------------
import requests
from PIL import Image # 미사용
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry
except Exception:
    from urllib3.util import Retry  # type: ignore

print('X bookmark backuptool by qus20000\n')
migrate_legacy_local_paths()

def _quick_guess_ext_from_url(url: str) -> str:
    m = re.search(r"[?&]format=([a-zA-Z0-9]+)", url or "")
    if m:
        return "." + m.group(1).lower()
    path = (url or "").split("?", 1)[0]
    _, ext = os.path.splitext(path)
    return ext.lower() if ext else ".jpg"

def _quick_filename_safe(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r'[\\/*?:"<>|]+', "", s)
    s = re.sub(r"\s+", "_", s)
    return s

def _quick_build_filename_from_local(item: Dict[str, str]) -> str:
    author = (item.get("author") or "@unknown").strip() or "@unknown"
    created_norm = (item.get("created_at") or "").replace(":", "").replace("Z", "")
    media_key = (item.get("media_key") or "unknown").strip() or "unknown"
    tweet_id = (item.get("tweet_id") or "0").strip() or "0"
    url = item.get("url") or ""
    if author.startswith("@") and not author.startswith("@_"):
        author = "@_" + author[1:]
    ext = _quick_guess_ext_from_url(url)
    return _quick_filename_safe(f"{author}_{created_norm}_{media_key}{TID_TAG}{tweet_id}{ext}")

def _quick_write_open_by_tid_py(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "open_by_tid.py")
    c = f"""import sys
import re
import webbrowser
TID_TAG = {TID_TAG!r}
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else ""
    if not path:
        print("message: drag and drop an image file onto this script.")
        return
    m = re.search(re.escape(TID_TAG) + r"(\\d+)", path)
    if not m:
        print("message: tid not found in filename:", path)
        return
    tid = m.group(1)
    webbrowser.open(f"https://x.com/i/web/status/{{tid}}")
if __name__ == "__main__":
    main()
"""
    with open(p, "w", encoding="utf-8") as f:
        f.write(c)

def _run_ndjson_only_early() -> None:
    in_path = BOOKMARK_META_LOCAL_ITEMS_PATH
    out_dir = DOWNLOADED_LOCAL_DIR
    os.makedirs(out_dir, exist_ok=True)
    _quick_write_open_by_tid_py(out_dir)
    items: List[Dict[str, str]] = []
    if os.path.exists(in_path):
        with open(in_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        items.append(obj)
                except Exception:
                    continue
    print(f"message: NDJSON_ONLY loaded items={len(items)} from {in_path}")
    if not items:
        print("message: no items to download. exiting.")
        return
    ok = 0
    skipped = 0
    fail = 0
    result_path = os.path.join(out_dir, "download_result.txt")
    with open(result_path, "w", encoding="utf-8") as rf:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            future_map = {}
            for it in items:
                url = it.get("url") or ""
                fname = _quick_build_filename_from_local(it)
                fpath = os.path.join(out_dir, fname)
                if SKIP_IF_EXISTS and os.path.exists(fpath):
                    skipped += 1
                    rf.write(f"OK file={fname} status=skip_exists URL={url}\n")
                    continue
                future_map[ex.submit(requests.get, url, timeout=10)] = (it, fname, url, fpath)
            for fu in tqdm(as_completed(list(future_map.keys())), total=len(future_map), desc="Downloading", unit="file"):
                it, fname, url, fpath = future_map[fu]
                try:
                    resp = fu.result()
                    resp.raise_for_status()
                    with open(fpath, "wb") as f:
                        f.write(resp.content)
                    ok += 1
                    rf.write(f"OK file={fname} status=ok URL={url}\n")
                except Exception as e:
                    fail += 1
                    rf.write(f"FAIL file={fname} err={e} URL={url}\n")
    print(f"message: Downloaded new files: {ok}")
    print(f"message: Skipped already-downloaded files: {skipped}")
    print(f"message: Number of failed downloads: {fail}")
    print(f"message: result_log={result_path}")

def _run_dedupe_only_early() -> None:
    out_dir = DOWNLOADED_LOCAL_DIR
    dup_dir = os.path.join(out_dir, "duplicates")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(dup_dir, exist_ok=True)

    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")
    hash_to_paths: Dict[str, List[str]] = {}
    moved = 0
    unknown_moved = 0

    def _is_unknown_name(path: str) -> bool:
        return "unknown" in os.path.basename(path).lower()

    for name in os.listdir(out_dir):
        src = os.path.join(out_dir, name)
        if not os.path.isfile(src):
            continue
        if not name.lower().endswith(image_exts):
            continue
        try:
            with open(src, "rb") as f:
                h = hashlib.md5(f.read()).hexdigest()
        except Exception:
            continue
        hash_to_paths.setdefault(h, []).append(src)

    for _, paths in hash_to_paths.items():
        if len(paths) <= 1:
            continue
        # unknown 파일은 중복 폴더로 우선 격리: 가능하면 non-unknown 1개를 본 폴더에 남긴다.
        sorted_paths = sorted(paths, key=lambda p: (1 if _is_unknown_name(p) else 0, len(os.path.basename(p)), os.path.basename(p).lower()))
        keeper = sorted_paths[0]
        for src in sorted_paths:
            if src == keeper:
                continue
            name = os.path.basename(src)
            base, ext = os.path.splitext(name)
            dst = os.path.join(dup_dir, name)
            n = 1
            while os.path.exists(dst):
                dst = os.path.join(dup_dir, f"{base}__dup{n}{ext}")
                n += 1
            try:
                shutil.move(src, dst)
                moved += 1
                if _is_unknown_name(src):
                    unknown_moved += 1
            except Exception:
                continue

    print(f"message: dedupe scan completed. unique_hashes={len(hash_to_paths)} moved_duplicates={moved} unknown_moved={unknown_moved}")
    print(f"message: duplicates folder: {dup_dir}")

def _run_ndjson_cleanup_early() -> None:
    path = BOOKMARK_META_LOCAL_ITEMS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"message: ndjson not found: {path}")
        return

    def _canon_url(url: str) -> str:
        u = (url or "").strip()
        if not u:
            return u
        if "name=" in u:
            return re.sub(r"name=[^&]+", "name=orig", u)
        if "?" in u:
            return u + "&name=orig"
        return u + "?name=orig"

    def _normalize_media_key(url: str) -> str:
        u = (url or "").strip()
        m = re.search(r"/media/([^/.?]+)", u)
        if m:
            return m.group(1)
        m = re.search(r"/(ext_tw_video|amplify_video)/(\d+)/", u)
        if m:
            return f"{m.group(1)}_{m.group(2)}"
        m = re.search(r"/([^/?]+)\.mp4(?:\?|$)", u)
        return f"video_{m.group(1)}" if m else ""

    def _normalize_time(ts: str) -> str:
        return (ts or "").replace(":", "").replace("Z", "")

    def _is_unknown_author(author: str) -> bool:
        a = (author or "").strip().lower()
        return (not a) or ("unknown" in a)

    def _score(obj: Dict[str, str]) -> Tuple[int, int, int, int]:
        # 점수가 높을수록 보존 우선
        author = obj.get("author", "") or ""
        created = obj.get("created_at", "") or ""
        tweet_id = obj.get("tweet_id", "") or ""
        url = obj.get("url", "") or ""
        return (
            0 if _is_unknown_author(author) else 1,
            1 if created else 0,
            1 if (tweet_id.isdigit() and tweet_id != "0") else 0,
            1 if url else 0,
        )

    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            obj["url"] = _canon_url(obj.get("url", "") or "")
            mk = (obj.get("media_key", "") or "").strip()
            if not mk:
                mk = _normalize_media_key(obj["url"]) or ""
                obj["media_key"] = mk
            if not obj.get("created_at_norm"):
                obj["created_at_norm"] = _normalize_time(obj.get("created_at", "") or "")
            rows.append(obj)

    before = len(rows)
    kept: Dict[str, Dict[str, str]] = {}
    for obj in rows:
        mk = (obj.get("media_key", "") or "").strip()
        key = mk if mk else (obj.get("url", "") or "")
        if not key:
            continue
        prev = kept.get(key)
        if prev is None:
            kept[key] = obj
            continue
        # 더 좋은 메타를 가진 항목을 남기고, 비어있는 필드는 병합
        if _score(obj) > _score(prev):
            better, worse = obj, prev
        else:
            better, worse = prev, obj
        for fld in ("tweet_id", "author", "created_at", "created_at_norm", "media_key", "url"):
            if not (better.get(fld) or "") and (worse.get(fld) or ""):
                better[fld] = worse.get(fld)
        kept[key] = better

    after = len(kept)
    backup_path = path + ".bak"
    shutil.copyfile(path, backup_path)
    with open(path, "w", encoding="utf-8") as f:
        for obj in kept.values():
            out = {
                "tweet_id": obj.get("tweet_id", "") or "",
                "author": obj.get("author", "") or "",
                "created_at": obj.get("created_at", "") or "",
                "created_at_norm": obj.get("created_at_norm", "") or "",
                "media_key": obj.get("media_key", "") or "",
                "url": obj.get("url", "") or "",
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"message: ndjson cleanup completed. before={before} after={after} removed={before-after}")
    print(f"message: cleaned file: {path}")
    print(f"message: backup file: {backup_path}")

mode: str | None = None
backup_mode: str | None = None
media_mode: str = "IMAGE_ONLY"
# IMAGE_ONLY는 기존 이미지 백업 호환 경로, ALL은 video.twimg.com 수집/다운로드까지 포함한다.
print("\n============================================================")
print("Media scope:")
print("  1) IMAGE_ONLY")
print("  2) ALL (IMAGE + VIDEO)")
print("============================================================")
print("Press '1' or '2'...")
while True:
    ch = msvcrt.getwch()
    if ch == "1":
        media_mode = "IMAGE_ONLY"
        break
    if ch == "2":
        media_mode = "ALL"
        break

print("\n============================================================")
print("Backup strategy:")
print("  1) FULL backup mode     (collect all reachable URLs)")
print("  2) PERIODIC backup mode (stop early when existing keys are detected)")
print("============================================================")
print("Press '1' or '2'...")
while True:
    ch = msvcrt.getwch()
    if ch == "1":
        backup_mode = "FULL"
        break
    if ch == "2":
        backup_mode = "PERIODIC"
        break

print("\n============================================================")
print("Select mode:")
print("  1) CDP_ONLY")
print("  2) SAFE")
print("  3) NDJSON_ONLY (quick download-only, no browser attach)")
print("  4) DEDUPE_ONLY (move duplicate images in downloaded_images_local to subfolder)")
print("  5) NDJSON_CLEANUP (dedupe/normalize bookmark_meta_local/items.ndjson)")
print("  6) VIDEO_META_REPAIR (repair missing/unknown meta in items.ndjson videos)")
print("============================================================")
print("Press '1', '2', '3', '4', '5' or '6' to start...")
while True:
    ch = msvcrt.getwch()
    if ch == "1":
        mode = "CDP_ONLY"
        break
    if ch == "2":
        mode = "SAFE"
        break
    if ch == "3":
        mode = "NDJSON_ONLY"
        break
    if ch == "4":
        mode = "DEDUPE_ONLY"
        break
    if ch == "5":
        mode = "NDJSON_CLEANUP"
        break
    if ch == "6":
        mode = "VIDEO_META_REPAIR"
        break

if mode == "NDJSON_ONLY":
    _run_ndjson_only_early()
    sys.exit(0)
if mode == "DEDUPE_ONLY":
    _run_dedupe_only_early()
    sys.exit(0)
if mode == "NDJSON_CLEANUP":
    _run_ndjson_cleanup_early()
    sys.exit(0)

def parse_debugger_host_port(address: str) -> Tuple[str, int]:
    host, port = address.rsplit(":", 1)
    return host, int(port)

def is_debugger_port_open(address: str, timeout_s: float = 1.0) -> bool:
    try:
        host, port = parse_debugger_host_port(address)
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except Exception:
        return False

def fetch_debugger_browser_version(address: str, timeout_s: float = 1.5) -> str:
    try:
        host, port = parse_debugger_host_port(address)
        url = f"http://{host}:{port}/json/version"
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        return str(data.get("Browser", "") or "")
    except Exception:
        return ""

def parse_chrome_major(browser_version: str) -> str:
    # e.g. "Chrome/128.0.6613.137" -> "128"
    m = re.search(r"Chrome/(\d+)\.", browser_version or "")
    return m.group(1) if m else ""

def find_chrome_exe() -> str:
    candidates = [
        os.path.join(os.environ.get("PROGRAMFILES", ""), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("PROGRAMFILES(X86)", ""), "Google", "Chrome", "Application", "chrome.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return "chrome.exe"

def print_attach_guide() -> None:
    chrome_exe = find_chrome_exe()
    _, port = parse_debugger_host_port(DEBUGGER_ADDRESS)
    print("message: existing Chrome can be attached only when launched with remote-debugging-port.")
    print("message: if attach fails, close all Chrome windows and relaunch Chrome with this command:")
    print(f'message: "{chrome_exe}" --remote-debugging-port={port} --user-data-dir="{USER_DATA_DIR}" --profile-directory="{PROFILE_DIR_NAME}"')
    print("message: then open https://x.com/i/bookmarks in that Chrome and press Enter to retry attach.")

def try_auto_launch_debug_chrome() -> bool:
    def _wait_port() -> bool:
        time.sleep(max(0.2, float(AUTO_LAUNCH_WAIT_S)))
        return is_debugger_port_open(DEBUGGER_ADDRESS, timeout_s=1.0)

    def _popen_chrome(user_data_dir: str) -> None:
        _, port = parse_debugger_host_port(DEBUGGER_ADDRESS)
        args = [
            chrome_exe,
            "--remote-debugging-address=127.0.0.1",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={user_data_dir}",
            f"--profile-directory={PROFILE_DIR_NAME}",
        ]
        subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )

    def _kill_existing_chrome() -> None:
        if not AUTO_KILL_CHROME_BEFORE_ATTACH:
            return
        try:
            # ignore return code; "not found" is normal when chrome is already closed.
            subprocess.run(
                ["taskkill", "/F", "/IM", "chrome.exe"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=8,
            )
            time.sleep(0.6)
            print("message: auto-launch pre-step: existing chrome.exe processes terminated.")
        except Exception as e:
            print(f"message: auto-launch pre-step warning: could not terminate chrome.exe ({type(e).__name__}: {e})")

    try:
        chrome_exe = find_chrome_exe()
        if not chrome_exe or (chrome_exe != "chrome.exe" and (not os.path.isfile(chrome_exe))):
            print("message: auto-launch skipped: chrome executable not found.")
            return False

        _kill_existing_chrome()

        # 1st attempt: user-configured Chrome profile.
        _popen_chrome(USER_DATA_DIR)
        if _wait_port():
            print(f"message: auto-launched Chrome for debuggerAddress {DEBUGGER_ADDRESS} (user profile).")
            return True

        # 2nd attempt: temporary profile fallback (avoids profile lock/policy edge cases).
        temp_profile = os.path.join(tempfile.gettempdir(), "x-bookmark-debug-profile")
        os.makedirs(temp_profile, exist_ok=True)
        _popen_chrome(temp_profile)
        if _wait_port():
            print(f"message: auto-launched Chrome for debuggerAddress {DEBUGGER_ADDRESS} (temp profile fallback).")
            return True

        print("message: auto-launch attempted, but debugger port is still not reachable.")
        return False
    except Exception as e:
        print(f"message: auto-launch failed: {type(e).__name__}: {e}")
        return False

def get_chromedriver_service() -> Service:
    try:
        return Service()
    except Exception:
        pass
    try:
        driver_path = ChromeDriverManager().install()
        return Service(executable_path=driver_path)
    except Exception as e:
        raise RuntimeError(f"ChromeDriver setup failed: {e}")

def build_versioned_service_for_browser(browser_version: str) -> Service | None:
    major = parse_chrome_major(browser_version)
    if not major:
        return None
    try:
        # webdriver-manager는 major 버전 문자열도 허용
        driver_path = ChromeDriverManager(driver_version=major).install()
        return Service(executable_path=driver_path)
    except Exception:
        return None

def create_chrome_driver(
    options: webdriver.ChromeOptions,
    service: Service | None = None,
    browser_version: str = "",
):
    # 1) Selenium Manager 자동 매칭 우선 (버전 불일치 회피)
    try:
        return webdriver.Chrome(options=options)
    except Exception as e_auto:
        # 2) debugger 브라우저 메이저 버전에 맞춘 chromedriver 강제 매칭
        ver_service = build_versioned_service_for_browser(browser_version)
        if ver_service is not None:
            try:
                return webdriver.Chrome(service=ver_service, options=options)
            except Exception:
                pass
        if service is not None:
            # 3) 기존 service 경로 재시도
            try:
                return webdriver.Chrome(service=service, options=options)
            except Exception:
                raise e_auto
        raise e_auto

def apply_common_chrome_options(options: webdriver.ChromeOptions, for_attach: bool = False) -> None:
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--disable-webgpu")
    options.add_argument("--disable-accelerated-2d-canvas")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    # 일부 드라이버 조합에서는 attach 모드에서 아래 옵션이 invalid argument를 유발함.
    if not for_attach:
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
    # CDP 성능 로그 활성화 (Network.* 이벤트 수집)
    options.set_capability("goog:loggingPrefs", {"performance": "ALL", "browser": "ALL"})

def try_attach_existing_chrome(service: Service, browser_version: str = ""):
    try:
        options = webdriver.ChromeOptions()
        apply_common_chrome_options(options, for_attach=True)
        options.add_experimental_option("debuggerAddress", DEBUGGER_ADDRESS)
        driver = create_chrome_driver(options, service, browser_version=browser_version)
        print("message: attached to existing Chrome via debuggerAddress.")
        return driver
    except Exception as e:
        print(f"message: attach exception: {type(e).__name__}: {e}")
        return None

def try_launch_with_profile(service: Service):
    try:
        options = webdriver.ChromeOptions()
        apply_common_chrome_options(options)
        if USER_DATA_DIR and os.path.isdir(USER_DATA_DIR):
            options.add_argument(f'--user-data-dir={USER_DATA_DIR}')
            options.add_argument(f'--profile-directory={PROFILE_DIR_NAME}')
        driver = create_chrome_driver(options, service, browser_version="")
        print("message: launched Chrome with existing user profile.")
        return driver
    except Exception:
        return None

def launch_fresh_session(service: Service):
    options = webdriver.ChromeOptions()
    apply_common_chrome_options(options)
    driver = create_chrome_driver(options, service, browser_version="")
    print("message: launched fresh Chrome session.")
    return driver

service = get_chromedriver_service()
driver = None
attached_mode = False
allow_fallback_launch = (not ATTACH_ONLY)
auto_launch_attempted = False

if is_debugger_port_open(DEBUGGER_ADDRESS):
    browser_ver = fetch_debugger_browser_version(DEBUGGER_ADDRESS)
    if browser_ver:
        print(f"message: debugger browser = {browser_ver}")
    driver = try_attach_existing_chrome(service, browser_version=browser_ver)
    if driver is not None:
        attached_mode = True
else:
    print(f"message: debuggerAddress {DEBUGGER_ADDRESS} is not reachable.")
    if AUTO_LAUNCH_DEBUG_CHROME:
        auto_launch_attempted = True
        if try_auto_launch_debug_chrome():
            browser_ver = fetch_debugger_browser_version(DEBUGGER_ADDRESS)
            if browser_ver:
                print(f"message: debugger browser = {browser_ver}")
            driver = try_attach_existing_chrome(service, browser_version=browser_ver)
            if driver is not None:
                attached_mode = True

if driver is None:
    print_attach_guide()
    if ATTACH_ONLY:
        print("message: ATTACH_ONLY=True. fallback launch is disabled.")
        print("message: press Enter to retry attach, or press 'q' to quit.")
    else:
        print("message: press Enter to retry attach, 'f' to continue with fallback launch, or 'q' to quit.")

    while driver is None:
        ch = msvcrt.getwch()
        if ch == "\r":
            if is_debugger_port_open(DEBUGGER_ADDRESS):
                browser_ver = fetch_debugger_browser_version(DEBUGGER_ADDRESS)
                if browser_ver:
                    print(f"message: debugger browser = {browser_ver}")
                driver = try_attach_existing_chrome(service, browser_version=browser_ver)
                if driver is not None:
                    attached_mode = True
                    break
                print("message: attach failed even though debugger port is open. check Chrome/Driver version match.")
            else:
                if AUTO_LAUNCH_DEBUG_CHROME and (not auto_launch_attempted):
                    auto_launch_attempted = True
                    if try_auto_launch_debug_chrome():
                        browser_ver = fetch_debugger_browser_version(DEBUGGER_ADDRESS)
                        if browser_ver:
                            print(f"message: debugger browser = {browser_ver}")
                        driver = try_attach_existing_chrome(service, browser_version=browser_ver)
                        if driver is not None:
                            attached_mode = True
                            break
                print(f"message: debuggerAddress {DEBUGGER_ADDRESS} is still not reachable.")
            print("message: after opening Chrome with remote debugging, press Enter to retry.")
            continue
        if ch.lower() == "q":
            raise RuntimeError("Attach was not completed. Exiting because login session must be reused.")
        if (not ATTACH_ONLY) and ch.lower() == "f":
            allow_fallback_launch = True
            break
        if ATTACH_ONLY:
            print("message: invalid key. press Enter(retry) or q(quit).")
        else:
            print("message: invalid key. press Enter(retry), f(fallback), or q(quit).")

if driver is None and allow_fallback_launch:
    driver = try_launch_with_profile(service)
    if driver is not None:
        attached_mode = True
    else:
        try:
            driver = launch_fresh_session(service)
        except Exception as e:
            print(f"message: primary Chrome launch failed: {e}\nmessage: retry with webdriver-manager explicit path")
            driver_path = ChromeDriverManager().install()
            service = Service(executable_path=driver_path)
            driver = launch_fresh_session(service)

if driver is None:
    raise RuntimeError("No Chrome session available. Attach failed and fallback launch is disabled.")

driver.implicitly_wait(3)

# CDP enable + 옵션
try:
    driver.execute_cdp_cmd("Network.enable", {})
    if CDP_DISABLE_CACHE:
        driver.execute_cdp_cmd("Network.setCacheDisabled", {"cacheDisabled": True})
except Exception as _e:
    print(f"message: Network.enable/setCacheDisabled failed or not supported: {_e}")

# -----------------------------------------------------------------------------
# Page open & login (필요 시)
# -----------------------------------------------------------------------------
BOOKMARKS_URL = "https://x.com/i/bookmarks"

def wait_until_bookmarks(timeout_s: float = 180.0) -> bool:
    t0 = time.time()
    while (time.time() - t0) < timeout_s:
        try:
            cur = driver.current_url or ""
        except Exception:
            cur = ""
        if cur.startswith(BOOKMARKS_URL):
            return True
        time.sleep(0.3)
    return False

driver.get(BOOKMARKS_URL)

# 로그인 안 된 상태면 X가 /login 등으로 리다이렉트할 수 있음.
# 이 경우 자동 로그인을 시도하지 말고, 현재 열린 크롬 창에서 사용자가 직접 로그인하도록 유도.
if not wait_until_bookmarks(timeout_s=5.0):
    print('message: Login required. Please log in manually in the opened Chrome window.')
    print('message: After successful login, press Enter to retry (or q to quit).')
    while True:
        ch = msvcrt.getwch()
        if ch in ("q", "Q"):
            raise RuntimeError("User aborted while waiting for manual login.")
        if ch != "\r":
            continue
        try:
            driver.get(BOOKMARKS_URL)
        except Exception as e:
            print(f"message: bookmarks navigation retry failed: {e}")
        if wait_until_bookmarks(timeout_s=20.0):
            break
        try:
            cur = driver.current_url or ""
        except Exception:
            cur = ""
        print(f"message: still not on bookmarks. current_url={cur}")
        print('message: complete login in Chrome, then press Enter to retry (or q to quit).')

# 북마크 진입 후 하드 리로드(이벤트 누락 방지)
if HARD_RELOAD_ON_BOOKMARKS:
    try:
        driver.execute_cdp_cmd("Page.reload", {"ignoreCache": True})
        time.sleep(1.0)
    except Exception as e:
        print(f"message: Page.reload(ignoreCache) failed: {e}")

# -----------------------------------------------------------------------------
# Logger (터미널 + 파일 동시 기록)
# -----------------------------------------------------------------------------
class Logger:
    def __init__(self, log_file: str):
        self.terminal = sys.stdout
        self.log = open(log_file, "w", encoding="utf-8")
    def write(self, message: str):
        if message != "\r":
            self.terminal.write(message)
            self.log.write(message)
        else:
            self.terminal.write(message)
            self.log.write(message)
    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass
        try:
            self.log.flush()
        except Exception:
            pass

new_folder_path = DOWNLOADED_LOCAL_DIR
os.makedirs(new_folder_path, exist_ok=True)

log_file_path = os.path.join(new_folder_path, "log.txt")
sys.stdout = Logger(log_file_path)

# -----------------------------------------------------------------------------
# Mode selector (1= CDP_ONLY, 2 = SAFE, 3 = NDJSON_ONLY)
# -----------------------------------------------------------------------------
print(f"message: selected mode = {mode}")

# -----------------------------------------------------------------------------
# JS Collectors (IO/DOM) (IntersectionObserver 기반, CDP_ONLY/SAFE 모두에서 사용됨.)
# CDP는 pbs.twimg.com/media/... 요청을 수집하고, IO는 IntersectionObserver로 DOM에서 이미지 URL을 수집합니다.
# CDP는 이미지 URL을 빠르고 대량으로 잡아오지만, Uploader/UploadTime 메타가 없습니다.
# IO는 DOM에서 메타를 수집하지만, IntersectionObserver로 스크롤 위치에 따라 필요한 부분만 수집합니다.
# DOM은 트윗 카드 안에서 업로더 핸들(@username)과 업로드 시간을 추출할 수 있으며, 파일명에 사용되는 메타데이터를 수집하는 용도로 사용됩니다.
# 이 두 가지 방법을 조합하여, CDP_ONLY 모드에서는 빠르게 URL을 수집하고, SAFE 모드에서는 DOM을 통해 메타데이터를 최대한 채워서 파일명을 결정합니다.
# CDP Only 모드에서도 어느정도 메타데이터 수집이 가능하지만, 완전하지 않습니다. 신뢰성이 부족하므로, Safe 모드에서는 하강 후 천천히 스크롤을 상승시켜
# DOM을 통해 메타데이터를 안정적으로 수집하고, Target과 Current를 비교하여 매칭되는 이미지는 파일명을 부여하고, 누락되는 데이터는 log.txt에 기록합니다.
## 전역 버퍼/중복제거
# window.__xBuf: 새로 관측된 항목을 모아두는 버퍼
# window.__xSeen: URL 단위 중복 방지 세트\
# __xDump(): 파이썬 쪽에서 드레인(꺼내고 비우기)

## 메타 추출 로직(트윗 카드 단위)
# __xExtractFromArticle(art):
# <time datetime="...">에서 ISO 시각 추출(“.000Z” 트리밍)
# span 텍스트 중 @ 포함(핸들) 발견 시 업로더 후보로 사용
# img[src*="media"] 전부 순회, URL을 name=orig로 정규화 후 버퍼에 push

## 관측 장치
# IntersectionObserver: 뷰포트에 들어오는 ARTICLE마다 __xExtractFromArticle 호출 → 보이는 순간 수집
# MutationObserver: 새로 추가되는 ARTICLE을 자동 관측 대상으로 등록 → 가상 스크롤로 DOM이 바뀌어도 추적
# -----------------------------------------------------------------------------
JS_COLLECT_SNIPPET = r"""
const pad = arguments[0];
const topY = window.scrollY - pad;
const bottomY = window.scrollY + window.innerHeight + pad;
function inRange(el) {
  const r = el.getBoundingClientRect();
  const y1 = window.scrollY + r.top;
  const y2 = window.scrollY + r.bottom;
  return (y2 >= topY && y1 <= bottomY);
}
const results = [];
const articles = document.querySelectorAll('article');
for (const art of articles) {
  try {
    if (!inRange(art)) continue;
    let tweetId = '';
    const statusLinks = art.querySelectorAll('a[href*="/status/"]');
    for (const a of statusLinks) {
      const href = a.getAttribute('href') || '';
      const m = href.match(/\/status\/(\d+)/);
      if (m) tweetId = m[1];
    }
    if (!tweetId) {
      const html = art.innerHTML || '';
      const m2 = html.match(/\/status\/(\d+)/);
      if (m2) tweetId = m2[1];
    }
    const timeEl = art.querySelector('time');
    let dt = '';
    if (timeEl && timeEl.getAttribute('datetime')) {
      dt = timeEl.getAttribute('datetime');
    }
    let uploader = '';
    const userNameBox = art.querySelector('[data-testid="User-Name"]');
    if (userNameBox) {
      const spans2 = userNameBox.querySelectorAll('span');
      for (const s of spans2) {
        const t = (s.textContent || '').trim();
        if (t.startsWith('@')) { uploader = t; break; }
      }
    }
    if (!uploader) {
      const spans = art.querySelectorAll('span');
      for (const s of spans) {
        const t = (s.textContent || '').trim();
        if (t.startsWith('@')) { uploader = t; break; }
      }
    }
    const imgs = art.querySelectorAll('img[src*="media"]');
    for (const im of imgs) {
      let src = im.getAttribute('src') || '';
      if (!src) continue;
      src = src.replace(/name=[^&]+/, 'name=orig');
      results.push({url: src, uploader_name: uploader, upload_time: dt, tweet_id: tweetId});
    }
    const vids = art.querySelectorAll('video[src], video source[src]');
    for (const vd of vids) {
      let src = vd.getAttribute('src') || '';
      if (!src) continue;
      results.push({url: src, uploader_name: uploader, upload_time: dt, tweet_id: tweetId});
    }
  } catch(e) { }
}
return results;
"""

JS_OBSERVER_BOOTSTRAP = r"""
try {
  if (!window.__xInit) {
    window.__xInit = true;
    window.__xBuf = [];
    window.__xSeen = new Set();

    function __xPush(url, uploader, dt, tid) {
      if (!url) return;
      if (url.includes('pbs.twimg.com/media/')) {
        url = url.replace(/name=[^&]+/, 'name=orig');
      }
      if (window.__xSeen.has(url)) return;
      window.__xSeen.add(url);
      window.__xBuf.push({url, uploader_name: uploader || '', upload_time: dt || '', tweet_id: tid || ''});
    }

    function __xExtractFromArticle(art) {
      try {
        let tweetId = '';
        const statusLinks = art.querySelectorAll('a[href*="/status/"]');
        for (const a of statusLinks) {
          const href = a.getAttribute('href') || '';
          const m = href.match(/\/status\/(\d+)/);
          if (m) tweetId = m[1];
        }
        if (!tweetId) {
          const html = art.innerHTML || '';
          const m2 = html.match(/\/status\/(\d+)/);
          if (m2) tweetId = m2[1];
        }
        const timeEl = art.querySelector('time');
        let dt = '';
        if (timeEl && timeEl.getAttribute('datetime')) {
          dt = timeEl.getAttribute('datetime');
        }
        let uploader = '';
        const userNameBox = art.querySelector('[data-testid="User-Name"]');
        if (userNameBox) {
          const spans2 = userNameBox.querySelectorAll('span');
          for (const s of spans2) {
            const t = (s.textContent || '').trim();
            if (t.startsWith('@')) { uploader = t; break; }
          }
        }
        if (!uploader) {
          const spans = art.querySelectorAll('span');
          for (const s of spans) {
            const t = (s.textContent || '').trim();
            if (t.startsWith('@')) { uploader = t; break; }
          }
        }
        const imgs = art.querySelectorAll('img[src*="media"]');
        for (const im of imgs) {
          const src = im.getAttribute('src') || '';
          if (src) __xPush(src, uploader, dt, tweetId);
        }
        const vids = art.querySelectorAll('video[src], video source[src]');
        for (const vd of vids) {
          const src = vd.getAttribute('src') || '';
          if (src) __xPush(src, uploader, dt, tweetId);
        }
      } catch (e) {}
    }

    function __xSeed() {
      const arts = document.querySelectorAll('article');
      for (const a of arts) __xExtractFromArticle(a);
    }

    const io = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting && entry.target) {
          __xExtractFromArticle(entry.target);
        }
      }
    }, {root: null, threshold: 0.01});

    function __xObserveExisting() {
      const arts = document.querySelectorAll('article');
      for (const a of arts) io.observe(a);
    }

    const mo = new MutationObserver((list) => {
      for (const m of list) {
        for (const node of m.addedNodes || []) {
          if (!(node instanceof Element)) continue;
          if (node.tagName === 'ARTICLE') {
            io.observe(node);
            __xExtractFromArticle(node);
          } else {
            const arts = node.querySelectorAll ? node.querySelectorAll('article') : [];
            for (const a of arts) {
              io.observe(a);
              __xExtractFromArticle(a);
            }
          }
        }
      }
    });
    mo.observe(document.body, {childList: true, subtree: true});

    window.__xDump = function() {
      const out = window.__xBuf.slice();
      window.__xBuf.length = 0;
      return out;
    };

    __xSeed();
    __xObserveExisting();
  }
  return true;
} catch(e) {
  return false;
}
"""

# -----------------------------------------------------------------------------
# Utilities 
# -----------------------------------------------------------------------------
def normalize_media_key(url: str) -> str | None:
    """이미지/비디오 URL에서 결정적 키를 추출(실패 시 None)."""
    try:
        u = (url or "").strip()
        if not u:
            return None
        if "pbs.twimg.com/media/" in u:
            m = re.search(r"/media/([^/.?]+)", u)
            return m.group(1) if m else None
        # X GIF/동영상 URL (video.twimg.com)
        m = re.search(r"/(ext_tw_video|amplify_video)/(\d+)/", u)
        if m:
            return f"{m.group(1)}_{m.group(2)}"
        # fallback: 파일명 기반 키 (.mp4)
        m = re.search(r"/([^/?]+)\.mp4(?:\?|$)", u)
        if m:
            return f"video_{m.group(1)}"
        return None
    except Exception:
        return None

def canon_media_url(url: str) -> str:
    """이미지는 name=orig 정규화, 비디오는 원본 URL 유지."""
    try:
        u = (url or "").strip()
        if "pbs.twimg.com/media/" in u:
            if "name=" in u:
                u = re.sub(r"name=[^&]+", "name=orig", u)
            else:
                if "?" in u:
                    u = u + "&name=orig"
                else:
                    u = u + "?name=orig"
        return u
    except Exception:
        return url

def normalize_time(ts: str) -> str:
    return (ts or "").replace(":", "").replace("Z", "")

def _guess_ext_from_url(url: str) -> str:
    m = re.search(r"[?&]format=([a-zA-Z0-9]+)", url)
    if m:
        return "." + m.group(1).lower()
    path = url.split("?", 1)[0]
    _, ext = os.path.splitext(path)
    if ext:
        return ext.lower()
    return ".jpg"

def _video_quality_score(url: str) -> int:
    """Higher is better. Prefer MP4 vid variants with larger resolution."""
    u = (url or "").strip().lower()
    if "video.twimg.com" not in u:
        return -1
    if "/aud/" in u:
        return 10_000
    if "/0/0/" in u:
        return 20_000
    m = re.search(r"/vid/(\d+)x(\d+)/", u)
    if m:
        w = int(m.group(1))
        h = int(m.group(2))
        return 2_000_000 + (w * h)
    m = re.search(r"bandwidth=(\d+)", u)
    if m:
        return 1_500_000 + int(m.group(1))
    if u.endswith(".mp4") or ".mp4?" in u:
        return 1_200_000
    if u.endswith(".m3u8") or ".m3u8?" in u:
        return 200_000
    return 100_000

def _pick_best_variant_from_m3u8(master_url: str, session: requests.Session) -> str:
    """Return best variant URL from m3u8 master playlist. Fallback to input URL."""
    try:
        resp = session.get(master_url, timeout=10)
        resp.raise_for_status()
        text = resp.text or ""
    except Exception:
        return master_url

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    best_url = master_url
    best_score = -1
    pending_score = None

    for ln in lines:
        if ln.startswith("#EXT-X-STREAM-INF:"):
            bw = 0
            area = 0
            m_bw = re.search(r"BANDWIDTH=(\d+)", ln)
            if m_bw:
                bw = int(m_bw.group(1))
            m_res = re.search(r"RESOLUTION=(\d+)x(\d+)", ln)
            if m_res:
                area = int(m_res.group(1)) * int(m_res.group(2))
            pending_score = (bw * 10) + area
            continue
        if ln.startswith("#"):
            continue
        if pending_score is None:
            continue
        cand = urllib.parse.urljoin(master_url, ln)
        cl = cand.lower()
        if "/aud/" in cl or "/0/0/" in cl:
            pending_score = None
            continue
        score = pending_score
        if score > best_score:
            best_score = score
            best_url = cand
        pending_score = None

    return best_url

def is_preferred_video_mp4_url(url: str) -> bool:
    u = (url or "").lower()
    if "video.twimg.com/" not in u:
        return False
    if "/aud/" in u or "/0/0/" in u:
        return False
    if ".mp4" not in u:
        return False
    # prefer complete variant path: .../vid/<codec>/<WxH>/<file>.mp4
    return re.search(r"/vid/[^/]+/\d+x\d+/[^/?]+\.mp4(?:\?|$)", u) is not None

def resolve_best_video_url(url: str, session: requests.Session) -> str:
    """Resolve to higher-quality video URL when possible."""
    u = (url or "").strip()
    if "video.twimg.com" not in u:
        return u
    if "/aud/" in u:
        return u
    if u.endswith(".m3u8") or ".m3u8?" in u:
        # master -> best variant. If variant still m3u8, keep it as is.
        return _pick_best_variant_from_m3u8(u, session)
    return u

def _download_hls_to_mp4(m3u8_url: str, out_path_mp4: str) -> Tuple[bool, str | None]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        # VSCode Run/Debug 등 PATH 미주입 환경을 위해 로컬 경로 fallback 탐색
        base_dir = os.path.dirname(os.path.abspath(__file__))
        local_candidates = [
            os.path.join(base_dir, ".venv", "tools", "ffmpeg", "bin", "ffmpeg.exe"),
            os.path.join(base_dir, "ffmpeg.exe"),
        ]
        for c in local_candidates:
            if os.path.isfile(c):
                ffmpeg = c
                break
    if not ffmpeg:
        return False, "ffmpeg_not_found"
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        m3u8_url,
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        out_path_mp4,
    ]
    try:
        p = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        if p.returncode == 0 and os.path.exists(out_path_mp4) and os.path.getsize(out_path_mp4) > 4096:
            return True, None
        err = (p.stderr or b"").decode("utf-8", errors="ignore").strip()
        return False, err or "ffmpeg_failed"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def _slug(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r'[\\/*?:"<>|]+', '', s)
    s = re.sub(r'\s+', '_', s)
    return s[:80]

def _filename_safe(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r'[\\/*?:"<>|]+', '', s)
    s = re.sub(r"\s+", "_", s)
    return s

def make_deterministic_filename(url: str, uploader_name: str, upload_time: str, tweet_id: str = "") -> str:
    """
    FILENAME_MODE:
      - "meta_if_available": 업로더/시간 있으면 uploader_time_key.ext, 없으면 key.ext
      - "key_only":          항상 key.ext
    항상 MEDIA_KEY는 포함해서 재실행/병합 시에도 일관성 보장.
    """
    mk = normalize_media_key(url) or "unknown"
    ext = _guess_ext_from_url(url)

    if FILENAME_MODE == "key_only":
        return f"{mk}{ext}"

    # OAuth2 downloader와 동일 규칙:
    # @_handle_YYYY-mm-ddTHHMMSS.xxx_mediaKey_tid_<tweetid>.<ext>
    author = (uploader_name or "@unknown").strip() or "@unknown"
    created_norm = normalize_time(upload_time or "")
    twid = (tweet_id or "0").strip() or "0"

    if author.startswith("@") and not author.startswith("@_"):
        author = "@_" + author[1:]

    name = f"{author}_{created_norm}_{mk}{TID_TAG}{twid}{ext}"
    return _filename_safe(name)

def ensure_video_filename_extension(url: str, filename: str) -> str:
    u = (url or "").lower()
    if ".m3u8" in u:
        base, _ = os.path.splitext(filename)
        return base + ".mp4"
    return filename

def _extract_status_id_from_text(s: str) -> str:
    m = re.search(r"/status/(\d+)", s or "")
    return m.group(1) if m else ""

def _extract_status_id_deep(obj) -> str:
    try:
        if obj is None:
            return ""
        if isinstance(obj, str):
            return _extract_status_id_from_text(obj)
        if isinstance(obj, dict):
            for k in ("documentURL", "url", "referer", "referrer", "initiator"):
                if k in obj:
                    tid = _extract_status_id_deep(obj.get(k))
                    if tid:
                        return tid
            for v in obj.values():
                tid = _extract_status_id_deep(v)
                if tid:
                    return tid
            return ""
        if isinstance(obj, list):
            for v in obj:
                tid = _extract_status_id_deep(v)
                if tid:
                    return tid
        return ""
    except Exception:
        return ""

# CDP 이벤트 간(tweet_id) 연결 캐시
CDP_TID_BY_REQUEST_ID: Dict[str, str] = {}
CDP_TID_BY_MEDIA_KEY: Dict[str, str] = {}
CDP_TID_SOURCE_BY_MEDIA_KEY: Dict[str, str] = {}
CDP_REQ_URL_BY_ID: Dict[str, str] = {}
CDP_RESP_MIME_BY_ID: Dict[str, str] = {}
CDP_FINISHED_REQ_IDS: set[str] = set()
CDP_GRAPHQL_BODY_PARSED_REQ_IDS: set[str] = set()
LAST_FULL_DESCENT_BOTTOM_Y = 0
LAST_FULL_DESCENT_SCROLL_H = 0
FULL_DESCENT_MIN_CDP_KEYS_BEFORE_STALL = 0

def drain_cdp_media_with_meta() -> List[Dict[str, str]]:
    # CDP 성능로그에서 이미지/비디오 URL + 가능한 tweet_id 추출.
    out: List[Dict[str, str]] = []
    seen = set()
    try:
        logs = driver.get_log("performance")
    except Exception:
        logs = []
    for entry in logs:
        try:
            msg = json.loads(entry.get("message", "{}")).get("message", {})
            method = msg.get("method", "")
            params = msg.get("params", {})
            url = None
            tweet_id = ""
            req_id = (params.get("requestId", "") or "").strip()
            if method == "Network.requestWillBeSent":
                req = params.get("request", {}) or {}
                url = req.get("url")
                if req_id and url:
                    CDP_REQ_URL_BY_ID[req_id] = str(url)
                tweet_id = _extract_status_id_from_text(params.get("documentURL", "") or "")
                if not tweet_id:
                    tweet_id = _extract_status_id_from_text(req.get("headers", {}).get("Referer", "") if isinstance(req.get("headers"), dict) else "")
                if not tweet_id:
                    initiator = params.get("initiator", {}) or {}
                    if isinstance(initiator, dict):
                        tweet_id = _extract_status_id_from_text(initiator.get("url", "") or "")
                if not tweet_id:
                    tweet_id = _extract_status_id_deep(params)
                if req_id and tweet_id:
                    CDP_TID_BY_REQUEST_ID[req_id] = tweet_id
            elif method == "Network.responseReceived":
                res = params.get("response", {})
                url = res.get("url")
                if req_id and url:
                    CDP_REQ_URL_BY_ID[req_id] = str(url)
                if req_id:
                    CDP_RESP_MIME_BY_ID[req_id] = str(res.get("mimeType") or "")
                if req_id:
                    tweet_id = CDP_TID_BY_REQUEST_ID.get(req_id, "")
                if not tweet_id:
                    tweet_id = _extract_status_id_deep(params)
            elif method == "Network.loadingFinished":
                if req_id:
                    CDP_FINISHED_REQ_IDS.add(req_id)
                continue
            if not url:
                continue
            is_image = ("pbs.twimg.com/media/" in url)
            is_video = is_preferred_video_mp4_url(url)
            is_video_master = ("video.twimg.com/" in url and ".m3u8" in url)
            if not (is_image or is_video or is_video_master):
                continue
            cu = canon_media_url(url)
            if not tweet_id:
                mk2 = normalize_media_key(cu) or ""
                if mk2:
                    tweet_id = CDP_TID_BY_MEDIA_KEY.get(mk2, "")
            if cu not in seen:
                seen.add(cu)
                out.append({"url": cu, "tweet_id": tweet_id})
            if tweet_id:
                mk = normalize_media_key(cu) or ""
                if mk and not CDP_TID_BY_MEDIA_KEY.get(mk):
                    CDP_TID_BY_MEDIA_KEY[mk] = tweet_id
                    CDP_TID_SOURCE_BY_MEDIA_KEY[mk] = "cdp"
        except Exception:
            continue
    return out

def _iter_dicts(x):
    if isinstance(x, dict):
        yield x
        for v in x.values():
            yield from _iter_dicts(v)
    elif isinstance(x, list):
        for v in x:
            yield from _iter_dicts(v)

def _at_handle(sn: str) -> str:
    s = (sn or "").strip()
    if not s:
        return ""
    return s if s.startswith("@") else f"@{s}"

def _dict_path(root: dict, keys: List[str]):
    cur = root
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur

def _screen_name_from_graphql_node(d: dict) -> str:
    paths = [
        ["core", "user_results", "result", "legacy", "screen_name"],
        ["user_results", "result", "legacy", "screen_name"],
        ["user", "result", "legacy", "screen_name"],
        ["author", "legacy", "screen_name"],
        ["legacy", "screen_name"],
        ["screen_name"],
    ]
    for p in paths:
        v = _dict_path(d, p)
        if isinstance(v, str) and v.strip():
            return _at_handle(v)
    return ""

def _tweet_meta_from_graphql_node(d: dict, user_screen_by_id: Dict[str, str]) -> Dict[str, str]:
    legacy = d.get("legacy") if isinstance(d.get("legacy"), dict) else {}
    created_at = (legacy.get("created_at") or d.get("created_at") or "").strip()
    tid = ""
    if isinstance(legacy, dict):
        tid = (legacy.get("id_str") or "").strip()
    if not tid:
        tid = (d.get("rest_id") or d.get("id_str") or "").strip()

    # Avoid treating user objects as tweets. Tweet-like legacy nodes normally have
    # created_at/user_id_str/full_text/entities or nested media/card data.
    looks_tweet_like = bool(
        created_at
        or (isinstance(legacy, dict) and (
            legacy.get("user_id_str")
            or legacy.get("full_text")
            or legacy.get("entities")
            or legacy.get("extended_entities")
        ))
        or d.get("core")
        or d.get("card")
    )
    if not tid.isdigit() or not looks_tweet_like:
        return {}

    author = _screen_name_from_graphql_node(d)
    if not author and isinstance(legacy, dict):
        uid = (legacy.get("user_id_str") or "").strip()
        if uid:
            author = user_screen_by_id.get(uid, "")
    return {"tweet_id": tid, "author": author, "created_at": created_at}

def _merge_graphql_video_meta(a: Dict[str, str], b: Dict[str, str]) -> Dict[str, str]:
    out = dict(a or {})
    for k in ("tweet_id", "author", "created_at"):
        if not (out.get(k) or "") and (b.get(k) or ""):
            out[k] = b[k]
    return out

def _graphql_video_meta_score(m: Dict[str, str]) -> int:
    return (4 if (m.get("tweet_id") or "").strip() else 0) + (2 if (m.get("author") or "").strip() else 0) + (1 if (m.get("created_at") or "").strip() else 0)

def _collect_video_key_to_tweet_meta_from_json_obj(obj) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    user_screen_by_id: Dict[str, str] = {}
    # Pass 1: build user id -> @screen_name map from any user-like nodes.
    for d in _iter_dicts(obj):
        if not isinstance(d, dict):
            continue
        legacy_u = d.get("legacy")
        if isinstance(legacy_u, dict):
            sn = (legacy_u.get("screen_name") or "").strip()
            if sn:
                sn = sn if sn.startswith("@") else f"@{sn}"
                uid = (d.get("rest_id") or d.get("id_str") or legacy_u.get("id_str") or "").strip()
                if uid.isdigit() and uid not in user_screen_by_id:
                    user_screen_by_id[uid] = sn

    def walk(x, inherited_meta: Dict[str, str]) -> None:
        if isinstance(x, dict):
            local_meta = inherited_meta
            node_meta = _tweet_meta_from_graphql_node(x, user_screen_by_id)
            if node_meta:
                local_meta = _merge_graphql_video_meta(node_meta, inherited_meta)

            urls: List[str] = []
            for v in x.values():
                if isinstance(v, str) and "video.twimg.com/" in v:
                    urls.append(v)
            if urls and (local_meta.get("tweet_id") or "").strip().isdigit():
                for u in urls:
                    mk = normalize_media_key(canon_media_url(u) or "")
                    if not mk:
                        continue
                    prev = out.get(mk, {}) or {}
                    merged = _merge_graphql_video_meta(local_meta, prev)
                    if _graphql_video_meta_score(merged) >= _graphql_video_meta_score(prev):
                        out[mk] = merged

            for v in x.values():
                if isinstance(v, (dict, list)):
                    walk(v, local_meta)
        elif isinstance(x, list):
            for v in x:
                walk(v, inherited_meta)

    walk(obj, {})

    # Fallback: broad per-node scan keeps compatibility with older behavior.
    for d in _iter_dicts(obj):
        if not isinstance(d, dict):
            continue
        meta = _tweet_meta_from_graphql_node(d, user_screen_by_id)
        if not (meta.get("tweet_id") or "").strip().isdigit():
            continue
        urls: List[str] = []
        for sd in _iter_dicts(d):
            if not isinstance(sd, dict):
                continue
            for v in sd.values():
                if isinstance(v, str) and "video.twimg.com/" in v:
                    urls.append(v)
        for u in urls:
            mk = normalize_media_key(canon_media_url(u) or "")
            if not mk:
                continue
            prev = out.get(mk, {}) or {}
            merged = _merge_graphql_video_meta(meta, prev)
            if _graphql_video_meta_score(merged) >= _graphql_video_meta_score(prev):
                out[mk] = merged
    return out

def _tid_confidence(source: str) -> str:
    s = (source or "").strip().lower()
    if s == "graphql":
        return "high"
    if s in ("dom_article_video", "dom_status"):
        return "medium"
    if s:
        return "low"
    return ""

def backfill_tweet_id_from_graphql_bodies(
    cdp_meta_by_key: Dict[str, Dict[str, str]],
    max_bodies: int = 2000,
    quiet: bool = False,
) -> int:
    updated = 0
    attempted = 0
    try:
        logs = driver.get_log("performance")
    except Exception:
        logs = []
    req_ids: List[str] = []
    for entry in logs:
        try:
            msg = json.loads(entry.get("message", "{}")).get("message", {})
            params = msg.get("params", {}) or {}
            method = msg.get("method", "")
            rid = (params.get("requestId") or "").strip()
            if not rid:
                continue
            if method == "Network.requestWillBeSent":
                req = params.get("request", {}) or {}
                u = str(req.get("url") or "")
                if u:
                    CDP_REQ_URL_BY_ID[rid] = u
            elif method == "Network.responseReceived":
                res = params.get("response", {}) or {}
                u = str(res.get("url") or "")
                if u:
                    CDP_REQ_URL_BY_ID[rid] = u
                CDP_RESP_MIME_BY_ID[rid] = str(res.get("mimeType") or "")
            elif method == "Network.loadingFinished":
                CDP_FINISHED_REQ_IDS.add(rid)
        except Exception:
            continue
    for rid, u in list(CDP_REQ_URL_BY_ID.items()):
        ul = (u or "").lower()
        mime = (CDP_RESP_MIME_BY_ID.get(rid, "") or "").lower()
        is_candidate = (
            ("graphql" in ul)
            or ("bookmarks" in ul)
            or ("tweetresultbyrestid" in ul)
            or ("timeline" in ul and "x.com/i/api" in ul)
            or ("x.com/i/api/2/" in ul)
        )
        if is_candidate and (("json" in mime) or (rid in CDP_FINISHED_REQ_IDS)) and rid not in CDP_GRAPHQL_BODY_PARSED_REQ_IDS:
            req_ids.append(rid)
    req_ids = list(dict.fromkeys(req_ids))[-max(1, int(max_bodies)):]
    for rid in req_ids:
        try:
            attempted += 1
            body_obj = driver.execute_cdp_cmd("Network.getResponseBody", {"requestId": rid})
            CDP_GRAPHQL_BODY_PARSED_REQ_IDS.add(rid)
            body = (body_obj or {}).get("body", "")
            if not body:
                continue
            if body.startswith(")]}'"):
                body = body.split("\n", 1)[-1]
            parsed = json.loads(body)
            m = _collect_video_key_to_tweet_meta_from_json_obj(parsed)
            for mk, meta in m.items():
                tid = (meta.get("tweet_id") or "").strip()
                author = (meta.get("author") or "").strip()
                created_at = (meta.get("created_at") or "").strip()
                prev = cdp_meta_by_key.get(mk, {}) or {}
                if not (prev.get("tweet_id") or ""):
                    cdp_meta_by_key[mk] = {
                        "tweet_id": tid,
                        "uploader_name": author or prev.get("uploader_name", ""),
                        "upload_time": created_at or prev.get("upload_time", ""),
                    }
                    updated += 1
                    CDP_TID_SOURCE_BY_MEDIA_KEY[mk] = "graphql"
                elif author and not (prev.get("uploader_name") or ""):
                    prev["uploader_name"] = author
                    cdp_meta_by_key[mk] = prev
                if created_at and not (cdp_meta_by_key.get(mk, {}) or {}).get("upload_time", ""):
                    prev2 = cdp_meta_by_key.get(mk, {}) or {}
                    prev2["upload_time"] = created_at
                    cdp_meta_by_key[mk] = prev2
                if mk and mk not in CDP_TID_BY_MEDIA_KEY:
                    CDP_TID_BY_MEDIA_KEY[mk] = tid
                    CDP_TID_SOURCE_BY_MEDIA_KEY[mk] = "graphql"
        except Exception:
            continue
    if attempted > 0 and not quiet:
        print(f"debug: [GQL_BACKFILL] attemptedBodies={attempted}, updated={updated}")
    return updated

def backfill_tweet_id_from_cdp_video_requests(cdp_meta_by_key: Dict[str, Dict[str, str]]) -> int:
    """
    Fallback mapper: use Network.request/response logs to associate video.twimg media_key
    with nearby status/tweet id context (URL/referrer/documentURL/initiator).
    """
    updated = 0
    seen_req_tid: Dict[str, str] = {}
    seen_req_url: Dict[str, str] = {}
    try:
        logs = driver.get_log("performance")
    except Exception:
        logs = []

    for entry in logs:
        try:
            msg = json.loads(entry.get("message", "{}")).get("message", {})
            method = msg.get("method", "")
            params = msg.get("params", {}) or {}
            rid = (params.get("requestId") or "").strip()
            if not rid:
                continue
            if method == "Network.requestWillBeSent":
                req = params.get("request", {}) or {}
                url = str(req.get("url") or "")
                seen_req_url[rid] = url
                tid = _extract_status_id_from_text(url)
                if not tid:
                    headers = req.get("headers", {}) or {}
                    if isinstance(headers, dict):
                        for hk in ("Referer", "referer", "Origin", "origin"):
                            tid = _extract_status_id_from_text(str(headers.get(hk, "") or ""))
                            if tid:
                                break
                if not tid:
                    doc = str(params.get("documentURL") or "")
                    tid = _extract_status_id_from_text(doc)
                if not tid:
                    init = params.get("initiator", {}) or {}
                    if isinstance(init, dict):
                        tid = _extract_status_id_from_text(str(init.get("url", "") or ""))
                if not tid:
                    tid = _extract_status_id_deep(params)
                if tid:
                    seen_req_tid[rid] = tid
            elif method == "Network.responseReceived":
                res = params.get("response", {}) or {}
                u = str(res.get("url") or "")
                if u:
                    seen_req_url[rid] = u
                if rid not in seen_req_tid:
                    tid = _extract_status_id_deep(params)
                    if tid:
                        seen_req_tid[rid] = tid
        except Exception:
            continue

    for rid, u in seen_req_url.items():
        cu = canon_media_url(u)
        mk = normalize_media_key(cu) or ""
        if not mk:
            continue
        if "video.twimg.com/" not in cu:
            continue
        tid = (seen_req_tid.get(rid) or CDP_TID_BY_MEDIA_KEY.get(mk) or "").strip()
        if not tid:
            continue
        prev = cdp_meta_by_key.get(mk, {}) or {}
        if not (prev.get("tweet_id") or ""):
            cdp_meta_by_key[mk] = {
                "tweet_id": tid,
                "uploader_name": prev.get("uploader_name", ""),
                "upload_time": prev.get("upload_time", ""),
            }
            updated += 1
            CDP_TID_BY_MEDIA_KEY[mk] = tid
            CDP_TID_SOURCE_BY_MEDIA_KEY[mk] = "cdp_request_ctx"
    if updated > 0:
        print(f"debug: [CDP_REQ_BACKFILL] updated={updated}")
    return updated

def drain_cdp_media() -> List[str]:
    return [x.get("url", "") for x in drain_cdp_media_with_meta() if x.get("url")]

def update_cdp_seen_from_logs(cdp_seen_keys: set[str], cdp_url_by_key: Dict[str, str]) -> int:
# CDP drain → 고유 미디어키 집합/URL 맵 갱신. 반환: 이번 호출에서 새로 추가된 key 개수 
    _, added_keys = update_cdp_seen_from_logs_with_keys(cdp_seen_keys, cdp_url_by_key)
    return len(added_keys)

def update_cdp_seen_from_logs_with_keys(
    cdp_seen_keys: set[str],
    cdp_url_by_key: Dict[str, str],
    cdp_meta_by_key: Dict[str, Dict[str, str]] | None = None,
) -> Tuple[int, List[str]]:
# CDP drain → 고유 미디어키 집합/URL 맵 갱신. 반환: (새로 추가된 key 개수, 새 key 목록)
    new_items = drain_cdp_media_with_meta()
    added_keys: List[str] = []
    for it in new_items:
        u = it.get("url", "") or ""
        tid = (it.get("tweet_id", "") or "").strip()
        mk = normalize_media_key(u)
        if not mk:
            continue
        if mk not in cdp_seen_keys:
            cdp_seen_keys.add(mk)
            if mk not in cdp_url_by_key:
                cdp_url_by_key[mk] = u
            added_keys.append(mk)
        else:
            # same key observed again: keep better-quality video variant URL.
            prev = cdp_url_by_key.get(mk, "")
            if _video_quality_score(u) > _video_quality_score(prev):
                cdp_url_by_key[mk] = u
        if cdp_meta_by_key is not None and tid:
            prevm = cdp_meta_by_key.get(mk, {})
            if not (prevm.get("tweet_id") or ""):
                cdp_meta_by_key[mk] = {
                    "tweet_id": tid,
                    "uploader_name": prevm.get("uploader_name", "") if isinstance(prevm, dict) else "",
                    "upload_time": prevm.get("upload_time", "") if isinstance(prevm, dict) else "",
                }
    return len(added_keys), added_keys

def flush_io_buffer() -> List[Dict[str, str]]:
# IO 버퍼를 비우고 표준화된 dict 목록으로 반환. 
    try:
        arr = driver.execute_script("return (window.__xDump && window.__xDump()) || []") or []
    except Exception:
        arr = []
    out = []
    for it in arr:
        try:
            u = it.get("url") if isinstance(it, dict) else ""
            if not u:
                continue
            out.append({
                "url": canon_media_url(u),
                "uploader_name": (it.get("uploader_name") or "") if isinstance(it, dict) else "",
                "upload_time": (it.get("upload_time") or "") if isinstance(it, dict) else "",
                "tweet_id": (it.get("tweet_id") or "") if isinstance(it, dict) else "",
            })
        except Exception:
            continue
    return out

def collect_visible_tweet_ids(viewport_pad: int = VIEWPORT_PAD) -> List[str]:
    js = r"""
const pad = arguments[0];
const topY = window.scrollY - pad;
const bottomY = window.scrollY + window.innerHeight + pad;
function inRange(el) {
  const r = el.getBoundingClientRect();
  const y1 = window.scrollY + r.top;
  const y2 = window.scrollY + r.bottom;
  return (y2 >= topY && y1 <= bottomY);
}
const ids = [];
const seen = new Set();
const arts = document.querySelectorAll('article');
for (const art of arts) {
  if (!inRange(art)) continue;
  let tid = '';
  const links = art.querySelectorAll('a[href*="/status/"]');
  for (const a of links) {
    const href = a.getAttribute('href') || '';
    const m = href.match(/\/status\/(\d+)/);
    if (m) { tid = m[1]; break; }
  }
  if (!tid) {
    const html = art.innerHTML || '';
    const m2 = html.match(/\/status\/(\d+)/);
    if (m2) tid = m2[1];
  }
  if (tid && !seen.has(tid)) {
    seen.add(tid);
    ids.push(tid);
  }
}
return ids;
"""
    try:
        out = driver.execute_script(js, viewport_pad) or []
        return [str(x).strip() for x in out if str(x).strip().isdigit()]
    except Exception:
        return []

def collect_visible_author_by_tweet_id(viewport_pad: int = VIEWPORT_PAD) -> Dict[str, str]:
    js = r"""
const pad = arguments[0];
const topY = window.scrollY - pad;
const bottomY = window.scrollY + window.innerHeight + pad;
function inRange(el) {
  const r = el.getBoundingClientRect();
  const y1 = window.scrollY + r.top;
  const y2 = window.scrollY + r.bottom;
  return (y2 >= topY && y1 <= bottomY);
}
const out = {};
const arts = document.querySelectorAll('article');
for (const art of arts) {
  if (!inRange(art)) continue;
  let tid = '';
  const links = art.querySelectorAll('a[href*="/status/"]');
  for (const a of links) {
    const href = a.getAttribute('href') || '';
    const m = href.match(/\/status\/(\d+)/);
    if (m) { tid = m[1]; break; }
  }
  if (!tid) continue;
  let author = '';
  const userNameBox = art.querySelector('[data-testid="User-Name"]');
  if (userNameBox) {
    const spans2 = userNameBox.querySelectorAll('span');
    for (const s of spans2) {
      const t = (s.textContent || '').trim();
      if (t.startsWith('@')) { author = t; break; }
    }
  }
  if (!author) {
    const spans = art.querySelectorAll('span');
    for (const s of spans) {
      const t = (s.textContent || '').trim();
      if (t.startsWith('@')) { author = t; break; }
    }
  }
  if (tid && author && !out[tid]) out[tid] = author;
}
return out;
"""
    try:
        m = driver.execute_script(js, viewport_pad) or {}
        if isinstance(m, dict):
            return {str(k): str(v) for k, v in m.items() if str(k).isdigit() and str(v).startswith("@")}
    except Exception:
        pass
    return {}

def collect_visible_time_by_tweet_id(viewport_pad: int = VIEWPORT_PAD) -> Dict[str, str]:
    js = r"""
const pad = arguments[0];
const topY = window.scrollY - pad;
const bottomY = window.scrollY + window.innerHeight + pad;
function inRange(el) {
  const r = el.getBoundingClientRect();
  const y1 = window.scrollY + r.top;
  const y2 = window.scrollY + r.bottom;
  return (y2 >= topY && y1 <= bottomY);
}
const out = {};
const arts = document.querySelectorAll('article');
for (const art of arts) {
  if (!inRange(art)) continue;
  let tid = '';
  const links = art.querySelectorAll('a[href*="/status/"]');
  for (const a of links) {
    const href = a.getAttribute('href') || '';
    const m = href.match(/\/status\/(\d+)/);
    if (m) { tid = m[1]; break; }
  }
  if (!tid) continue;
  let dt = '';
  const t = art.querySelector('time');
  if (t && t.getAttribute('datetime')) dt = t.getAttribute('datetime');
  if (tid && dt && !out[tid]) out[tid] = dt;
}
return out;
"""
    try:
        m = driver.execute_script(js, viewport_pad) or {}
        if isinstance(m, dict):
            return {str(k): str(v) for k, v in m.items() if str(k).isdigit() and str(v).strip()}
    except Exception:
        pass
    return {}

def backfill_authors_for_video_entries(entries: List[Dict[str, str]], rounds: int = 36) -> int:
    need = { (e.get("tweet_id") or "").strip() for e in entries if (e.get("tweet_id") or "").strip().isdigit() and not (e.get("uploader_name") or "").strip() }
    if not need:
        return 0
    y0, y1 = ensure_repair_bottom_start("AUTHOR_BACKFILL")
    print(f"message: AUTHOR_BACKFILL start yOffset {y0} -> {y1}")
    found: Dict[str, str] = {}
    vh = _get_vh()
    # Use SAFE-like upward coverage so author-only pass scans the whole region.
    step = max(220, min(UP_STEP_PX, int(vh * (1.0 - SAFE_OVERLAP_RATIO))))
    max_steps = _dynamic_upward_scan_limit(y1, step, rounds=rounds, margin_steps=120)
    print(f"message: AUTHOR_BACKFILL scan step={step}, maxSteps={max_steps}")
    i = 0
    with tqdm(total=0, desc="AuthorBackfill", unit="step", dynamic_ncols=True, leave=True) as pbar:
        while True:
            i += 1
            m = collect_visible_author_by_tweet_id()
            for tid in list(need):
                if tid in m:
                    found[tid] = m[tid]
                    need.discard(tid)
            y_now = _get_scroll_y()
            pbar.update(1)
            pbar.set_postfix({"y": y_now, "found": len(found), "remain": len(need)}, refresh=False)
            if not need or y_now <= 2:
                break
            driver.execute_script("window.scrollBy(0, arguments[0]);", -int(step))
            time.sleep(max(0.08, UP_DELAY_S))
            if i >= max_steps:
                break

    updated = 0
    for e in entries:
        tid = (e.get("tweet_id") or "").strip()
        if tid in found and not (e.get("uploader_name") or "").strip():
            e["uploader_name"] = found[tid]
            updated += 1
    return updated

def backfill_authors_in_video_rows(rows: List[Dict[str, str]], rounds: int = 48) -> int:
    entries = []
    for r in rows:
        entries.append({
            "tweet_id": (r.get("tweet_id") or "").strip(),
            "uploader_name": (r.get("author") or "").strip(),
        })
    updated = backfill_authors_for_video_entries(entries, rounds=rounds)
    if updated <= 0:
        return 0
    changed = 0
    for i, r in enumerate(rows):
        new_author = (entries[i].get("uploader_name") or "").strip()
        if new_author and not (r.get("author") or "").strip():
            r["author"] = new_author
            changed += 1
    return changed

def backfill_created_at_in_video_rows(rows: List[Dict[str, str]], rounds: int = 60) -> int:
    need = { (r.get("tweet_id") or "").strip() for r in rows if (r.get("tweet_id") or "").strip().isdigit() and not (r.get("created_at") or "").strip() }
    if not need:
        return 0
    y0, y1 = ensure_repair_bottom_start("CREATEDAT_BACKFILL")
    print(f"message: CREATEDAT_BACKFILL start yOffset {y0} -> {y1}")
    found: Dict[str, str] = {}
    vh = _get_vh()
    step = max(220, min(UP_STEP_PX, int(vh * (1.0 - SAFE_OVERLAP_RATIO))))
    max_steps = _dynamic_upward_scan_limit(y1, step, rounds=rounds, margin_steps=120)
    print(f"message: CREATEDAT_BACKFILL scan step={step}, maxSteps={max_steps}")
    i = 0
    with tqdm(total=0, desc="CreatedAtBackfill", unit="step", dynamic_ncols=True, leave=True) as pbar:
        while True:
            i += 1
            m = collect_visible_time_by_tweet_id()
            for tid in list(need):
                if tid in m:
                    found[tid] = m[tid]
                    need.discard(tid)
            y_now = _get_scroll_y()
            pbar.update(1)
            pbar.set_postfix({"y": y_now, "found": len(found), "remain": len(need)}, refresh=False)
            if not need or y_now <= 2:
                break
            driver.execute_script("window.scrollBy(0, arguments[0]);", -int(step))
            time.sleep(max(0.08, UP_DELAY_S))
            if i >= max_steps:
                break
    updated = 0
    for r in rows:
        tid = (r.get("tweet_id") or "").strip()
        if tid in found and not (r.get("created_at") or "").strip():
            r["created_at"] = found[tid]
            r["created_at_norm"] = normalize_time(r["created_at"])
            updated += 1
    return updated

def warmup_scroll_for_repair(cdp_seen_keys: set[str], cdp_url_by_key: Dict[str, str], cdp_meta_by_key: Dict[str, Dict[str, str]], cycles: int = 40) -> None:
    vh = _get_vh()
    step_px = max(220, min(DOWN_STEP_PX, int(0.70 * vh)))
    print(f"message: REPAIR warmup start ({cycles} down + {cycles} up), step={step_px}")
    with tqdm(total=cycles, desc="RepairWarmup-Down", unit="step", dynamic_ncols=True, leave=True) as pbar_down:
        for _ in range(cycles):
            driver.execute_script("window.scrollBy(0, arguments[0]);", step_px)
            time.sleep(DOWN_DELAY_S)
            new_added, _ = update_cdp_seen_from_logs_with_keys(cdp_seen_keys, cdp_url_by_key, cdp_meta_by_key=cdp_meta_by_key)
            pbar_down.update(1)
            pbar_down.set_postfix({
                "cdpNew": new_added,
                "cdpKeys": len(cdp_seen_keys),
                "y": _get_scroll_y(),
            }, refresh=False)
    with tqdm(total=cycles, desc="RepairWarmup-Up", unit="step", dynamic_ncols=True, leave=True) as pbar_up:
        for _ in range(cycles):
            driver.execute_script("window.scrollBy(0, arguments[0]);", -step_px)
            time.sleep(DOWN_DELAY_S)
            new_added, _ = update_cdp_seen_from_logs_with_keys(cdp_seen_keys, cdp_url_by_key, cdp_meta_by_key=cdp_meta_by_key)
            pbar_up.update(1)
            pbar_up.set_postfix({
                "cdpNew": new_added,
                "cdpKeys": len(cdp_seen_keys),
                "y": _get_scroll_y(),
            }, refresh=False)

def warmup_scroll_for_repair_full(
    cdp_seen_keys: set[str],
    cdp_url_by_key: Dict[str, str],
    cdp_meta_by_key: Dict[str, Dict[str, str]],
    return_to_top: bool = True,
) -> None:
    vh = _get_vh()
    step_px = max(220, min(DOWN_STEP_PX, int(0.70 * vh)))
    print(f"message: REPAIR warmup FULL start, step={step_px}")

    # Use the same descent engine as normal FULL mode to avoid early-stop mismatches.
    desc_meta_dummy: Dict[str, Dict[str, str]] = {}
    _ = full_descent(
        cdp_seen_keys,
        cdp_url_by_key,
        desc_meta_dummy,
        stop_when_seen_keys=None,
        cdp_meta_by_key=cdp_meta_by_key,
    )

    if not return_to_top:
        return

    # up to top
    with tqdm(total=0, desc="RepairWarmup-UpFull", unit="step", dynamic_ncols=True, leave=True) as pbar_up:
        while True:
            y_prev = _get_scroll_y()
            if y_prev <= 2:
                break
            driver.execute_script("window.scrollBy(0, arguments[0]);", -step_px)
            time.sleep(DOWN_DELAY_S)
            new_added, _ = update_cdp_seen_from_logs_with_keys(cdp_seen_keys, cdp_url_by_key, cdp_meta_by_key=cdp_meta_by_key)
            y = _get_scroll_y()
            pbar_up.update(1)
            pbar_up.set_postfix({"cdpNew": new_added, "cdpKeys": len(cdp_seen_keys), "y": y}, refresh=False)
            if y >= y_prev - 1 and y <= 2:
                break

def collect_visible_status_url_by_video_key(viewport_pad: int = VIEWPORT_PAD) -> Dict[str, str]:
    js = r"""
const pad = arguments[0];
const topY = window.scrollY - pad;
const bottomY = window.scrollY + window.innerHeight + pad;
function inRange(el) {
  const r = el.getBoundingClientRect();
  const y1 = window.scrollY + r.top;
  const y2 = window.scrollY + r.bottom;
  return (y2 >= topY && y1 <= bottomY);
}
const out = {};
const arts = document.querySelectorAll('article');
for (const art of arts) {
  if (!inRange(art)) continue;
  let statusUrl = '';
  const links = art.querySelectorAll('a[href*="/status/"]');
  for (const a of links) {
    const href = a.getAttribute('href') || '';
    const m = href.match(/\/status\/\d+/);
    if (m) { statusUrl = href; break; }
  }
  if (!statusUrl) continue;
  const vids = art.querySelectorAll('video[src], video source[src]');
  for (const vd of vids) {
    const src = vd.getAttribute('src') || '';
    if (!src) continue;
    out[src] = statusUrl;
  }
}
return out;
"""
    ret: Dict[str, str] = {}
    try:
        m = driver.execute_script(js, viewport_pad) or {}
        if isinstance(m, dict):
            for u, s in m.items():
                cu = canon_media_url(str(u or ""))
                ret[cu] = str(s or "")
    except Exception:
        return {}
    return ret

def collect_visible_status_by_media_key(viewport_pad: int = VIEWPORT_PAD) -> Dict[str, str]:
    by_url = collect_visible_status_url_by_video_key(viewport_pad)
    out: Dict[str, str] = {}
    for u, s in by_url.items():
        mk = normalize_media_key(u) or ""
        if not mk or not s:
            continue
        if mk not in out:
            out[mk] = s
    return out

def collect_visible_video_key_to_tweet_id(viewport_pad: int = VIEWPORT_PAD) -> Dict[str, str]:
    js = r"""
const pad = arguments[0];
const topY = window.scrollY - pad;
const bottomY = window.scrollY + window.innerHeight + pad;
function inRange(el) {
  const r = el.getBoundingClientRect();
  const y1 = window.scrollY + r.top;
  const y2 = window.scrollY + r.bottom;
  return (y2 >= topY && y1 <= bottomY);
}
const out = {};
const arts = document.querySelectorAll('article');
for (const art of arts) {
  if (!inRange(art)) continue;
  let tid = '';
  const links = art.querySelectorAll('a[href*="/status/"]');
  for (const a of links) {
    const href = a.getAttribute('href') || '';
    const m = href.match(/\/status\/(\d+)/);
    if (m) { tid = m[1]; break; }
  }
  if (!tid) {
    const html = art.innerHTML || '';
    const m2 = html.match(/\/status\/(\d+)/);
    if (m2) tid = m2[1];
  }
  if (!tid) continue;
  const vids = art.querySelectorAll('video, video source');
  for (const vd of vids) {
    const cands = [];
    const src = vd.getAttribute('src') || '';
    if (src) cands.push(src);
    const poster = vd.getAttribute('poster') || '';
    if (poster) cands.push(poster);
    const cs = vd.currentSrc || '';
    if (cs) cands.push(cs);
    for (const u of cands) out[u] = tid;
  }
}
return out;
"""
    out: Dict[str, str] = {}
    try:
        raw = driver.execute_script(js, viewport_pad) or {}
        if not isinstance(raw, dict):
            return out
        for u, tid in raw.items():
            cu = canon_media_url(str(u or ""))
            mk = normalize_media_key(cu) or ""
            t = str(tid or "").strip()
            if mk and t.isdigit():
                out[mk] = t
    except Exception:
        return {}
    return out

def collect_visible_video_key_to_meta(viewport_pad: int = VIEWPORT_PAD) -> Dict[str, Dict[str, str]]:
    js = r"""
const pad = arguments[0];
const topY = window.scrollY - pad;
const bottomY = window.scrollY + window.innerHeight + pad;
function inRange(el) {
  const r = el.getBoundingClientRect();
  const y1 = window.scrollY + r.top;
  const y2 = window.scrollY + r.bottom;
  return (y2 >= topY && y1 <= bottomY);
}
const out = {};
const arts = document.querySelectorAll('article');
for (const art of arts) {
  if (!inRange(art)) continue;
  let tid = '';
  const links = art.querySelectorAll('a[href*="/status/"]');
  for (const a of links) {
    const href = a.getAttribute('href') || '';
    const m = href.match(/\/status\/(\d+)/);
    if (m) { tid = m[1]; break; }
  }
  let author = '';
  const userNameBox = art.querySelector('[data-testid="User-Name"]');
  if (userNameBox) {
    const spans2 = userNameBox.querySelectorAll('span');
    for (const s of spans2) {
      const t = (s.textContent || '').trim();
      if (t.startsWith('@')) { author = t; break; }
    }
  }
  if (!author) {
    const spans = art.querySelectorAll('span');
    for (const s of spans) {
      const t = (s.textContent || '').trim();
      if (t.startsWith('@')) { author = t; break; }
    }
  }
  let dt = '';
  const tm = art.querySelector('time');
  if (tm && tm.getAttribute('datetime')) dt = tm.getAttribute('datetime');

  const vids = art.querySelectorAll('video, video source');
  for (const vd of vids) {
    const cands = [];
    const src = vd.getAttribute('src') || '';
    if (src) cands.push(src);
    const poster = vd.getAttribute('poster') || '';
    if (poster) cands.push(poster);
    const cs = vd.currentSrc || '';
    if (cs) cands.push(cs);
    for (const u of cands) {
      out[u] = { tweet_id: tid || '', author: author || '', created_at: dt || '' };
    }
  }
}
return out;
"""
    out: Dict[str, Dict[str, str]] = {}
    try:
        raw = driver.execute_script(js, viewport_pad) or {}
        if not isinstance(raw, dict):
            return out
        for u, m in raw.items():
            cu = canon_media_url(str(u or ""))
            mk = normalize_media_key(cu) or ""
            if not mk:
                continue
            md = m if isinstance(m, dict) else {}
            out[mk] = {
                "tweet_id": str(md.get("tweet_id", "") or "").strip(),
                "author": str(md.get("author", "") or "").strip(),
                "created_at": str(md.get("created_at", "") or "").strip(),
            }
    except Exception:
        return {}
    return out

def _media_key_aliases(mk: str) -> List[str]:
    m = re.match(r"^(ext_tw_video|amplify_video)_(\d+)$", (mk or "").strip())
    if not m:
        return [mk] if mk else []
    num = m.group(2)
    return [f"ext_tw_video_{num}", f"amplify_video_{num}"]

def _media_numeric_id(mk: str) -> str:
    m = re.match(r"^(ext_tw_video|amplify_video)_(\d+)$", (mk or "").strip())
    return m.group(2) if m else ""

def _lookup_meta_by_video_key_alias(
    mk: str,
    meta_by_key: Dict[str, Dict[str, str]] | None,
) -> Tuple[Dict[str, str], str]:
    """Find video meta by exact key, ext/amplify alias, then numeric id."""
    if not meta_by_key:
        return {}, ""
    candidates: List[str] = []
    for ak in _media_key_aliases(mk):
        if ak and ak not in candidates:
            candidates.append(ak)
    if mk and mk not in candidates:
        candidates.insert(0, mk)

    best: Dict[str, str] = {}
    best_key = ""
    for ck in candidates:
        cm = meta_by_key.get(ck, {}) or {}
        if _meta_score(cm) > _meta_score(best):
            best = cm
            best_key = ck
    if best:
        return best, best_key

    kid = _media_numeric_id(mk)
    if not kid:
        return {}, ""
    for ck, cm in meta_by_key.items():
        if _media_numeric_id(ck) != kid:
            continue
        if _meta_score(cm or {}) > _meta_score(best):
            best = cm or {}
            best_key = ck
    return best, best_key

def _source_for_video_key_alias(mk: str, matched_key: str = "") -> str:
    for ck in [matched_key] + _media_key_aliases(mk):
        if ck and CDP_TID_SOURCE_BY_MEDIA_KEY.get(ck):
            return CDP_TID_SOURCE_BY_MEDIA_KEY.get(ck, "")
    kid = _media_numeric_id(mk)
    if kid:
        for ck, src in CDP_TID_SOURCE_BY_MEDIA_KEY.items():
            if src and _media_numeric_id(ck) == kid:
                return src
    return ""

def collect_visible_tid_by_keyid_probe(key_ids: List[str], viewport_pad: int = VIEWPORT_PAD) -> Dict[str, str]:
    ids = [str(x).strip() for x in key_ids if str(x).strip().isdigit()]
    if not ids:
        return {}
    js = r"""
const ids = arguments[0] || [];
const pad = arguments[1];
const topY = window.scrollY - pad;
const bottomY = window.scrollY + window.innerHeight + pad;
function inRange(el) {
  const r = el.getBoundingClientRect();
  const y1 = window.scrollY + r.top;
  const y2 = window.scrollY + r.bottom;
  return (y2 >= topY && y1 <= bottomY);
}
const out = {};
const arts = document.querySelectorAll('article');
for (const art of arts) {
  if (!inRange(art)) continue;
  let tid = '';
  const links = art.querySelectorAll('a[href*="/status/"]');
  for (const a of links) {
    const href = a.getAttribute('href') || '';
    const m = href.match(/\/status\/(\d+)/);
    if (m) { tid = m[1]; break; }
  }
  if (!tid) continue;
  // Strict mode: key id must be observed in VIDEO-related URLs inside the same article.
  const cands = [];
  const vids = art.querySelectorAll('video, video source');
  for (const vd of vids) {
    const src = vd.getAttribute('src') || '';
    if (src) cands.push(src);
    const poster = vd.getAttribute('poster') || '';
    if (poster) cands.push(poster);
    const cs = vd.currentSrc || '';
    if (cs) cands.push(cs);
  }
  // fallback: parse direct video.twimg.com links inside anchors (still article-local)
  const links2 = art.querySelectorAll('a[href*="video.twimg.com/"]');
  for (const a of links2) {
    const href = a.getAttribute('href') || '';
    if (href) cands.push(href);
  }
  for (const id of ids) {
    if (out[id]) continue;
    let hit = false;
    for (const u of cands) {
      if (!u) continue;
      if (u.includes('/amplify_video/' + id + '/') || u.includes('/ext_tw_video/' + id + '/')) {
        hit = true; break;
      }
    }
    if (hit) out[id] = tid;
  }
}
return out;
"""
    try:
        out = driver.execute_script(js, ids, viewport_pad) or {}
        if isinstance(out, dict):
            return {str(k): str(v) for k, v in out.items() if str(v).isdigit()}
    except Exception:
        pass
    return {}

def backfill_cdp_tweet_id_from_visible_context(new_keys: List[str], cdp_meta_by_key: Dict[str, Dict[str, str]]) -> int:
    if not new_keys:
        return 0
    key_tid_map = collect_visible_video_key_to_tweet_id()
    updated = 0
    for mk in new_keys:
        tid = key_tid_map.get(mk, "")
        if not tid:
            for ak in _media_key_aliases(mk):
                tid = key_tid_map.get(ak, "")
                if tid:
                    break
        if not tid:
            continue
        prev = cdp_meta_by_key.get(mk, {}) or {}
        if not (prev.get("tweet_id") or ""):
            cdp_meta_by_key[mk] = {
                "tweet_id": tid,
                "uploader_name": prev.get("uploader_name", ""),
                "upload_time": prev.get("upload_time", ""),
            }
            updated += 1
            CDP_TID_SOURCE_BY_MEDIA_KEY[mk] = "dom_article_video"
    if updated > 0:
        return updated

    tids = collect_visible_tweet_ids()
    if len(tids) != 1:
        return 0
    tid = tids[0]
    for mk in new_keys:
        prev = cdp_meta_by_key.get(mk, {}) or {}
        if not (prev.get("tweet_id") or ""):
            cdp_meta_by_key[mk] = {
                "tweet_id": tid,
                "uploader_name": prev.get("uploader_name", ""),
                "upload_time": prev.get("upload_time", ""),
            }
            updated += 1
            CDP_TID_SOURCE_BY_MEDIA_KEY[mk] = "dom_status"
    return updated

def repair_video_items_meta_focus(items_path: str, rounds: int = 16) -> int:
    rows = read_local_items_ndjson(items_path)
    by_key: Dict[str, Dict[str, str]] = {}
    for r in rows:
        mk = (r.get("media_key") or "").strip() or (normalize_media_key(r.get("url", "") or "") or "")
        if mk:
            r["media_key"] = mk
            by_key[mk] = r
    targets = {mk for mk, r in by_key.items() if _row_missing_video_meta(r)}
    if not targets:
        return 0

    updated = 0
    vh = _get_vh()
    step = max(260, int(vh * 0.55))
    ensure_bottom_start("RepairFocus")
    idle = 0
    max_steps = max(4000, rounds * 40)
    with tqdm(total=0, desc="RepairFocus", unit="step", dynamic_ncols=True, leave=True) as pbar:
        for _ in range(max_steps):
            # supplemental mapping: strict key-id + article video->tid + status-url parse
            mapping_by_key = collect_visible_status_by_media_key()
            key_tid_map = collect_visible_video_key_to_tweet_id()
            probe = collect_visible_tid_by_keyid_probe([_media_numeric_id(k) for k in targets if _media_numeric_id(k)])
            step_updated = 0
            for mk in list(targets):
                row = by_key.get(mk)
                if not row:
                    continue
                tid = ""
                for ak in _media_key_aliases(mk):
                    if not tid:
                        status = mapping_by_key.get(ak, "")
                        tid = _extract_status_id_from_text(status)
                    if not tid:
                        tid = key_tid_map.get(ak, "")
                    if tid:
                        break
                if not tid:
                    kid = _media_numeric_id(mk)
                    if kid:
                        tid = probe.get(kid, "")
                if tid and (row.get("tweet_id", "") or "") != tid:
                    row["tweet_id"] = tid
                    row["tid_source"] = "focus_mix"
                    row["tid_confidence"] = _tid_confidence("dom_status")
                    updated += 1
                    step_updated += 1
                if not _row_missing_video_meta(row):
                    targets.discard(mk)
            y_now = _get_scroll_y()
            pbar.update(1)
            pbar.set_postfix({"y": y_now, "unresolved": len(targets), "updated": updated}, refresh=False)
            if not targets:
                break
            if step_updated == 0:
                idle += 1
            else:
                idle = 0
            if y_now <= 2 and idle >= 6:
                break
            driver.execute_script("window.scrollBy(0, arguments[0]);", -int(step))
            time.sleep(max(0.10, UP_DELAY_S))

    _rewrite_video_items_ndjson(items_path, list(by_key.values()))
    return updated

def repair_video_items_meta_strict(items_path: str, rounds: int = 24) -> int:
    rows = read_local_items_ndjson(items_path)
    by_key: Dict[str, Dict[str, str]] = {}
    for r in rows:
        mk = (r.get("media_key") or "").strip() or (normalize_media_key(r.get("url", "") or "") or "")
        if mk:
            r["media_key"] = mk
            by_key[mk] = r
    unresolved = {mk for mk, r in by_key.items() if _row_missing_video_meta(r)}
    if not unresolved:
        return 0

    vh = _get_vh()
    step = max(220, int(vh * 0.50))
    updated = 0
    ensure_bottom_start("RepairStrict")
    key_ids = {mk: _media_numeric_id(mk) for mk in unresolved}
    idle = 0
    max_steps = max(4000, rounds * 40)
    with tqdm(total=0, desc="RepairStrict", unit="step", dynamic_ncols=True, leave=True) as pbar:
        for _ in range(max_steps):
            probe = collect_visible_tid_by_keyid_probe(list(key_ids.values()))
            step_updated = 0
            for mk in list(unresolved):
                kid = key_ids.get(mk, "")
                tid = probe.get(kid, "")
                if not tid:
                    continue
                row = by_key.get(mk)
                if not row:
                    continue
                # Strict repair never overrides an existing tweet_id to avoid false-positive corruption.
                if (row.get("tweet_id", "") or ""):
                    if not _row_missing_video_meta(row):
                        unresolved.discard(mk)
                    continue
                if (row.get("tweet_id", "") or "") != tid:
                    row["tweet_id"] = tid
                    updated += 1
                    step_updated += 1
                if not _row_missing_video_meta(row):
                    unresolved.discard(mk)
            y_now = _get_scroll_y()
            pbar.update(1)
            pbar.set_postfix({"y": y_now, "unresolved": len(unresolved), "updated": updated}, refresh=False)
            if not unresolved:
                break
            if step_updated == 0:
                idle += 1
            else:
                idle = 0
            if y_now <= 2 and idle >= 6:
                break
            driver.execute_script("window.scrollBy(0, arguments[0]);", -int(step))
            time.sleep(max(0.08, UP_DELAY_S))
    _rewrite_video_items_ndjson(items_path, list(by_key.values()))
    return updated

def _collect_repair_video_dom_signals(ids: List[str], rounds: int = 3) -> Tuple[
    Dict[str, str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, Dict[str, str]],
    Dict[str, str],
    Dict[str, str],
]:
    probe_all: Dict[str, str] = {}
    key_tid_all: Dict[str, str] = {}
    status_all: Dict[str, str] = {}
    key_meta_all: Dict[str, Dict[str, str]] = {}
    auth_all: Dict[str, str] = {}
    time_all: Dict[str, str] = {}
    for i in range(max(1, rounds)):
        probe_all.update({k: v for k, v in collect_visible_tid_by_keyid_probe(ids).items() if v})
        key_tid_all.update({k: v for k, v in collect_visible_video_key_to_tweet_id().items() if v})
        status_all.update({k: v for k, v in collect_visible_status_by_media_key().items() if v})
        for k, v in collect_visible_video_key_to_meta().items():
            prev = key_meta_all.get(k, {}) or {}
            key_meta_all[k] = pick_better_meta(
                {
                    "tweet_id": prev.get("tweet_id", ""),
                    "uploader_name": prev.get("author", ""),
                    "upload_time": prev.get("created_at", ""),
                },
                {
                    "tweet_id": v.get("tweet_id", ""),
                    "uploader_name": v.get("author", ""),
                    "upload_time": v.get("created_at", ""),
                },
            )
            key_meta_all[k] = {
                "tweet_id": key_meta_all[k].get("tweet_id", ""),
                "author": key_meta_all[k].get("uploader_name", ""),
                "created_at": key_meta_all[k].get("upload_time", ""),
            }
        auth_all.update({k: v for k, v in collect_visible_author_by_tweet_id().items() if v})
        time_all.update({k: v for k, v in collect_visible_time_by_tweet_id().items() if v})
        if i < rounds - 1:
            time.sleep(max(0.12, UP_DELAY_S * 2))
    return probe_all, key_tid_all, status_all, key_meta_all, auth_all, time_all

def backfill_video_tid_by_keyid_sweep(
    rows_by_key: Dict[str, Dict[str, str]],
    max_steps: int = 2400,
    assume_at_bottom: bool = False,
) -> int:
    unresolved = {mk for mk, r in rows_by_key.items() if _row_missing_video_meta(r)}
    if not unresolved:
        return 0
    key_ids = {mk: _media_numeric_id(mk) for mk in unresolved}
    if not assume_at_bottom:
        y0, y1 = ensure_bottom_start("TID_SWEEP")
        print(f"message: TID_SWEEP start yOffset {y0} -> {y1}, unresolved={len(unresolved)}")
    else:
        y0, y1 = ensure_bottom_start("TID_SWEEP")
        print(f"message: TID_SWEEP start-from-bottom unresolved={len(unresolved)}, yOffset={y1}")
    vh = _get_vh()
    step = max(220, min(UP_STEP_PX, int(vh * (1.0 - SAFE_OVERLAP_RATIO))))
    updated = 0
    author_updated = 0
    created_updated = 0
    no_progress_seq = 0
    with tqdm(total=0, desc="RepairTidSweep", unit="step", dynamic_ncols=True, leave=True) as pbar:
        for _ in range(max_steps):
            ids = [kid for mk, kid in key_ids.items() if mk in unresolved and kid]
            if not ids:
                break
            probe, key_tid_map, status_by_key, key_meta_map, vis_auth, vis_time = _collect_repair_video_dom_signals(ids, rounds=3)
            hit = 0
            source_signal = 0
            for mk in list(unresolved):
                kid = key_ids.get(mk, "")
                if not kid:
                    continue
                tid = probe.get(kid, "")
                if not tid:
                    for ak in _media_key_aliases(mk):
                        tid = key_tid_map.get(ak, "")
                        if not tid:
                            tid = _extract_status_id_from_text(status_by_key.get(ak, ""))
                        if tid:
                            break
                if tid:
                    source_signal += 1
                if not tid:
                    continue
                row = rows_by_key.get(mk)
                if not row:
                    continue
                if not (row.get("tweet_id", "") or "").strip():
                    row["tweet_id"] = tid
                    row["tid_source"] = "dom_status"
                    row["tid_confidence"] = _tid_confidence("dom_status")
                    updated += 1
                    hit += 1
                tid_now = (row.get("tweet_id", "") or "").strip()
                if tid_now:
                    if not (row.get("author", "") or "").strip():
                        for ak in _media_key_aliases(mk):
                            a2 = (key_meta_map.get(ak, {}) or {}).get("author", "")
                            if a2:
                                row["author"] = a2
                                author_updated += 1
                                break
                    if not (row.get("author", "") or "").strip():
                        a = vis_auth.get(tid_now, "")
                        if a:
                            row["author"] = a
                            author_updated += 1
                    if not (row.get("created_at", "") or "").strip():
                        for ak in _media_key_aliases(mk):
                            t2 = (key_meta_map.get(ak, {}) or {}).get("created_at", "")
                            if t2:
                                row["created_at"] = t2
                                row["created_at_norm"] = normalize_time(t2)
                                created_updated += 1
                                break
                    if not (row.get("created_at", "") or "").strip():
                        t = vis_time.get(tid_now, "")
                        if t:
                            row["created_at"] = t
                            row["created_at_norm"] = normalize_time(t)
                            created_updated += 1
                if not _row_missing_video_meta(row):
                    unresolved.discard(mk)
            y_now = _get_scroll_y()
            if hit == 0 and source_signal == 0:
                no_progress_seq += 1
            else:
                no_progress_seq = 0
            pbar.update(1)
            pbar.set_postfix(
                {
                    "y": y_now,
                    "hit": hit,
                    "sig": source_signal,
                    "idle": no_progress_seq,
                    "tid": updated,
                    "auth": author_updated,
                    "time": created_updated,
                    "unresolved": len(unresolved),
                },
                refresh=False,
            )
            if not unresolved:
                break
            if y_now <= 2 and no_progress_seq >= 8:
                break
            driver.execute_script("window.scrollBy(0, arguments[0]);", -int(step))
            time.sleep(max(0.08, UP_DELAY_S))
    if author_updated > 0 or created_updated > 0:
        print(f"message: TID_SWEEP meta-updated author={author_updated}, createdAt={created_updated}")
    return updated

def _meta_score(it: Dict[str, str]) -> int:
    s = 0
    if (it.get("tweet_id") or "").strip():
        s += 4
    if (it.get("uploader_name") or "").strip():
        s += 2
    if (it.get("upload_time") or "").strip():
        s += 1
    return s

def _meta_core_ready(meta: Dict[str, str] | None) -> bool:
    if not isinstance(meta, dict):
        return False
    tid = (meta.get("tweet_id") or "").strip()
    author = (meta.get("uploader_name") or meta.get("author") or "").strip()
    ctime = (meta.get("upload_time") or meta.get("created_at") or "").strip()
    return bool(tid and author and ctime)

def _count_ready_meta_for_keys(
    keys: set[str],
    primary_meta: Dict[str, Dict[str, str]] | None,
    secondary_meta: Dict[str, Dict[str, str]] | None = None,
) -> int:
    p = primary_meta or {}
    s = secondary_meta or {}
    ready = 0
    for k in keys:
        m1 = p.get(k, {}) if isinstance(p, dict) else {}
        m2 = s.get(k, {}) if isinstance(s, dict) else {}
        merged = pick_better_meta(m1, m2)
        if _meta_core_ready(merged):
            ready += 1
    return ready

def pick_better_meta(a: Dict[str, str] | None, b: Dict[str, str] | None) -> Dict[str, str]:
    aa = a or {}
    bb = b or {}
    if _meta_score(bb) > _meta_score(aa):
        base = dict(bb)
        other = aa
    else:
        base = dict(aa)
        other = bb
    for fld in ("uploader_name", "upload_time", "tweet_id"):
        if not (base.get(fld) or "") and (other.get(fld) or ""):
            base[fld] = other.get(fld) or ""
    return {
        "uploader_name": base.get("uploader_name", "") or "",
        "upload_time": base.get("upload_time", "") or "",
        "tweet_id": base.get("tweet_id", "") or "",
    }

def enrich_entries_with_dom_meta(entries: List[Dict[str, str]], viewport_pad: int = VIEWPORT_PAD) -> int:
    """
    One-shot meta enrichment pass.
    Re-snap DOM + IO buffer and fill missing uploader/time/tweet_id for existing entries by media_key.
    """
    updated = 0
    try:
        io_items = flush_io_buffer()
    except Exception:
        io_items = []
    try:
        dom_raw = driver.execute_script(JS_COLLECT_SNIPPET, viewport_pad) or []
    except Exception:
        dom_raw = []

    dom_items: List[Dict[str, str]] = []
    for d in dom_raw:
        try:
            dom_items.append({
                "url": canon_media_url(d.get("url") or ""),
                "uploader_name": d.get("uploader_name") or "",
                "upload_time": d.get("upload_time") or "",
                "tweet_id": d.get("tweet_id") or "",
            })
        except Exception:
            continue

    by_key_meta: Dict[str, Dict[str, str]] = {}
    for it in io_items + dom_items:
        mk = normalize_media_key(it.get("url", "") or "")
        if not mk:
            continue
        prev = by_key_meta.get(mk)
        if (prev is None) or (_meta_score(it) > _meta_score(prev)):
            by_key_meta[mk] = {
                "uploader_name": it.get("uploader_name", "") or "",
                "upload_time": it.get("upload_time", "") or "",
                "tweet_id": it.get("tweet_id", "") or "",
            }

    for e in entries:
        mk = normalize_media_key(e.get("url", "") or "")
        if not mk:
            continue
        m = by_key_meta.get(mk)
        if not m:
            continue
        changed = False
        if not (e.get("uploader_name") or "") and (m.get("uploader_name") or ""):
            e["uploader_name"] = m["uploader_name"]
            changed = True
        if not (e.get("upload_time") or "") and (m.get("upload_time") or ""):
            e["upload_time"] = m["upload_time"]
            changed = True
        if not (e.get("tweet_id") or "") and (m.get("tweet_id") or ""):
            e["tweet_id"] = m["tweet_id"]
            changed = True
        if changed:
            updated += 1
    return updated

def merge_into_meta_map(meta_map: Dict[str, Dict[str, str]], items: List[Dict[str, str]]) -> int:
     
    # items(url,uploader_name,upload_time)를 meta_map(key->{meta})에 병합.
    # 새 키가 추가된 개수를 반환. 기존 키에 비어있는 필드는 새 값으로 채움.
     
    new_keys = 0
    for it in items:
        mk = normalize_media_key(it.get("url", ""))
        if not mk:
            continue
        prev = meta_map.get(mk)
        if not prev:
            meta_map[mk] = {
                "uploader_name": it.get("uploader_name", ""),
                "upload_time": it.get("upload_time", ""),
                "tweet_id": it.get("tweet_id", ""),
            }
            new_keys += 1
        else:
            # 빈 필드는 채움
            if not prev.get("uploader_name") and it.get("uploader_name"):
                prev["uploader_name"] = it.get("uploader_name", "")
            if not prev.get("upload_time") and it.get("upload_time"):
                prev["upload_time"] = it.get("upload_time", "")
            if not prev.get("tweet_id") and it.get("tweet_id"):
                prev["tweet_id"] = it.get("tweet_id", "")
    return new_keys

# -----------------------------------------------------------------------------
# Descent: 완전 바닥 탐지 (+ 간헐 CDP/IO 수집/로그)
# -----------------------------------------------------------------------------
def _get_scroll_y() -> int:
    try:
        return int(driver.execute_script(
            "return Math.max(window.pageYOffset||0, document.documentElement.scrollTop||0, document.body.scrollTop||0);"
        ) or 0)
    except Exception:
        return 0

def _get_scroll_h() -> int:
    try:
        return int(driver.execute_script("return document.body.scrollHeight") or 0)
    except Exception:
        return 0

# === MOD: 뷰포트 높이(vh) 조회 유틸과, 방향에 따라 한 번 흔드는(jiggle) 유틸 추가
def _get_vh() -> int:
    try:
        return int(driver.execute_script("return window.innerHeight || 900") or 900)
    except Exception:
        return 900

def _jiggle_once(delta_px: int, wait_s: float = 0.08):
    try:
        cur_y = _get_scroll_y()
        max_h = _get_scroll_h()
        # top 근처면 down→up, 그 외(대개 bottom)면 up→down
        if cur_y <= 32:
            driver.execute_script("window.scrollBy(0, arguments[0]);",  int(delta_px))
            time.sleep(wait_s)
            driver.execute_script("window.scrollBy(0, arguments[0]);", -int(delta_px))
        else:
            driver.execute_script("window.scrollBy(0, arguments[0]);", -int(delta_px))
            time.sleep(wait_s)
            driver.execute_script("window.scrollBy(0, arguments[0]);",  int(delta_px))
        time.sleep(max(0.04, wait_s * 0.5))
    except Exception:
        pass

def _descent_rescue_probe(
    cdp_seen_keys: set[str],
    cdp_url_by_key: Dict[str, str],
    cdp_meta_by_key: Dict[str, Dict[str, str]] | None = None,
) -> Tuple[bool, int, int, int]:
    y0 = _get_scroll_y()
    h0 = _get_scroll_h()
    k0 = len(cdp_seen_keys)
    vh = _get_vh()
    try:
        driver.execute_script("window.scrollBy(0, arguments[0]);", -int(max(600, vh * 0.9)))
        time.sleep(max(0.18, DOWN_DELAY_S * 5))
        _ = update_cdp_seen_from_logs_with_keys(cdp_seen_keys, cdp_url_by_key, cdp_meta_by_key=cdp_meta_by_key)
        _ = backfill_tweet_id_from_graphql_bodies(cdp_meta_by_key or {}, max_bodies=250, quiet=True)
        driver.execute_script("window.scrollBy(0, arguments[0]);", int(max(1400, vh * 1.8)))
        time.sleep(max(0.35, DOWN_DELAY_S * 8))
        _ = update_cdp_seen_from_logs_with_keys(cdp_seen_keys, cdp_url_by_key, cdp_meta_by_key=cdp_meta_by_key)
        _ = backfill_tweet_id_from_graphql_bodies(cdp_meta_by_key or {}, max_bodies=250, quiet=True)
        driver.execute_script("window.scrollBy(0, arguments[0]);", int(max(1400, vh * 1.8)))
        time.sleep(max(0.35, DOWN_DELAY_S * 8))
        _ = update_cdp_seen_from_logs_with_keys(cdp_seen_keys, cdp_url_by_key, cdp_meta_by_key=cdp_meta_by_key)
        _ = backfill_tweet_id_from_graphql_bodies(cdp_meta_by_key or {}, max_bodies=250, quiet=True)
    except Exception:
        pass
    y1 = _get_scroll_y()
    h1 = _get_scroll_h()
    k1 = len(cdp_seen_keys)
    progressed = (h1 > h0 + 8) or (y1 > y0 + 8) or (k1 > k0)
    return progressed, y1, h1, k1 - k0

def _dynamic_upward_scan_limit(start_y: int, step: int, rounds: int = 0, margin_steps: int = 80) -> int:
    """
    Safety cap for bottom-to-top repair scans.
    It is based on the known descent bottom instead of a small fixed round count,
    so scans can actually reach top on long virtual timelines.
    """
    step_eff = max(1, int(step or 1))
    known_y = max(int(start_y or 0), int(LAST_FULL_DESCENT_BOTTOM_Y or 0))
    if LAST_FULL_DESCENT_SCROLL_H > 0:
        known_y = max(known_y, int(LAST_FULL_DESCENT_SCROLL_H - _get_vh()))
    expected = int(known_y / step_eff) + int(margin_steps)
    round_floor = max(0, int(rounds or 0) * 3)
    return max(120, round_floor, expected)

def bootstrap_observers():
    # IO/MutationObserver를 페이지에 주입(하강/SAFE 모두에서 사용). 
    try:
        ok = driver.execute_script(JS_OBSERVER_BOOTSTRAP)
        print(f"message: observer bootstrap: {'ok' if ok else 'failed'}")
    except Exception as e:
        print(f"message: observer bootstrap error: {e}")

# === MOD: 하강 시작 전(top jiggle) 1회 수행
def pre_descent_jiggle(cdp_seen_keys: set[str], cdp_url_by_key: Dict[str, str]) -> None:
    vh = _get_vh()
    delta = max(80, int(0.45 * vh))
    _jiggle_once(delta, 0.08)
    added = update_cdp_seen_from_logs(cdp_seen_keys, cdp_url_by_key)
    print(f"message: top-jiggle done (vh={vh}, delta={delta}, cdpNew={added})")

def full_descent(cdp_seen_keys: set[str], cdp_url_by_key: Dict[str, str],
                 desc_meta_by_key: Dict[str, Dict[str, str]],
                 stop_when_seen_keys: set[str] | None = None,
                 cdp_meta_by_key: Dict[str, Dict[str, str]] | None = None) -> int:
    global LAST_FULL_DESCENT_BOTTOM_Y, LAST_FULL_DESCENT_SCROLL_H
     
    # 끝까지 하강. Burst마다 진행 로그 출력, DESCENT_CDP_LOG_INTERVAL마다 CDP/IO drain.
    # 반환: 하강 구간에서 관측한 cdpKeys 피크값(peak)
     
    time.sleep(2)
    print("message: full-descent mode start...")

    # === MOD: VH 기반 하강 스텝 자동 클램프(줌-가드)
    vh = _get_vh()
    down_step_px_eff = max(200, min(DOWN_STEP_PX, int(0.70 * vh)))  # 70% vh 상한
    zoom_guard = "on" if down_step_px_eff != DOWN_STEP_PX else "off"
    ratio = (down_step_px_eff / max(1, vh))
    print(f"debug: viewportInnerHeight={vh}, downStepPx={DOWN_STEP_PX}, stepEff={down_step_px_eff}, ratioUsed={ratio:.2f}, zoomGuard={zoom_guard}, burst={DOWN_SCROLL_BURST}")

    prev_h = _get_scroll_h()
    stall_cycles = 0
    burst_idx = 0
    last_y = _get_scroll_y()
    y_stall_seq = 0
    stop_reason = "unknown"
    cdp_peak = len(cdp_seen_keys)
    seen_hit_streak = 0
    meta_ready_base = _count_ready_meta_for_keys(cdp_seen_keys, desc_meta_by_key, cdp_meta_by_key)
    rescue_attempts = 0
    max_rescue_attempts = 36 if FULL_DESCENT_MIN_CDP_KEYS_BEFORE_STALL > 0 else 4
    gql_updated_total = 0
    gql_updated_recent = 0

    # verbose per-burst line logs are noisy for long runs; keep one-line tqdm status by default.
    verbose_descent_debug = False
    with tqdm(total=0, desc="Descent", unit="burst", ncols=140, dynamic_ncols=False, leave=True, ascii=True) as pbar:
        while True:
            burst_idx += 1

            for _ in range(DOWN_SCROLL_BURST):
                driver.execute_script("window.scrollBy(0, arguments[0]);", down_step_px_eff)
                time.sleep(DOWN_DELAY_S)

            grew = False
            grew_px = 0
            for _ in range(DOWN_BUFFER_CHECKS):
                time.sleep(DOWN_BUFFER_SLEEP_S)
                cur_h = _get_scroll_h()
                if cur_h > prev_h:
                    grew = True
                    grew_px = cur_h - prev_h
                    prev_h = cur_h
                    break

            cur_y = _get_scroll_y()
            delta_y = cur_y - last_y
            if abs(delta_y) <= YOFFSET_EPS:
                y_stall_seq += 1
            else:
                y_stall_seq = 0
                last_y = cur_y

            # 한번 더 찔러보기
            if not grew:
                driver.execute_script("window.scrollBy(0, arguments[0]);", down_step_px_eff)
                time.sleep(DOWN_DELAY_S)
                cur_h2 = _get_scroll_h()
                if cur_h2 > prev_h:
                    grew = True
                    grew_px = cur_h2 - prev_h
                    prev_h = cur_h2

            stall_cycles = 0 if grew else (stall_cycles + 1)

            # 주기적 CDP/IO drain
            ioNew = 0
            new_added = 0
            if (burst_idx % DESCENT_CDP_LOG_INTERVAL) == 0:
                new_added, new_keys = update_cdp_seen_from_logs_with_keys(
                    cdp_seen_keys, cdp_url_by_key, cdp_meta_by_key=cdp_meta_by_key
                )
                cdp_peak = max(cdp_peak, len(cdp_seen_keys))
                io_items = flush_io_buffer()
                ioNew = merge_into_meta_map(desc_meta_by_key, io_items)
                if cdp_meta_by_key is not None and (burst_idx % 5) == 0:
                    gql_updated_recent = backfill_tweet_id_from_graphql_bodies(cdp_meta_by_key, max_bodies=400, quiet=True)
                    gql_updated_total += gql_updated_recent
                if verbose_descent_debug:
                    print(
                        f"debug: downBurst={burst_idx}, perBurstScrolls={DOWN_SCROLL_BURST}, "
                        f"yOffset={cur_y}, deltaY={delta_y}, scrollHeight={prev_h}, grewPx={grew_px}, grew={int(grew)}, "
                        f"heightStallSeq={stall_cycles}/{DOWN_STALL_TOLERANCE}, yStallSeq={y_stall_seq}/{YOFFSET_STALL_BURSTS}, "
                        f"cdpKeys={len(cdp_seen_keys)}, cdpNew={new_added}, ioKeys={len(desc_meta_by_key)}, ioNew={ioNew}"
                    )
                if stop_when_seen_keys:
                    hit_keys = [k for k in new_keys if k in stop_when_seen_keys]
                    fresh_keys = [k for k in new_keys if k not in stop_when_seen_keys]
                    if hit_keys and fresh_keys:
                        if seen_hit_streak > 0:
                            print(
                                f"message: periodic stop-streak reset "
                                f"(hit+new in same burst, prevStreak={seen_hit_streak}, "
                                f"hitSample={hit_keys[0]}, newSample={fresh_keys[0]})"
                            )
                        seen_hit_streak = 0
                    elif hit_keys:
                        seen_hit_streak += 1
                        print(
                            f"message: periodic stop-streak {seen_hit_streak}/{PERIODIC_STOP_HIT_STREAK} "
                            f"(hitOnly, hitSample={hit_keys[0]})"
                        )
                        if seen_hit_streak >= PERIODIC_STOP_HIT_STREAK:
                            stop_reason = f"seen-existing-key-streak x{seen_hit_streak} (sample={hit_keys[0]})"
                            print(f"message: stop trigger reached by periodic hit-streak. media_key={hit_keys[0]}")
                            break
                    else:
                        if seen_hit_streak > 0:
                            print(
                                f"message: periodic stop-streak reset "
                                f"(no hit in this burst, prevStreak={seen_hit_streak})"
                            )
                        seen_hit_streak = 0
            elif verbose_descent_debug:
                print(
                    f"debug: downBurst={burst_idx}, perBurstScrolls={DOWN_SCROLL_BURST}, "
                    f"yOffset={cur_y}, deltaY={delta_y}, scrollHeight={prev_h}, grewPx={grew_px}, grew={int(grew)}, "
                    f"heightStallSeq={stall_cycles}/{DOWN_STALL_TOLERANCE}, yStallSeq={y_stall_seq}/{YOFFSET_STALL_BURSTS}"
                )

            pbar.update(1)
            meta_ready_now = _count_ready_meta_for_keys(cdp_seen_keys, desc_meta_by_key, cdp_meta_by_key)
            meta_recovered = max(0, meta_ready_now - meta_ready_base)
            meta_missing = max(0, len(cdp_seen_keys) - meta_ready_now)
            pbar.set_postfix({
                "y": cur_y,
                "h": prev_h,
                "cdp": len(cdp_seen_keys),
                "io": len(desc_meta_by_key),
                "metaMiss": meta_missing,
                "metaRec": meta_recovered,
                "gql": gql_updated_total,
                "stall": f"{stall_cycles}/{DOWN_STALL_TOLERANCE}",
                "hit": seen_hit_streak if stop_when_seen_keys else "-"
            }, refresh=False)

            if stall_cycles >= DOWN_STALL_TOLERANCE:
                if rescue_attempts < max_rescue_attempts and (
                    FULL_DESCENT_MIN_CDP_KEYS_BEFORE_STALL <= 0
                    or len(cdp_seen_keys) < FULL_DESCENT_MIN_CDP_KEYS_BEFORE_STALL
                ):
                    rescue_attempts += 1
                    progressed, ry, rh, rk = _descent_rescue_probe(
                        cdp_seen_keys,
                        cdp_url_by_key,
                        cdp_meta_by_key=cdp_meta_by_key,
                    )
                    print(
                        f"message: descent rescue {rescue_attempts}/{max_rescue_attempts} "
                        f"progress={int(progressed)}, y={ry}, h={rh}, cdpNew={rk}, "
                        f"cdp={len(cdp_seen_keys)}/{FULL_DESCENT_MIN_CDP_KEYS_BEFORE_STALL or '-'}"
                    )
                    if progressed:
                        prev_h = max(prev_h, rh)
                        last_y = ry
                        stall_cycles = 0
                        y_stall_seq = 0
                        cdp_peak = max(cdp_peak, len(cdp_seen_keys))
                        continue
                stop_reason = f"height-stall x{stall_cycles}"
                break
            if y_stall_seq >= YOFFSET_STALL_BURSTS:
                if rescue_attempts < max_rescue_attempts and (
                    FULL_DESCENT_MIN_CDP_KEYS_BEFORE_STALL <= 0
                    or len(cdp_seen_keys) < FULL_DESCENT_MIN_CDP_KEYS_BEFORE_STALL
                ):
                    rescue_attempts += 1
                    progressed, ry, rh, rk = _descent_rescue_probe(
                        cdp_seen_keys,
                        cdp_url_by_key,
                        cdp_meta_by_key=cdp_meta_by_key,
                    )
                    print(
                        f"message: descent rescue {rescue_attempts}/{max_rescue_attempts} "
                        f"progress={int(progressed)}, y={ry}, h={rh}, cdpNew={rk}, "
                        f"cdp={len(cdp_seen_keys)}/{FULL_DESCENT_MIN_CDP_KEYS_BEFORE_STALL or '-'}"
                    )
                    if progressed:
                        prev_h = max(prev_h, rh)
                        last_y = ry
                        stall_cycles = 0
                        y_stall_seq = 0
                        cdp_peak = max(cdp_peak, len(cdp_seen_keys))
                        continue
                stop_reason = f"yoffset-stall x{y_stall_seq}"
                break

    # 최종 drain(마지막 남은 이벤트/버퍼 수거)
    _ = update_cdp_seen_from_logs(cdp_seen_keys, cdp_url_by_key)
    io_items = flush_io_buffer()
    _ = merge_into_meta_map(desc_meta_by_key, io_items)

    LAST_FULL_DESCENT_BOTTOM_Y = max(LAST_FULL_DESCENT_BOTTOM_Y, _get_scroll_y())
    LAST_FULL_DESCENT_SCROLL_H = max(LAST_FULL_DESCENT_SCROLL_H, _get_scroll_h())
    print(f"message: full-descent finished. stopReason={stop_reason}, yOffset={_get_scroll_y()}, scrollHeight={_get_scroll_h()}, cdpKeys={len(cdp_seen_keys)}, ioKeys={len(desc_meta_by_key)}")
    return cdp_peak

# -----------------------------------------------------------------------------
# SAFE: 업스크롤 수집(타겟 매칭 및 메타 확보)
# -----------------------------------------------------------------------------
def _poll_once_and_confirm(
    target_keys: set[str],
    confirmed_keys: set[str],
    confirmed: Dict[str, Dict[str, str]],
    cdp_seen_keys_ref: set[str] | None = None,
    cdp_url_by_key_ref: Dict[str, str] | None = None,
    cdp_meta_by_key_ref: Dict[str, Dict[str, str]] | None = None,
) -> Tuple[int,int,int,int]:
    # IO 버퍼+DOM 스냅샷을 병합하여 Target에 해당하는 키를 확정. 
    jsCalls = 0

    # CDP 로그를 우선 drain해서 비디오 URL도 동일한 confirm 경로로 태운다.
    local_seen = cdp_seen_keys_ref if cdp_seen_keys_ref is not None else set()
    local_url_by_key = cdp_url_by_key_ref if cdp_url_by_key_ref is not None else {}
    local_meta_by_key = cdp_meta_by_key_ref if cdp_meta_by_key_ref is not None else {}

    cdp_new, _ = update_cdp_seen_from_logs_with_keys(
        local_seen,
        local_url_by_key,
        cdp_meta_by_key=local_meta_by_key,
    )

    io_items = flush_io_buffer()
    jsCalls += 1

    dom_raw = driver.execute_script(JS_COLLECT_SNIPPET, VIEWPORT_PAD) or []
    jsCalls += 1
    dom_items = []
    for d in dom_raw:
        try:
            url = canon_media_url(d.get("url") or "")
            dom_items.append({
                "url": url,
                "uploader_name": d.get("uploader_name") or "",
                "upload_time": d.get("upload_time") or "",
                "tweet_id": d.get("tweet_id") or "",
            })
        except Exception:
            continue

    merged: Dict[str, Dict[str, str]] = {}
    for src in (io_items, dom_items):
        for it in src:
            u = it["url"]
            if u not in merged:
                merged[u] = it
    # CDP에서 확보한 타겟 URL/메타도 merge에 포함(특히 video.twimg.com 계열).
    for mk in target_keys:
        cu = local_url_by_key.get(mk, "")
        if not cu:
            continue
        if cu not in merged:
            cmeta = local_meta_by_key.get(mk, {}) or {}
            merged[cu] = {
                "url": cu,
                "uploader_name": cmeta.get("uploader_name", ""),
                "upload_time": cmeta.get("upload_time", ""),
                "tweet_id": cmeta.get("tweet_id", ""),
            }
    visible_tids = collect_visible_tweet_ids()
    single_visible_tid = visible_tids[0] if len(visible_tids) == 1 else ""

    new_cnt = 0
    dup_cnt = 0
    target_key_by_id: Dict[str, str] = {}
    for tk in target_keys:
        m = re.match(r"^(ext_tw_video|amplify_video)_(\d+)$", tk)
        if m:
            target_key_by_id[m.group(2)] = tk

    for u, it in merged.items():
        mk = normalize_media_key(u)
        if not mk:
            continue
        matched_key = mk
        if matched_key not in target_keys:
            m = re.match(r"^(ext_tw_video|amplify_video)_(\d+)$", matched_key)
            if m:
                matched_key = target_key_by_id.get(m.group(2), matched_key)
        if matched_key in target_keys:
            if matched_key not in confirmed_keys:
                confirmed_keys.add(matched_key)
                confirmed[matched_key] = {
                    "url": u,
                    "uploader_name": it.get("uploader_name", ""),
                    "upload_time": it.get("upload_time", ""),
                    "tweet_id": (it.get("tweet_id", "") or single_visible_tid),
                }
                new_cnt += 1
            else:
                # 이미 확정된 키라도, 이번에 더 좋은 메타가 보이면 갱신한다.
                prev = confirmed.get(matched_key, {})
                prev_meta = {
                    "uploader_name": prev.get("uploader_name", "") or "",
                    "upload_time": prev.get("upload_time", "") or "",
                    "tweet_id": prev.get("tweet_id", "") or "",
                }
                cur_meta = {
                    "uploader_name": it.get("uploader_name", "") or "",
                    "upload_time": it.get("upload_time", "") or "",
                    "tweet_id": (it.get("tweet_id", "") or single_visible_tid or ""),
                }
                best = pick_better_meta(prev_meta, cur_meta)
                changed = False
                if (best.get("uploader_name", "") or "") != (prev.get("uploader_name", "") or ""):
                    prev["uploader_name"] = best.get("uploader_name", "") or ""
                    changed = True
                if (best.get("upload_time", "") or "") != (prev.get("upload_time", "") or ""):
                    prev["upload_time"] = best.get("upload_time", "") or ""
                    changed = True
                if (best.get("tweet_id", "") or "") != (prev.get("tweet_id", "") or ""):
                    prev["tweet_id"] = best.get("tweet_id", "") or ""
                    changed = True
                # URL도 더 나은 품질(특히 video)로 업데이트
                if _video_quality_score(u) > _video_quality_score(prev.get("url", "") or ""):
                    prev["url"] = u
                    changed = True
                if changed:
                    confirmed[matched_key] = prev
                if single_visible_tid and not (confirmed.get(matched_key, {}).get("tweet_id") or ""):
                    confirmed[matched_key]["tweet_id"] = single_visible_tid
                dup_cnt += 1
    verbose_safe_poll = False
    if verbose_safe_poll and new_cnt == 0 and cdp_new > 0 and target_keys:
        sample_targets = ", ".join(list(target_keys)[:3])
        print(
            f"debug: [SAFE-POLL] cdpNew={cdp_new}, merged={len(merged)}, "
            f"sampleTargets={sample_targets}"
        )
    return new_cnt, dup_cnt, len(merged), jsCalls

def _poll_until_settled(
    target_keys: set[str],
    confirmed_keys: set[str],
    confirmed: Dict[str, Dict[str, str]],
    cdp_seen_keys_ref: set[str] | None = None,
    cdp_url_by_key_ref: Dict[str, str] | None = None,
    cdp_meta_by_key_ref: Dict[str, Dict[str, str]] | None = None,
) -> Tuple[int,int,int,int]:
    # 단계 내에서 수집이 안정될 때까지 짧게 폴링. 
    STEP_MIN_SETTLE_S = 0.30
    idle_seq = 0
    t0 = time.time()
    new_total = 0
    dup_total = 0
    last_batch_size = 0
    js_calls = 0

    while True:
        a, d, b, c = _poll_once_and_confirm(
            target_keys,
            confirmed_keys,
            confirmed,
            cdp_seen_keys_ref=cdp_seen_keys_ref,
            cdp_url_by_key_ref=cdp_url_by_key_ref,
            cdp_meta_by_key_ref=cdp_meta_by_key_ref,
        )
        new_total += a
        dup_total += d
        last_batch_size = b
        js_calls += c

        if a == 0:
            idle_seq += 1
        else:
            idle_seq = 0

        if (idle_seq >= 2) and ((time.time() - t0) >= STEP_MIN_SETTLE_S):
            break
        time.sleep(max(UP_DELAY_S * 0.5, 0.02))

    return new_total, dup_total, last_batch_size, js_calls

# === MOD: SAFE 상승 전(bottom 근처) jiggle 1회 수행
def pre_upward_jiggle(cdp_seen_keys: set[str], cdp_url_by_key: Dict[str, str]) -> None:
    vh = _get_vh()
    delta = max(80, int(0.45 * vh))
    _jiggle_once(delta, 0.08)
    added = update_cdp_seen_from_logs(cdp_seen_keys, cdp_url_by_key)
    print(f"message: pre-upward jiggle done (vh={vh}, delta={delta}, cdpNew={added})")

def force_scroll_near_bottom(max_iters: int = 60) -> Tuple[int, int]:
    """
    Force page near bottom using repeated downward scrolls.
    Returns (start_y, end_y).
    """
    start_y = _get_scroll_y()
    vh = _get_vh()
    step = max(400, int(0.9 * vh))
    stall = 0
    last_y = start_y
    for _ in range(max_iters):
        driver.execute_script("window.scrollBy(0, arguments[0]);", step)
        time.sleep(max(0.03, DOWN_DELAY_S))
        y = _get_scroll_y()
        if y <= last_y + 1:
            stall += 1
        else:
            stall = 0
            last_y = y
        # reached bottom-ish
        h = _get_scroll_h()
        if y + vh >= h - 8:
            break
        if stall >= 5:
            break
    return start_y, _get_scroll_y()

def force_scroll_true_bottom(max_steps: int = 6000) -> Tuple[int, int]:
    """
    Force scroll to real bottom for virtualized timelines.
    Uses height-growth + y-stall checks (similar spirit to full_descent).
    Returns (start_y, end_y).
    """
    start_y = _get_scroll_y()
    vh = _get_vh()
    step = max(240, min(DOWN_STEP_PX, int(0.70 * vh)))
    prev_h = _get_scroll_h()
    last_y = start_y
    stall_h = 0
    stall_y = 0
    from_top = start_y <= 2
    known_bottom_floor = int(LAST_FULL_DESCENT_BOTTOM_Y * 0.92) if LAST_FULL_DESCENT_BOTTOM_Y > 0 else 0
    min_seek_px_from_top = max(50000, int(vh * 20), known_bottom_floor)
    min_steps_from_top = 60 if not known_bottom_floor else max(60, int(known_bottom_floor / max(1, step) * 0.60))
    i = 0
    for _ in range(max_steps):
        i += 1
        driver.execute_script("window.scrollBy(0, arguments[0]);", step)
        time.sleep(max(0.03, DOWN_DELAY_S))
        y = _get_scroll_y()
        h = _get_scroll_h()

        if h > prev_h:
            prev_h = h
            stall_h = 0
        else:
            stall_h += 1

        if y <= last_y + 1:
            stall_y += 1
        else:
            stall_y = 0
            last_y = y

        ready_to_stop_from_top = (not from_top) or (y >= min_seek_px_from_top and i >= min_steps_from_top)
        if ready_to_stop_from_top and y + vh >= prev_h - 8 and stall_h >= 2:
            break
        if ready_to_stop_from_top and (stall_h >= DOWN_STALL_TOLERANCE or stall_y >= YOFFSET_STALL_BURSTS):
            break
        if from_top and not ready_to_stop_from_top and (stall_h >= DOWN_STALL_TOLERANCE or stall_y >= YOFFSET_STALL_BURSTS):
            try:
                driver.execute_script("window.scrollBy(0, arguments[0]);", -int(max(120, vh * 0.25)))
                time.sleep(max(0.08, DOWN_DELAY_S * 2))
                driver.execute_script("window.scrollBy(0, arguments[0]);", int(max(step, vh * 0.9)))
                time.sleep(max(0.12, DOWN_DELAY_S * 3))
            except Exception:
                pass
            stall_h = 0
            stall_y = 0
            last_y = _get_scroll_y()
            prev_h = max(prev_h, _get_scroll_h())
    return start_y, _get_scroll_y()

def ensure_top_start(label: str = "") -> Tuple[int, int]:
    y0 = _get_scroll_y()
    if y0 > 2:
        try:
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(max(0.03, UP_DELAY_S))
        except Exception:
            pass
    y1 = _get_scroll_y()
    if label:
        print(f"message: {label} ensure-top startY {y0} -> {y1}")
    return y0, y1

def ensure_bottom_start(label: str = "") -> Tuple[int, int]:
    y0 = _get_scroll_y()
    if y0 <= 2:
        s0, s1 = force_scroll_true_bottom()
        if label:
            print(f"message: {label} ensure-bottom via true-bottom {s0} -> {s1}")
        return y0, s1
    y1 = _get_scroll_y()
    if label:
        print(f"message: {label} ensure-bottom reuse current y={y1}")
    return y0, y1

def ensure_repair_bottom_start(label: str = "") -> Tuple[int, int]:
    # SAFE-Up/RepairWarmup-UpFull 직후에는 y=0 근처일 수 있다.
    # 이때 author/date backfill은 bottom→top 방향이어야 하므로, full_descent에서 기록한 최하단 위치로 먼저 복귀한다.
    y0 = _get_scroll_y()
    if y0 <= 50 and LAST_FULL_DESCENT_BOTTOM_Y > 0:
        try:
            driver.execute_script("window.scrollTo(0, arguments[0]);", int(LAST_FULL_DESCENT_BOTTOM_Y))
            time.sleep(max(0.20, DOWN_DELAY_S * 6))
        except Exception:
            pass
        y_mid = _get_scroll_y()
        if y_mid >= int(LAST_FULL_DESCENT_BOTTOM_Y * 0.85):
            if label:
                print(f"message: {label} ensure-bottom via remembered-bottom {y0} -> {y_mid}")
            return y0, y_mid
    return ensure_bottom_start(label)

def safe_upward_collect(
    target_keys: set[str],
    target_url_by_key: Dict[str, str],
    cdp_seen_keys_ref: set[str] | None = None,
    cdp_url_by_key_ref: Dict[str, str] | None = None,
    cdp_meta_by_key_ref: Dict[str, Dict[str, str]] | None = None,
) -> Tuple[Dict[str, Dict[str, str]], set[str], List[str]]:
    # SAFE 업스크롤 수집. 반환: (확정맵, 확정키집합, Missing 키 List) 
    confirmed: Dict[str, Dict[str, str]] = {}
    confirmed_keys: set[str] = set()

    # === MOD: 상승 시작 직전에 jiggle 1회로 바닥 부근 로딩 유도
    pre_upward_jiggle(cdp_seen_keys_ref or set(), cdp_url_by_key_ref or {})

    try:
        vh = driver.execute_script("return window.innerHeight") or 900
    except Exception:
        vh = 900
    MARGIN_PX = 100
    coverage = int(vh + 2 * VIEWPORT_PAD)
    max_safe_step = max(200, int(coverage - MARGIN_PX))
    stepPxEff = min(UP_STEP_PX, max_safe_step)
    move_px0 = max(50, stepPxEff - 300)
    max_move_by_ratio = int(coverage * (1.0 - SAFE_OVERLAP_RATIO))
    move_px = min(move_px0, max_move_by_ratio)

    TargetTotalSeen = len(target_keys)
    meta_ready_base = _count_ready_meta_for_keys(target_keys, confirmed, cdp_meta_by_key_ref)

    step = 0
    top_stall_seq = 0
    crawl_t0 = time.time()
    with tqdm(total=0, desc="SAFE-Up", unit="step", dynamic_ncols=True, leave=True) as pbar:
        while True:
            step += 1
            curr_y = _get_scroll_y()

            if curr_y <= 2:
                a, d, b, js_calls = _poll_once_and_confirm(
                    target_keys,
                    confirmed_keys,
                    confirmed,
                    cdp_seen_keys_ref=cdp_seen_keys_ref,
                    cdp_url_by_key_ref=cdp_url_by_key_ref,
                    cdp_meta_by_key_ref=cdp_meta_by_key_ref,
                )
                pbar.update(1)
                meta_ready_now = _count_ready_meta_for_keys(target_keys, confirmed, cdp_meta_by_key_ref)
                meta_recovered = max(0, meta_ready_now - meta_ready_base)
                meta_missing = max(0, TargetTotalSeen - meta_ready_now)
                pbar.set_postfix({
                    "new": a,
                    "dup": d,
                    "batch": b,
                    "js": js_calls,
                    "y": _get_scroll_y(),
                    "seen": f"{len(confirmed_keys)}/{TargetTotalSeen}",
                    "metaMiss": meta_missing,
                    "metaRec": meta_recovered,
                    "final": 1
                }, refresh=False)
                break

            new_total, dup_total, last_batch_size, total_js_calls = _poll_until_settled(
                target_keys,
                confirmed_keys,
                confirmed,
                cdp_seen_keys_ref=cdp_seen_keys_ref,
                cdp_url_by_key_ref=cdp_url_by_key_ref,
                cdp_meta_by_key_ref=cdp_meta_by_key_ref,
            )
            pbar.update(1)
            meta_ready_now = _count_ready_meta_for_keys(target_keys, confirmed, cdp_meta_by_key_ref)
            meta_recovered = max(0, meta_ready_now - meta_ready_base)
            meta_missing = max(0, TargetTotalSeen - meta_ready_now)
            pbar.set_postfix({
                "new": new_total,
                "dup": dup_total,
                "batch": last_batch_size,
                "js": total_js_calls,
                "y": curr_y,
                "seen": f"{len(confirmed_keys)}/{TargetTotalSeen}",
                "metaMiss": meta_missing,
                "metaRec": meta_recovered,
            }, refresh=False)

            prev_y = _get_scroll_y()
            driver.execute_script("window.scrollBy(0, arguments[0]);", -int(move_px))
            time.sleep(UP_DELAY_S)
            cur_y = _get_scroll_y()

            if cur_y >= prev_y - 1:
                top_stall_seq += 1
            else:
                top_stall_seq = 0

            if cur_y <= 2 and top_stall_seq >= 3:
                a, d, b, js_calls = _poll_once_and_confirm(
                    target_keys,
                    confirmed_keys,
                    confirmed,
                    cdp_seen_keys_ref=cdp_seen_keys_ref,
                    cdp_url_by_key_ref=cdp_url_by_key_ref,
                    cdp_meta_by_key_ref=cdp_meta_by_key_ref,
                )
                pbar.update(1)
                meta_ready_now = _count_ready_meta_for_keys(target_keys, confirmed, cdp_meta_by_key_ref)
                meta_recovered = max(0, meta_ready_now - meta_ready_base)
                meta_missing = max(0, TargetTotalSeen - meta_ready_now)
                pbar.set_postfix({
                    "new": a,
                    "dup": d,
                    "batch": b,
                    "js": js_calls,
                    "y": _get_scroll_y(),
                    "seen": f"{len(confirmed_keys)}/{TargetTotalSeen}",
                    "metaMiss": meta_missing,
                    "metaRec": meta_recovered,
                    "final": 1
                }, refresh=False)
                break

    elapsed = time.time() - crawl_t0
    CurrentTotalSeen = len(confirmed_keys)
    missing_keys = [k for k in target_keys if k not in confirmed_keys]
    print(f"message: SAFE upward collection finished in {elapsed:.2f} seconds")
    print(f"message: TargetTotalSeen={TargetTotalSeen}, CurrentTotalSeen={CurrentTotalSeen}, Missing={TargetTotalSeen - CurrentTotalSeen}")
    if missing_keys:
        print("message: Missing detail follows (key -> url):")
        for k in missing_keys:
            print(f"message: MISSING key={k} url={target_url_by_key.get(k, '')}")

    return confirmed, confirmed_keys, missing_keys

# -----------------------------------------------------------------------------
# Downloader / Post-process
# -----------------------------------------------------------------------------
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    retry = Retry(total=3, connect=3, read=3, backoff_factor=0.5,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=frozenset(["GET", "HEAD"]))
    adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

def download_one(index: int, data: Dict[str, str], out_dir: str) -> Tuple[bool, str | None, str, str | None, str]:
    url = data["url"]
    uploader_name = data.get("uploader_name", "")
    upload_time = data.get("upload_time", "")
    tweet_id = data.get("tweet_id", "")

    session = make_session()
    filename: str | None = None
    try:
        resolved_url = resolve_best_video_url(url, session)
        filename = make_deterministic_filename(resolved_url, uploader_name, upload_time, tweet_id=tweet_id)
        filename = ensure_video_filename_extension(resolved_url, filename)
        file_path = os.path.join(out_dir, filename)

        if SKIP_IF_EXISTS and os.path.exists(file_path):
            return True, os.path.basename(file_path), resolved_url, None, "skip_exists"

        if ".m3u8" in (resolved_url or "").lower():
            ok_hls, hls_err = _download_hls_to_mp4(resolved_url, file_path)
            if ok_hls:
                return True, os.path.basename(file_path), resolved_url, None, "ok_hls"
            return False, os.path.basename(file_path), resolved_url, hls_err or "hls_download_failed", "error_hls"

        resp = session.get(resolved_url, timeout=10)
        resp.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(resp.content)
        return True, os.path.basename(file_path), resolved_url, None, "ok"
    except Exception as e:
        fallback_name = filename or make_deterministic_filename(url, uploader_name, upload_time, tweet_id=tweet_id)
        return False, fallback_name, url, str(e), "error"
    finally:
        try:
            session.close()
        except Exception:
            pass

def move_duplicate_images(directory_path: str):
    duplicate_folder = os.path.join(directory_path, "duplicates")
    os.makedirs(duplicate_folder, exist_ok=True)
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    hash_dict = {}
    duplicate_count = 0
    for dirpath, _, filenames in os.walk(directory_path):
        if dirpath != directory_path and not dirpath.startswith(directory_path):
            continue
        for filename in filenames:
            if not filename.lower().endswith(image_exts):
                continue
            file_path = os.path.join(dirpath, filename)
            try:
                with open(file_path, "rb") as f:
                    h = hashlib.md5(f.read()).hexdigest()
                if h in hash_dict:
                    shutil.move(file_path, os.path.join(duplicate_folder, filename))
                    duplicate_count += 1
                else:
                    hash_dict[h] = file_path
            except Exception:
                continue
    try:
        deleted_files = len(os.listdir(duplicate_folder))
        shutil.rmtree(duplicate_folder)
    except Exception:
        deleted_files = 0
    return duplicate_count, deleted_files

def read_local_items_ndjson(items_path: str) -> List[Dict[str, str]]:
    read_path = items_path
    if (not os.path.exists(read_path)) and (read_path == BOOKMARK_META_LOCAL_ITEMS_PATH) and os.path.exists(LEGACY_BOOKMARK_META_OLDVER_ITEMS_PATH):
        read_path = LEGACY_BOOKMARK_META_OLDVER_ITEMS_PATH
        print(f"message: local ndjson not found; using legacy path: {read_path}")
    if not os.path.exists(read_path):
        return []
    items: List[Dict[str, str]] = []
    with open(read_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    items.append(obj)
            except Exception:
                continue
    return items

def load_seen_media_keys_from_local(items_path: str) -> set[str]:
    seen: set[str] = set()
    for obj in read_local_items_ndjson(items_path):
        mk = (obj.get("media_key") or "").strip()
        if mk:
            seen.add(mk)
    return seen

def append_local_items_ndjson(items_path: str, entries: List[Dict[str, str]]) -> Tuple[int, int]:
    os.makedirs(os.path.dirname(items_path), exist_ok=True)
    existing_rows = read_local_items_ndjson(items_path)
    by_key: Dict[str, Dict[str, str]] = {}
    for obj in existing_rows:
        mk = (obj.get("media_key") or "").strip()
        if mk:
            by_key[mk] = dict(obj)

    added = 0
    updated = 0
    for it in entries:
        url = canon_media_url(it.get("url", ""))
        mk = normalize_media_key(url) or ""
        if not mk:
            continue
        incoming = {
            "tweet_id": it.get("tweet_id", "") or "",
            "author": it.get("uploader_name", "") or "",
            "created_at": it.get("upload_time", "") or "",
            "created_at_norm": normalize_time(it.get("upload_time", "") or ""),
            "tid_source": it.get("tid_source", "") or "",
            "tid_confidence": it.get("tid_confidence", "") or "",
            "media_key": mk,
            "url": url,
        }
        if incoming["tweet_id"] and not incoming["tid_source"]:
            incoming["tid_source"] = "unknown"
        if incoming["tweet_id"] and not incoming["tid_confidence"]:
            incoming["tid_confidence"] = _tid_confidence(incoming["tid_source"])
        prev = by_key.get(mk)
        if prev is None:
            by_key[mk] = incoming
            added += 1
            continue

        changed = False
        # fill only missing fields; keep existing non-empty data
        for fld in ("tweet_id", "author", "created_at", "created_at_norm", "tid_source", "tid_confidence"):
            if not (prev.get(fld) or "") and (incoming.get(fld) or ""):
                prev[fld] = incoming[fld]
                changed = True
        if (not (prev.get("url") or "")) and incoming["url"]:
            prev["url"] = incoming["url"]
            changed = True
        by_key[mk] = prev
        if changed:
            updated += 1

    with open(items_path, "w", encoding="utf-8") as f:
        for obj in by_key.values():
            out = {
                "tweet_id": obj.get("tweet_id", "") or "",
                "author": obj.get("author", "") or "",
                "created_at": obj.get("created_at", "") or "",
                "created_at_norm": obj.get("created_at_norm", "") or "",
                "tid_source": obj.get("tid_source", "") or "",
                "tid_confidence": obj.get("tid_confidence", "") or "",
                "media_key": obj.get("media_key", "") or "",
                "url": obj.get("url", "") or "",
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"message: local ndjson merge summary: added={added}, updated_missing_meta={updated}")
    return added, len(by_key)

def is_video_url(url: str) -> bool:
    u = (url or "").lower()
    if "video.twimg.com/" not in u:
        return False
    if "/aud/" in u:
        return False
    if ".m3u8" in u:
        return True
    return is_preferred_video_mp4_url(u)

def _is_unknown_author_value(author: str) -> bool:
    a = (author or "").strip().lower()
    return (not a) or ("unknown" in a)

def _row_missing_video_meta(row: Dict[str, str]) -> bool:
    tid = (row.get("tweet_id") or "").strip()
    # video.twimg.com 계열은 author/created_at가 DOM에서 안정적으로 안 잡히는 경우가 많다.
    # 우선 tweet_id 보강을 1차 목표로 본다.
    return not tid.isdigit()

def _rewrite_video_items_ndjson(items_path: str, rows: List[Dict[str, str]]) -> None:
    # If operating on unified items.ndjson, preserve non-video rows and update only video rows.
    unified = os.path.abspath(items_path) == os.path.abspath(BOOKMARK_META_LOCAL_ITEMS_PATH)
    out_rows: List[Dict[str, str]] = []
    if unified and os.path.exists(items_path):
        existing = read_local_items_ndjson(items_path)
        by_key_new: Dict[str, Dict[str, str]] = {}
        for r in rows:
            if not is_video_url(r.get("url", "") or ""):
                continue
            mk = (r.get("media_key") or "").strip() or (normalize_media_key(r.get("url", "") or "") or "")
            if mk:
                by_key_new[mk] = r
        seen_new = set()
        for ex in existing:
            ex_url = ex.get("url", "") or ""
            ex_mk = (ex.get("media_key") or "").strip() or (normalize_media_key(ex_url) or "")
            if is_video_url(ex_url) and ex_mk in by_key_new:
                out_rows.append(by_key_new[ex_mk])
                seen_new.add(ex_mk)
            else:
                out_rows.append(ex)
        for mk, r in by_key_new.items():
            if mk not in seen_new:
                out_rows.append(r)
    else:
        out_rows = list(rows)

    with open(items_path, "w", encoding="utf-8") as f:
        for obj in out_rows:
            out = {
                "tweet_id": obj.get("tweet_id", "") or "",
                "author": obj.get("author", "") or "",
                "created_at": obj.get("created_at", "") or "",
                "created_at_norm": obj.get("created_at_norm", "") or "",
                "tid_source": obj.get("tid_source", "") or "",
                "tid_confidence": obj.get("tid_confidence", "") or "",
                "media_key": obj.get("media_key", "") or "",
                "url": obj.get("url", "") or "",
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

def reset_video_tweet_ids(items_path: str) -> int:
    rows = read_local_items_ndjson(items_path)
    changed = 0
    for r in rows:
        if (r.get("tweet_id", "") or "").strip():
            r["tweet_id"] = ""
            r["tid_source"] = ""
            r["tid_confidence"] = ""
            changed += 1
    _rewrite_video_items_ndjson(items_path, rows)
    return changed

def _video_only_stats(rows: List[Dict[str, str]]) -> Tuple[int, int, int]:
    vids = [r for r in rows if is_video_url(r.get("url", "") or "")]
    unresolved = sum(1 for r in vids if _row_missing_video_meta(r))
    filled_tid = sum(1 for r in vids if (r.get("tweet_id", "") or "").isdigit())
    return unresolved, filled_tid, len(vids)

def repair_video_items_meta(
    items_path: str,
    preloaded_timeline: bool = False,
    preloaded_cdp_seen_keys: set[str] | None = None,
    preloaded_cdp_url_by_key: Dict[str, str] | None = None,
    preloaded_cdp_meta_by_key: Dict[str, Dict[str, str]] | None = None,
) -> None:
    # VIDEO_META_REPAIR:
    # - 단독 6번 모드에서는 full descent를 다시 수행해 GraphQL/CDP 메타를 최대한 재수집한다.
    # - 다운로드 직전 repair(preloaded_timeline=True)는 이미 수집 중 누적한 CDP/GraphQL 캐시를 재사용한다.
    # - tweet_id는 GraphQL/CDP alias 매칭으로 먼저 채우고, author/date는 전체 업스크롤 backfill로 보강한다.
    global FULL_DESCENT_MIN_CDP_KEYS_BEFORE_STALL
    cdp_seen_keys_local: set[str] = set(preloaded_cdp_seen_keys or set())
    cdp_url_by_key_local: Dict[str, str] = dict(preloaded_cdp_url_by_key or {})
    cdp_meta_by_key_local: Dict[str, Dict[str, str]] = dict(preloaded_cdp_meta_by_key or {})

    rows = read_local_items_ndjson(items_path)
    if not rows:
        print(f"message: VIDEO_META_REPAIR skipped. no rows in {items_path}")
        return

    by_key: Dict[str, Dict[str, str]] = {}
    for r in rows:
        if not is_video_url(r.get("url", "") or ""):
            continue
        mk = (r.get("media_key") or "").strip()
        if not mk:
            mk = normalize_media_key(r.get("url", "") or "") or ""
            r["media_key"] = mk
        if mk:
            by_key[mk] = r

    target_keys = {mk for mk, r in by_key.items() if _row_missing_video_meta(r)}
    author_missing_exists = any(
        (r.get("tweet_id", "") or "").strip().isdigit() and not (r.get("author") or "").strip()
        for r in by_key.values()
    )
    if not target_keys and not author_missing_exists:
        print("message: VIDEO_META_REPAIR nothing to fix. no missing-meta rows.")
        return

    target_url_by_key = {mk: (by_key[mk].get("url", "") or "") for mk in target_keys}
    print(
        f"message: VIDEO_META_REPAIR targets={len(target_keys)} totalRows={len(by_key)} "
        f"authorMissing={author_missing_exists} preloaded={preloaded_timeline}"
    )

    # In pre-download path, timeline has already been loaded by main collection.
    # Skip expensive warmup/retarget scans and run backfill-focused passes.
    if not preloaded_timeline:
        prev_min_cdp_floor = FULL_DESCENT_MIN_CDP_KEYS_BEFORE_STALL
        FULL_DESCENT_MIN_CDP_KEYS_BEFORE_STALL = max(1200, int(len(rows) * 0.55)) if len(rows) >= 1000 else 0
        if FULL_DESCENT_MIN_CDP_KEYS_BEFORE_STALL:
            print(
                f"message: VIDEO_META_REPAIR descent min-cdp floor="
                f"{FULL_DESCENT_MIN_CDP_KEYS_BEFORE_STALL} from localRows={len(rows)}"
            )
        warmup_scroll_for_repair_full(
            cdp_seen_keys_local,
            cdp_url_by_key_local,
            cdp_meta_by_key_local,
            return_to_top=False,
        )
        FULL_DESCENT_MIN_CDP_KEYS_BEFORE_STALL = prev_min_cdp_floor
    else:
        print("message: VIDEO_META_REPAIR preloaded mode: skip warmup/full SAFE retarget scans.")

    confirmed_map: Dict[str, Dict[str, str]] = {}
    confirmed_keys: set[str] = set()
    if target_keys and (not preloaded_timeline):
        print(f"message: VIDEO_META_REPAIR at-bottom start yOffset={_get_scroll_y()}")
        _ = update_cdp_seen_from_logs_with_keys(
            cdp_seen_keys_local,
            cdp_url_by_key_local,
            cdp_meta_by_key=cdp_meta_by_key_local,
        )
        gql_updates = backfill_tweet_id_from_graphql_bodies(cdp_meta_by_key_local)
        req_updates = backfill_tweet_id_from_cdp_video_requests(cdp_meta_by_key_local)
        exact_hits = sum(1 for mk in target_keys if mk in cdp_meta_by_key_local)
        alias_hits = sum(1 for mk in target_keys if _lookup_meta_by_video_key_alias(mk, cdp_meta_by_key_local)[0])
        print(
            f"message: VIDEO_META_REPAIR no-safe mode: skip SAFE-Up, run backfill chain. "
            f"gqlBackfill={gql_updates}, cdpReqBackfill={req_updates}, "
            f"cdpExactHits={exact_hits}, cdpAliasHits={alias_hits}"
        )
    elif target_keys and preloaded_timeline:
        print("message: VIDEO_META_REPAIR preloaded mode: skip SAFE pass, use lightweight GraphQL/CDP backfill path.")
        # Reuse already-loaded timeline logs to fill tweet_id without re-running heavy scroll passes.
        _ = update_cdp_seen_from_logs_with_keys(
            cdp_seen_keys_local,
            cdp_url_by_key_local,
            cdp_meta_by_key=cdp_meta_by_key_local,
        )
        gql_updates = backfill_tweet_id_from_graphql_bodies(cdp_meta_by_key_local)
        req_updates = backfill_tweet_id_from_cdp_video_requests(cdp_meta_by_key_local)
        cdp_applied = 0
        for mk in target_keys:
            prev = by_key.get(mk)
            if not prev:
                continue
            if (prev.get("tweet_id", "") or "").strip():
                continue
            cmeta, matched_key = _lookup_meta_by_video_key_alias(mk, cdp_meta_by_key_local)
            tid = (cmeta.get("tweet_id", "") or "").strip()
            if tid:
                prev["tweet_id"] = tid
                if cmeta.get("uploader_name", "") and not (prev.get("author", "") or "").strip():
                    prev["author"] = cmeta.get("uploader_name", "")
                if cmeta.get("upload_time", "") and not (prev.get("created_at", "") or "").strip():
                    prev["created_at"] = cmeta.get("upload_time", "")
                    prev["created_at_norm"] = normalize_time(prev.get("created_at", "") or "")
                src = _source_for_video_key_alias(mk, matched_key) or "graphql"
                prev["tid_source"] = src
                prev["tid_confidence"] = _tid_confidence(src)
                cdp_applied += 1
        exact_hits = sum(1 for mk in target_keys if mk in cdp_meta_by_key_local)
        alias_hits = sum(1 for mk in target_keys if _lookup_meta_by_video_key_alias(mk, cdp_meta_by_key_local)[0])
        print(
            f"message: VIDEO_META_REPAIR preloaded backfill done. "
            f"gqlBackfill={gql_updates}, cdpReqBackfill={req_updates}, "
            f"cdpExactHits={exact_hits}, cdpAliasHits={alias_hits}, cdpApplied={cdp_applied}"
        )
    else:
        print("message: VIDEO_META_REPAIR tweet_id target is empty. skip SAFE tweet_id pass and run author-only pass.")

    updated = 0
    # Apply whatever metadata we already have from CDP/GraphQL without SAFE-Up dependency.
    for mk, prev in by_key.items():
        cmeta, matched_key = _lookup_meta_by_video_key_alias(mk, cdp_meta_by_key_local)
        new_meta = {
            "uploader_name": cmeta.get("uploader_name", "") or "",
            "upload_time": cmeta.get("upload_time", "") or "",
            "tweet_id": cmeta.get("tweet_id", "") or "",
        }
        old_meta = {
            "uploader_name": prev.get("author", "") or "",
            "upload_time": prev.get("created_at", "") or "",
            "tweet_id": prev.get("tweet_id", "") or "",
        }
        best = pick_better_meta(old_meta, new_meta)
        changed = False
        if (best.get("tweet_id", "") or "") != (prev.get("tweet_id", "") or ""):
            prev["tweet_id"] = best.get("tweet_id", "") or ""
            src = _source_for_video_key_alias(mk, matched_key) if prev["tweet_id"] else ""
            prev["tid_source"] = src
            prev["tid_confidence"] = _tid_confidence(src) if prev["tweet_id"] else ""
            changed = True
        if (best.get("uploader_name", "") or "") != (prev.get("author", "") or ""):
            prev["author"] = best.get("uploader_name", "") or ""
            changed = True
        if (best.get("upload_time", "") or "") != (prev.get("created_at", "") or ""):
            prev["created_at"] = best.get("upload_time", "") or ""
            prev["created_at_norm"] = normalize_time(prev.get("created_at", "") or "")
            changed = True
        if changed:
            updated += 1

    rows_after = list(by_key.values())
    unresolved_meta = sum(1 for r in rows_after if _row_missing_video_meta(r))
    filled_tid = sum(1 for r in rows_after if (r.get("tweet_id", "") or "").isdigit())
    _rewrite_video_items_ndjson(items_path, rows_after)
    print(
        f"message: VIDEO_META_REPAIR updated={updated}, unresolved={unresolved_meta}, "
        f"with_tweet_id={filled_tid}/{len(rows_after)}, "
        f"path={items_path}"
    )
    if unresolved_meta > 0:
        # Sweep by media numeric id across the whole loaded timeline.
        # This is robust when GraphQL/CDP body parsing yields few/no tweet_id mappings.
        sweep_updated = backfill_video_tid_by_keyid_sweep(
            by_key,
            max_steps=2600 if not preloaded_timeline else 1400,
            assume_at_bottom=(not preloaded_timeline),
        )
        if sweep_updated > 0:
            rows_after_sweep = list(by_key.values())
            unresolved_meta = sum(1 for r in rows_after_sweep if _row_missing_video_meta(r))
            filled_tid = sum(1 for r in rows_after_sweep if (r.get("tweet_id", "") or "").isdigit())
            _rewrite_video_items_ndjson(items_path, rows_after_sweep)
            print(
                f"message: VIDEO_META_REPAIR tid-sweep updated={sweep_updated}, "
                f"unresolved={unresolved_meta}, with_tweet_id={filled_tid}/{len(rows_after_sweep)}"
            )

    if unresolved_meta > 0:
        print("message: VIDEO_META_REPAIR fallback: focus scan start")
        add_updated = repair_video_items_meta_focus(items_path, rounds=18)
        rows_after2 = read_local_items_ndjson(items_path)
        unresolved_meta2, filled_tid2, total2 = _video_only_stats(rows_after2)
        print(
            f"message: VIDEO_META_REPAIR focus result updated={add_updated}, "
            f"unresolved={unresolved_meta2}, with_tweet_id={filled_tid2}/{total2}"
        )
        if unresolved_meta2 > 0:
            print("message: VIDEO_META_REPAIR fallback: strict scan start")
            strict_updated = repair_video_items_meta_strict(items_path, rounds=24)
            rows_after3 = read_local_items_ndjson(items_path)
            unresolved_meta3, filled_tid3, total3 = _video_only_stats(rows_after3)
            print(
                f"message: VIDEO_META_REPAIR strict result updated={strict_updated}, "
                f"unresolved={unresolved_meta3}, with_tweet_id={filled_tid3}/{total3}"
            )

    # Final pass: keep tweet_id fixed, fill missing author only.
    rows_final = read_local_items_ndjson(items_path)
    author_missing_before = sum(
        1 for r in rows_final if (r.get("tweet_id", "") or "").strip().isdigit() and not (r.get("author") or "").strip()
    )
    if author_missing_before > 0:
        author_updated = backfill_authors_in_video_rows(rows_final, rounds=60)
        _rewrite_video_items_ndjson(items_path, rows_final)
        author_missing_after = sum(
            1 for r in rows_final if (r.get("tweet_id", "") or "").strip().isdigit() and not (r.get("author") or "").strip()
        )
        print(
            f"message: VIDEO_META_REPAIR author-only pass updated={author_updated}, "
            f"missingAuthor={author_missing_before}->{author_missing_after}"
        )
    created_missing_before = sum(
        1 for r in rows_final if (r.get("tweet_id", "") or "").strip().isdigit() and not (r.get("created_at") or "").strip()
    )
    if created_missing_before > 0:
        created_updated = backfill_created_at_in_video_rows(rows_final, rounds=70)
        _rewrite_video_items_ndjson(items_path, rows_final)
        created_missing_after = sum(
            1 for r in rows_final if (r.get("tweet_id", "") or "").strip().isdigit() and not (r.get("created_at") or "").strip()
        )
        print(
            f"message: VIDEO_META_REPAIR createdAt-only pass updated={created_updated}, "
            f"missingCreatedAt={created_missing_before}->{created_missing_after}"
        )

def append_local_video_items_ndjson(items_path: str, entries: List[Dict[str, str]]) -> Tuple[int, int]:
    os.makedirs(os.path.dirname(items_path), exist_ok=True)
    existing_rows = read_local_items_ndjson(items_path)
    by_key: Dict[str, Dict[str, str]] = {}
    for obj in existing_rows:
        mk = (obj.get("media_key") or "").strip()
        if mk:
            by_key[mk] = dict(obj)

    added = 0
    updated = 0
    for it in entries:
        url = (it.get("url") or "").strip()
        if not is_video_url(url):
            continue
        mk = normalize_media_key(url) or ""
        if not mk:
            continue
        incoming = {
            "tweet_id": it.get("tweet_id", "") or "",
            "author": it.get("uploader_name", "") or "",
            "created_at": it.get("upload_time", "") or "",
            "created_at_norm": normalize_time(it.get("upload_time", "") or ""),
            "tid_source": it.get("tid_source", "") or "",
            "tid_confidence": it.get("tid_confidence", "") or "",
            "media_key": mk,
            "url": url,
        }
        if incoming["tweet_id"] and not incoming["tid_source"]:
            incoming["tid_source"] = "unknown"
        if incoming["tweet_id"] and not incoming["tid_confidence"]:
            incoming["tid_confidence"] = _tid_confidence(incoming["tid_source"])
        prev = by_key.get(mk)
        if prev is None:
            by_key[mk] = incoming
            added += 1
            continue

        changed = False
        for fld in ("tweet_id", "author", "created_at", "created_at_norm", "tid_source", "tid_confidence"):
            if not (prev.get(fld) or "") and (incoming.get(fld) or ""):
                prev[fld] = incoming[fld]
                changed = True
        # prefer non-aud / non-0/0 url if previous url was lower quality
        prev_url = prev.get("url", "") or ""
        if _video_quality_score(incoming["url"]) > _video_quality_score(prev_url):
            prev["url"] = incoming["url"]
            changed = True
        by_key[mk] = prev
        if changed:
            updated += 1

    with open(items_path, "w", encoding="utf-8") as f:
        for obj in by_key.values():
            out = {
                "tweet_id": obj.get("tweet_id", "") or "",
                "author": obj.get("author", "") or "",
                "created_at": obj.get("created_at", "") or "",
                "created_at_norm": obj.get("created_at_norm", "") or "",
                "tid_source": obj.get("tid_source", "") or "",
                "tid_confidence": obj.get("tid_confidence", "") or "",
                "media_key": obj.get("media_key", "") or "",
                "url": obj.get("url", "") or "",
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"message: local video ndjson merge summary: added={added}, updated_missing_meta={updated}")
    return added, len(by_key)

def write_open_by_tid_py(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "open_by_tid.py")
    content = f"""import sys
import re
import webbrowser

TID_TAG = {TID_TAG!r}

def extract_tid(path: str):
    m = re.search(re.escape(TID_TAG) + r"(\\d+)", path)
    return m.group(1) if m else None

def pick_file_gui():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    p = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.webp;*.gif;*.bmp"), ("All files", "*.*")]
    )
    root.destroy()
    return p or None

def main():
    if len(sys.argv) >= 2:
        p = sys.argv[1]
    else:
        p = pick_file_gui()
        if not p:
            print("message: no file selected.")
            return
    tid = extract_tid(p)
    if not tid:
        print("message: tid not found in filename:", p)
        print("message: expected pattern:", TID_TAG + "<digits>")
        return
    url = f"https://x.com/i/web/status/{{tid}}"
    print("message: open:", url)
    webbrowser.open(url)

if __name__ == "__main__":
    main()
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

def ask_download_now() -> bool:
    print("\n============================================================")
    print("Collection finished. Choose next action:")
    print("  1) Download now")
    print("  2) Exit without downloading")
    print("============================================================")
    print("Press '1' or '2'...")
    while True:
        ch = msvcrt.getwch()
        if ch == "1":
            return True
        if ch == "2":
            return False

def ask_repair_now() -> bool:
    print("\n============================================================")
    print("Before download:")
    print("  1) Run VIDEO_META_REPAIR now (reuse currently loaded timeline)")
    print("  2) Skip repair and continue")
    print("============================================================")
    print("Press '1' or '2'...")
    while True:
        ch = msvcrt.getwch()
        if ch == "1":
            return True
        if ch == "2":
            return False

def run_download(entries: List[Dict[str, str]], out_dir: str) -> Tuple[int, int, int, int]:
    print("message: Start downloading images...")
    open_by_tid_path = write_open_by_tid_py(out_dir)
    print(f"message: opener created: {open_by_tid_path}")

    result_file_path = os.path.join(out_dir, "download_result.txt")
    write_lock = threading.Lock()
    fail_count = 0
    ok_count = 0
    skip_exists_count = 0
    non_meta_ok_count = 0

    from concurrent.futures import Future
    with open(result_file_path, "w", encoding="utf-8") as result_file:
        future_to_entry: Dict[Future, Dict[str, str]] = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for idx_, entry in enumerate(entries, start=1):
                fu = ex.submit(download_one, idx_, entry, out_dir)
                future_to_entry[fu] = entry
            for fu in tqdm(as_completed(list(future_to_entry.keys())), total=len(future_to_entry), desc="Downloading", unit="file"):
                ok, fname, url, err, status = fu.result()
                entry = future_to_entry[fu]
                has_meta = bool(entry.get("uploader_name") or entry.get("upload_time"))
                with write_lock:
                    if ok:
                        if status == "skip_exists":
                            skip_exists_count += 1
                        else:
                            ok_count += 1
                            if not has_meta:
                                non_meta_ok_count += 1
                        mk = normalize_media_key(url) or ""
                        meta_flag = "" if has_meta else " MISSING_META"
                        result_file.write(f"OK file={fname} status={status} URL={url} KEY={mk}{meta_flag}\n")
                    else:
                        fail_count += 1
                        result_file.write(f"FAIL file={fname} err={err} URL={url}\n")
                        print(f"message: Failed to download {url}: {err}")

    print(f"message: Downloaded new files: {ok_count}")
    print(f"message: Skipped already-downloaded files: {skip_exists_count}")
    print(f"message: Saved non-meta(raw) images: {non_meta_ok_count}")
    print(f"message: Number of failed downloads: {fail_count}")
    print(f"message: result_log={result_file_path}")
    return ok_count, skip_exists_count, non_meta_ok_count, fail_count

# -----------------------------------------------------------------------------
# Main flow
# -----------------------------------------------------------------------------
image_entries: List[Dict[str, str]] = []

if mode == "NDJSON_ONLY":
    nd_items = read_local_items_ndjson(BOOKMARK_META_LOCAL_ITEMS_PATH)
    print(f"message: NDJSON_ONLY loaded items={len(nd_items)} from {BOOKMARK_META_LOCAL_ITEMS_PATH}")
    for obj in nd_items:
        image_entries.append({
            "url": obj.get("url", "") or "",
            "uploader_name": obj.get("author", "") or "",
            "upload_time": obj.get("created_at", "") or "",
            "tweet_id": obj.get("tweet_id", "") or "",
        })
    _ = run_download(image_entries, new_folder_path)
else:
    # IO/MutationObserver는 하강 시작 전에 주입(두 모드 공통으로 메타를 최대한 확보)
    bootstrap_observers()

    if mode == "VIDEO_META_REPAIR":
        repair_video_items_meta(BOOKMARK_META_LOCAL_ITEMS_PATH)
        try:
            driver.quit()
        except Exception:
            pass
        print("message: VIDEO_META_REPAIR completed.")
        sys.exit(0)

    # === MOD: 하강 시작 전에 top jiggle 1회로 초기 로딩 유도
    pre_descent_jiggle(set(), {})  # 초기 CDP 누계 의미 없으므로 더미로 호출, 메시지 용도

    # 1) 완전 하강 (도중에 주기적으로 CDP/IO drain 및 로그)
    cdp_seen_keys: set[str] = set()
    cdp_url_by_key: Dict[str, str] = {}
    cdp_meta_by_key: Dict[str, Dict[str, str]] = {}
    desc_meta_by_key: Dict[str, Dict[str, str]] = {}  # 하강 중 IO로 모은 메타(키 -> 메타)
    already_saved_keys = load_seen_media_keys_from_local(BOOKMARK_META_LOCAL_ITEMS_PATH)
    print(f"message: backup_mode={backup_mode}, loaded existing local keys={len(already_saved_keys)}")
    stop_keys = already_saved_keys if backup_mode == "PERIODIC" else None
    cdp_peak = full_descent(
        cdp_seen_keys,
        cdp_url_by_key,
        desc_meta_by_key,
        stop_when_seen_keys=stop_keys,
        cdp_meta_by_key=cdp_meta_by_key,
    )

    # 2) 하강 직후 CDP를 한 번 더 drain -> 타겟 최종 확정
    _ = update_cdp_seen_from_logs_with_keys(cdp_seen_keys, cdp_url_by_key, cdp_meta_by_key=cdp_meta_by_key)
    target_url_by_key: Dict[str, str] = dict(cdp_url_by_key)  # key -> canonical URL
    target_keys = set(target_url_by_key.keys())
    TargetTotalSeen = len(target_keys)
    print(f"message: CDP drain after descent: TargetTotalSeen={TargetTotalSeen}, cdpPeakDuringDescent={cdp_peak}")

    if backup_mode == "PERIODIC":
        run_target_keys = {k for k in target_keys if k not in already_saved_keys}
        print(f"message: periodic filter applied. newTarget={len(run_target_keys)}, alreadyKnown={len(target_keys) - len(run_target_keys)}")
    else:
        run_target_keys = set(target_keys)
        print(f"message: full mode target={len(run_target_keys)} (no early-stop filter)")

    # 2.5) CDP_ONLY 자동 폴백 판단
    if mode == "CDP_ONLY" and CDP_ONLY_AUTOFALLBACK:
        min_allowed = max(CDP_ONLY_MIN_KEYS, int(cdp_peak * CDP_ONLY_MIN_RATIO_OF_PEAK))
        if TargetTotalSeen < min_allowed:
            print(f"message: CDP_ONLY target too low (Target={TargetTotalSeen} < MinAllowed={min_allowed}). Auto-fallback to SAFE.")
            mode = "SAFE"

    # 3) 수집/다운로드 엔트리 구성 (IMAGE_ONLY / ALL 분기)
    run_image_keys: set[str] = set()
    run_video_keys: set[str] = set()
    for k in run_target_keys:
        u = target_url_by_key.get(k, "") or ""
        if (media_mode == "ALL") and is_video_url(u):
            run_video_keys.add(k)
        else:
            run_image_keys.add(k)
    print(f"message: media_mode={media_mode}, imageTargets={len(run_image_keys)}, videoTargets={len(run_video_keys)}")

    all_entries: List[Dict[str, str]] = []

    # 이미지 경로: 기존 로직 그대로
    missing_keys: List[str] = []
    if mode == "CDP_ONLY":
        for k in run_image_keys:
            meta = pick_better_meta(
                desc_meta_by_key.get(k, {"uploader_name": "", "upload_time": "", "tweet_id": ""}),
                cdp_meta_by_key.get(k, {"uploader_name": "", "upload_time": "", "tweet_id": ""}),
            )
            all_entries.append({
                "url": target_url_by_key[k],
                "uploader_name": meta.get("uploader_name", ""),
                "upload_time": meta.get("upload_time", ""),
                "tweet_id": meta.get("tweet_id", ""),
            })
    else:
        confirmed_map_img = {}
        confirmed_keys_img = set()
        if run_image_keys:
            confirmed_map_img, confirmed_keys_img, missing_keys = safe_upward_collect(
                run_image_keys,
                {k: target_url_by_key[k] for k in run_image_keys},
                cdp_seen_keys_ref=cdp_seen_keys,
                cdp_url_by_key_ref=cdp_url_by_key,
                cdp_meta_by_key_ref=cdp_meta_by_key,
            )
        for k in run_image_keys:
            if k in confirmed_map_img:
                best_meta = pick_better_meta(
                    confirmed_map_img[k],
                    cdp_meta_by_key.get(k, {"uploader_name": "", "upload_time": "", "tweet_id": ""}),
                )
                all_entries.append({
                    "url": confirmed_map_img[k]["url"],
                    "uploader_name": best_meta.get("uploader_name", ""),
                    "upload_time": best_meta.get("upload_time", ""),
                    "tweet_id": best_meta.get("tweet_id", ""),
                })
            else:
                fallback_meta = pick_better_meta(
                    desc_meta_by_key.get(k, {"uploader_name": "", "upload_time": "", "tweet_id": ""}),
                    cdp_meta_by_key.get(k, {"uploader_name": "", "upload_time": "", "tweet_id": ""}),
                )
                all_entries.append({
                    "url": target_url_by_key[k],
                    "uploader_name": fallback_meta.get("uploader_name", ""),
                    "upload_time": fallback_meta.get("upload_time", ""),
                    "tweet_id": fallback_meta.get("tweet_id", ""),
                })

    # ALL 모드일 때만 비디오 보강 파이프라인 수행
    if media_mode == "ALL" and run_video_keys:
        # 비디오 URL은 이미 CDP에서 확보되어 있으므로, 여기서는 가벼운 GraphQL/CDP 메타만 반영한다.
        # 무거운 author/date 전체 스캔은 사용자가 다운로드 직전 repair를 선택했을 때 수행한다.
        gql_updates = backfill_tweet_id_from_graphql_bodies(cdp_meta_by_key)
        print(f"message: ALL video meta pass targets={len(run_video_keys)}, gqlBackfill={gql_updates}")
        video_entries: List[Dict[str, str]] = []
        for k in run_video_keys:
            cdp_meta_k, cdp_meta_key = _lookup_meta_by_video_key_alias(k, cdp_meta_by_key)
            tid_src = _source_for_video_key_alias(k, cdp_meta_key)
            tid_conf = _tid_confidence(tid_src)
            meta = pick_better_meta(
                desc_meta_by_key.get(k, {"uploader_name": "", "upload_time": "", "tweet_id": ""}),
                cdp_meta_k or {"uploader_name": "", "upload_time": "", "tweet_id": ""},
            )
            video_entries.append({
                "url": target_url_by_key[k],
                "uploader_name": meta.get("uploader_name", ""),
                "upload_time": meta.get("upload_time", ""),
                "tweet_id": meta.get("tweet_id", ""),
                "tid_source": tid_src,
                "tid_confidence": tid_conf,
            })
        missing_author_video = sum(1 for v in video_entries if (v.get("tweet_id") or "").strip().isdigit() and not (v.get("uploader_name") or "").strip())
        print(f"message: ALL video pre-repair summary missingAuthorVideo={missing_author_video}/{len(video_entries)}")
        all_entries.extend(video_entries)

    enrich_updated = enrich_entries_with_dom_meta(all_entries)
    print(f"message: pre-save meta enrich updated={enrich_updated}")

    # 3.5) local 메타 누적 저장(bookmark_meta_local/items.ndjson) - 이미지/동영상 통합
    local_added, local_total = append_local_items_ndjson(BOOKMARK_META_LOCAL_ITEMS_PATH, all_entries)
    print(f"message: local ndjson appended={local_added}, total={local_total}, path={BOOKMARK_META_LOCAL_ITEMS_PATH}")

    # 3.6) 다운로드 전에 같은 세션에서 즉시 repair 수행 여부 선택 (ALL 모드만)
    if media_mode == "ALL":
        if ask_repair_now():
            # SAFE-Up 이후 top 근처에 있어도 repair 내부에서 기록된 descent bottom으로 복귀한 뒤
            # author/date 전체 backfill을 수행한다.
            repair_video_items_meta(
                BOOKMARK_META_LOCAL_ITEMS_PATH,
                preloaded_timeline=True,
                preloaded_cdp_seen_keys=cdp_seen_keys,
                preloaded_cdp_url_by_key=cdp_url_by_key,
                preloaded_cdp_meta_by_key=cdp_meta_by_key,
            )
        else:
            print("message: repair skipped by user choice.")

    # 4) 수집 완료 후 다운로드 여부 선택
    if ask_download_now():
        _ = run_download(all_entries, new_folder_path)
    else:
        print("message: download skipped by user choice. exiting.")

# 종료
try:
    driver.quit()
except Exception:
    pass
print("message: Process completed. The Chrome window automatically closed.")

try:
    sys.stdout.log.close()
    sys.stdout = sys.stdout.terminal
except Exception:
    pass



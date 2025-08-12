# X bookmark backuptool by qus20000
# Windows / Python 3.9.x ~ 3.10.x 권장
#
# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------
# 본 스크립트는 X(트위터) 북마크 페이지에서 이미지 URL을 수집하고 이미지를 다운로드하는 도구입니다.
#
# 공통 동작(본 버전: IO 제외):
# - 1) 완전 하강(북마크 페이지에서 스크롤 최하단까지 탐지). 하강 중에는 CDP(Network.*)만 사용하여
#      pbs.twimg.com/media/... 요청을 수집 → Target 집합 구성
# - 2) 바닥 도달 직후 near-bottom 프리로드 루프(±0.5*vh 왕복 + 짧은 대기/드레인)로 누락 최소화
# - 3) 모드에 따라:
#      - CDP_ONLY: 하강+프리로드 직후 바로 다운로드(업스크롤 없음). 메타는 비어 있을 수 있음.
#      - SAFE    : 업스크롤하며 DOM 스냅샷(JS 실행)으로 메타를 채우고, 관측된 새 URL은 Target에 즉시 편입(Union)
#
# 로그 정책:
#   - 하강 단계: 매 Burst 진행 상황 + CDP 누적/증가치 + viewport/scale/stepPxEff 로그
#   - near-bottom: 왕복 루프 단계별 CDP 증가치
#   - 업스크롤 단계: 단일 라인 포맷
#     debug: [MODE=SAFE] scrollstep=..., newURL=..., dupURL=..., batchSize=..., jsCalls=..., yOffset=..., TargetTotalSeen=..., CurrentTotalSeen=..., UnionTotalSeen=...
#   - 종료 요약: CDPOnly, Safe(Union) 최종 개수, 다운로드 통계
#
# 파일명 정책(결정적):
#   - "meta_if_available": 메타 있으면 uploader_time_key.ext, 없으면 key.ext (MEDIA_KEY 항상 포함)
#   - "key_only":          항상 key.ext
#
# 배포/튜닝:
#   - CONFIG 섹션만 수정해서 환경/성능 튜닝 가능
#   - 본 버전은 IntersectionObserver(=IO) 경로를 제거했음(원복 포인트 주석 참조)
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

# -----------------------------------------------------------------------------
# CONFIG: 사용자 튜닝 파라미터
# -----------------------------------------------------------------------------
CONFIG = {
    # 연결/프로필
    "DEBUGGER_ADDRESS": "127.0.0.1:9222",
    "USER_DATA_DIR": os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "User Data"),
    "PROFILE_DIR_NAME": "Default",

    # 하강(북마크 페이지에서 스크롤 최하단까지 탐지할 때 사용되는 파라미터입니다.)
    "DOWN_SCROLL_BURST": 60,         # Burst당 스크롤 횟수
    # NOTE: 본 버전은 DOWN_STEP_PX 대신 viewport 기반 DOWN_STEP_PX_EFF를 사용(아래 가변 로직)
    "DOWN_STEP_PX": 1800,            # (원복 포인트) 고정 스텝을 쓰고 싶을 때 사용
    "DOWN_DELAY_S": 0.02,            # 스크롤 호출 사이 대기
    "DOWN_BUFFER_CHECKS": 6,         # Burst 후 scrollHeight 증가 체크 횟수
    "DOWN_BUFFER_SLEEP_S": 0.18,     # 각 체크 간 대기
    "DOWN_STALL_TOLERANCE": 3,       # scrollHeight 증가 정지 연속 허용 횟수
    "YOFFSET_STALL_BURSTS": 5,       # 연속 Burst 동안 yOffset 변화 없음 허용 횟수
    "YOFFSET_EPS": 8,                # yOffset 변화 유효성 오차(px)
    "DESCENT_CDP_LOG_INTERVAL": 1,   # 하강 중 CDP drain/로그 주기(Burst 단위, 1=매 Burst)

    # viewport 기반 하강 스텝 비율(가변 로직)
    "DOWN_STEP_RATIO_MIN": 0.60,     # 0.6 * vh
    "DOWN_STEP_RATIO_MAX": 0.90,     # 0.9 * vh
    "DOWN_STEP_RATIO_DEFAULT": 0.80, # 기본 선택 비율(0.8 * vh) 

    # near-bottom 프리로드 루프(바닥 감지 직후 왕복으로 미디어 요청 유도)
    "NB_SWINGS": 3,                  # 위↔아래 왕복 횟수(증가 시 수 초 더 소요)
    "NB_STEP_RATIO": 0.50,           # 한 번 이동 크기 = NB_STEP_RATIO * vh
    "NB_WAIT_S": 0.08,               # 각 이동 후 대기(짧게, CDP 요청 발생 여유)

    # CDP 안정화 옵션
    "CDP_DISABLE_CACHE": True,               # Network.setCacheDisabled(true)로 캐시 무효화
    "HARD_RELOAD_ON_BOOKMARKS": True,        # 북마크 진입 직후 Page.reload(ignoreCache=true)

    # 모드 및 SAFE 보조 옵션
    "UP_STEP_PX": 4000,              # 의도 스텝 상한(실제 stepPxEff는 뷰포트 기반으로 클램프)
    "UP_DELAY_S": 0.04,              # 배치 폴링/스크롤 settle 대기 최소단위
    "VIEWPORT_PAD": 300,             # 수집 패딩(px)
    "SAFE_OVERLAP_RATIO": 0.30,      # 커버리지 대비 최소 겹침 비율(0.4~0.6 권장)
    "SAFE_ALLOW_UNION_EXPAND": True, # 업스크롤 중 DOM에서 새 URL 발견 시 Target에 즉시 편입

    # 다운로드
    "MAX_WORKERS": 10,               # 이미지 병렬 다운로드 스레드 수

    # 파일명 정책(결정적 파일명 + 존재시 스킵)
    "FILENAME_MODE": "meta_if_available",    # "meta_if_available" | "key_only"
    "SKIP_IF_EXISTS": True,          # 같은 파일명 있으면 다운로드 스킵
}

# 매크로 변수로 CONFIG 값 할당
DEBUGGER_ADDRESS     = CONFIG["DEBUGGER_ADDRESS"]
USER_DATA_DIR        = CONFIG["USER_DATA_DIR"]
PROFILE_DIR_NAME     = CONFIG["PROFILE_DIR_NAME"]

DOWN_SCROLL_BURST    = CONFIG["DOWN_SCROLL_BURST"]
DOWN_STEP_PX         = CONFIG["DOWN_STEP_PX"]  # (고정 스텝 원복 포인트)
DOWN_DELAY_S         = CONFIG["DOWN_DELAY_S"]
DOWN_BUFFER_CHECKS   = CONFIG["DOWN_BUFFER_CHECKS"]
DOWN_BUFFER_SLEEP_S  = CONFIG["DOWN_BUFFER_SLEEP_S"]
DOWN_STALL_TOLERANCE = CONFIG["DOWN_STALL_TOLERANCE"]
YOFFSET_STALL_BURSTS = CONFIG["YOFFSET_STALL_BURSTS"]
YOFFSET_EPS          = CONFIG["YOFFSET_EPS"]
DESCENT_CDP_LOG_INTERVAL = CONFIG["DESCENT_CDP_LOG_INTERVAL"]

DOWN_STEP_RATIO_MIN  = CONFIG["DOWN_STEP_RATIO_MIN"]
DOWN_STEP_RATIO_MAX  = CONFIG["DOWN_STEP_RATIO_MAX"]
DOWN_STEP_RATIO_DEFAULT = CONFIG["DOWN_STEP_RATIO_DEFAULT"]

NB_SWINGS            = CONFIG["NB_SWINGS"]
NB_STEP_RATIO        = CONFIG["NB_STEP_RATIO"]
NB_WAIT_S            = CONFIG["NB_WAIT_S"]

CDP_DISABLE_CACHE          = CONFIG["CDP_DISABLE_CACHE"]
HARD_RELOAD_ON_BOOKMARKS   = CONFIG["HARD_RELOAD_ON_BOOKMARKS"]

UP_STEP_PX           = CONFIG["UP_STEP_PX"]
UP_DELAY_S           = CONFIG["UP_DELAY_S"]
VIEWPORT_PAD         = CONFIG["VIEWPORT_PAD"]
SAFE_OVERLAP_RATIO   = CONFIG["SAFE_OVERLAP_RATIO"]
SAFE_ALLOW_UNION_EXPAND = CONFIG["SAFE_ALLOW_UNION_EXPAND"]

MAX_WORKERS          = CONFIG["MAX_WORKERS"]
FILENAME_MODE        = CONFIG["FILENAME_MODE"]
SKIP_IF_EXISTS       = CONFIG["SKIP_IF_EXISTS"]

# -----------------------------------------------------------------------------
# Dependencies: pip auto-installer
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
from PIL import Image  # 미사용 가능
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

def apply_common_chrome_options(options: webdriver.ChromeOptions) -> None:
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-software-rasterizer")
    options.add_argument("--disable-webgpu")
    options.add_argument("--disable-accelerated-2d-canvas")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    # CDP 성능 로그 활성화 (Network.* 이벤트 수집)
    options.set_capability("goog:loggingPrefs", {"performance": "ALL", "browser": "ALL"})

def try_attach_existing_chrome(service: Service):
    try:
        options = webdriver.ChromeOptions()
        apply_common_chrome_options(options)
        options.add_experimental_option("debuggerAddress", DEBUGGER_ADDRESS)
        driver = webdriver.Chrome(service=service, options=options)
        print("message: attached to existing Chrome via debuggerAddress.")
        return driver
    except Exception:
        return None

def try_launch_with_profile(service: Service):
    try:
        options = webdriver.ChromeOptions()
        apply_common_chrome_options(options)
        if USER_DATA_DIR and os.path.isdir(USER_DATA_DIR):
            options.add_argument(f'--user-data-dir={USER_DATA_DIR}')
            options.add_argument(f'--profile-directory={PROFILE_DIR_NAME}')
        driver = webdriver.Chrome(service=service, options=options)
        print("message: launched Chrome with existing user profile.")
        return driver
    except Exception:
        return None

def launch_fresh_session(service: Service):
    options = webdriver.ChromeOptions()
    apply_common_chrome_options(options)
    driver = webdriver.Chrome(service=service, options=options)
    print("message: launched fresh Chrome session.")
    return driver

service = get_chromedriver_service()
driver = try_attach_existing_chrome(service)
attached_mode = False
if driver is not None:
    attached_mode = True
else:
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
if attached_mode:
    driver.get("https://x.com/i/bookmarks")
else:
    driver.get("https://x.com/login")
    print('message: Please log in manually, and press "Enter" key to continue...')
    while True:
        ch = msvcrt.getwch()
        if ch == "\r":
            break
    driver.get("https://x.com/i/bookmarks")

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

folder_base_name = "images"
new_folder_path = folder_base_name
idx = 0
while os.path.exists(new_folder_path):
    new_folder_path = f"{folder_base_name}{idx}"
    idx += 1
os.mkdir(new_folder_path)

log_file_path = os.path.join(new_folder_path, "log.txt")
sys.stdout = Logger(log_file_path)

# -----------------------------------------------------------------------------
# Mode selector (1= CDP_ONLY, 2 = SAFE)
# -----------------------------------------------------------------------------
print("\n============================================================")
print("Select mode:")
print("  1) CDP_ONLY : Fastest. After full descent + near-bottom preload, drain CDP media URLs and download.")
print("  2) SAFE     : Recommended. After descent, upward scan with DOM snapshot; union-expand Target if enabled.")
print("============================================================")
print("Press '1' or '2' to start...")

mode = None
while True:
    ch = msvcrt.getwch()
    if ch == "1":
        mode = "CDP_ONLY"
        break
    if ch == "2":
        mode = "SAFE"
        break

print(f"message: selected mode = {mode}")

# -----------------------------------------------------------------------------
# JS DOM snapshot (SAFE에서만 사용 / IO 제외)
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
    const timeEl = art.querySelector('time');
    let dt = '';
    if (timeEl && timeEl.getAttribute('datetime')) {
      dt = timeEl.getAttribute('datetime');
      if (dt.endsWith('.000Z')) dt = dt.slice(0, -5);
    }
    let uploader = '';
    const spans = art.querySelectorAll('span');
    for (const s of spans) {
      if (s.textContent && s.textContent.includes('@')) { uploader = s.textContent; break; }
    }
    const imgs = art.querySelectorAll('img[src*="media"]');
    for (const im of imgs) {
      let src = im.getAttribute('src') || '';
      if (!src) continue;
      src = src.replace(/name=[^&]+/, 'name=orig');
      results.push({url: src, uploader_name: uploader, upload_time: dt});
    }
  } catch(e) {}
}
return results;
"""

# -----------------------------------------------------------------------------
# Utilities 
# -----------------------------------------------------------------------------
def normalize_media_key(url: str) -> str | None:
    """pbs.twimg.com/media/<MEDIA_KEY>[.ext]?... → MEDIA_KEY 추출(실패 시 None)."""
    try:
        if "pbs.twimg.com/media/" not in url:
            return None
        m = re.search(r"/media/([^/.?]+)", url)
        if not m:
            return None
        return m.group(1)
    except Exception:
        return None

def canon_media_url(url: str) -> str:
    """name=orig 로 정규화(기존 파라미터 유지)."""
    try:
        if "name=" in url:
            url = re.sub(r"name=[^&]+", "name=orig", url)
        else:
            if "?" in url:
                url = url + "&name=orig"
            else:
                url = url + "?name=orig"
    except Exception:
        pass
    return url

def _guess_ext_from_url(url: str) -> str:
    m = re.search(r"[?&]format=([a-zA-Z0-9]+)", url)
    if m:
        return "." + m.group(1).lower()
    path = url.split("?", 1)[0]
    _, ext = os.path.splitext(path)
    if ext:
        return ext.lower()
    return ".jpg"

def _slug(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r'[\\/*?:"<>|]+', '', s)
    s = re.sub(r'\s+', '_', s)
    return s[:80]

def make_deterministic_filename(url: str, uploader_name: str, upload_time: str) -> str:
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

    if FILENAME_MODE == "meta_if_available":
        u = _slug(uploader_name)
        t = _slug(upload_time)
        if u or t:
            base = "_".join([x for x in [u, t, mk] if x])
            return f"{base}{ext}"

    return f"{mk}{ext}"

# -----------------------------------------------------------------------------
# CDP drain helpers
# -----------------------------------------------------------------------------
def drain_cdp_media() -> List[str]:
    # CDP 성능로그에서 media URL 추출(고유 URL, name=orig 정규화).
    urls: List[str] = []
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
            if method == "Network.requestWillBeSent":
                req = params.get("request", {})
                url = req.get("url")
            elif method == "Network.responseReceived":
                res = params.get("response", {})
                url = res.get("url")
            if not url or "pbs.twimg.com/media/" not in url:
                continue
            cu = canon_media_url(url)
            if cu not in seen:
                seen.add(cu)
                urls.append(cu)
        except Exception:
            continue
    return urls

def update_cdp_seen_from_logs(cdp_seen_keys: set[str], cdp_url_by_key: Dict[str, str]) -> int:
    # CDP drain → 고유 미디어키 집합/URL 맵 갱신. 반환: 이번 호출에서 새로 추가된 key 개수
    new_urls = drain_cdp_media()
    added = 0
    for u in new_urls:
        mk = normalize_media_key(u)
        if not mk:
            continue
        if mk not in cdp_seen_keys:
            cdp_seen_keys.add(mk)
            if mk not in cdp_url_by_key:
                cdp_url_by_key[mk] = u
            added += 1
    return added

# -----------------------------------------------------------------------------
# Descent: 완전 바닥 탐지 (+ CDP drain)
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

def _get_vh_and_scale() -> Tuple[int, float]:
    try:
        res = driver.execute_script("""
            return {
                vh: window.innerHeight || 900,
                scale: (window.visualViewport && window.visualViewport.scale) ? window.visualViewport.scale : 1.0
            };
        """) or {}
        vh = int(res.get("vh") or 900)
        scale = float(res.get("scale") or 1.0)
        return vh, scale
    except Exception:
        return 900, 1.0

def full_descent_and_preload(cdp_seen_keys: set[str], cdp_url_by_key: Dict[str, str]) -> Tuple[int, int, int, int]:
    """
    끝까지 하강 + near-bottom 프리로드 루프 수행.
    반환: (cdp_peak, vh, step_eff, total_added_in_nb)
    """
    time.sleep(2)
    print("message: full-descent mode start...")

    vh, scale = _get_vh_and_scale()
    # viewport 기반 DOWN_STEP_PX_EFF 계산 (기본 0.8*vh, [0.6, 0.9] 범위로 클램프)
    ratio = DOWN_STEP_RATIO_DEFAULT
    ratio = max(DOWN_STEP_RATIO_MIN, min(ratio, DOWN_STEP_RATIO_MAX))
    DOWN_STEP_PX_EFF = max(200, int(ratio * vh))  # (원복 포인트) 고정값 쓰려면 DOWN_STEP_PX 사용

    print(f"debug: viewportInnerHeight={vh}, scale={scale:.3f}, DOWN_STEP_PX_EFF={DOWN_STEP_PX_EFF}, "
          f"ratio={ratio:.2f}, burst={DOWN_SCROLL_BURST}")

    prev_h = _get_scroll_h()
    stall_cycles = 0
    burst_idx = 0
    last_y = _get_scroll_y()
    y_stall_seq = 0
    stop_reason = "unknown"
    cdp_peak = len(cdp_seen_keys)

    while True:
        burst_idx += 1

        # 빠른 하강: 버스트로 여러 번 내리기
        for _ in range(DOWN_SCROLL_BURST):
            driver.execute_script("window.scrollBy(0, arguments[0]);", DOWN_STEP_PX_EFF)
            time.sleep(DOWN_DELAY_S)

        # scrollHeight 증가 감시
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

        # yOffset 정지 감시
        cur_y = _get_scroll_y()
        delta_y = cur_y - last_y
        if abs(delta_y) <= YOFFSET_EPS:
            y_stall_seq += 1
        else:
            y_stall_seq = 0
            last_y = cur_y

        # 한번 더 찔러보기
        if not grew:
            driver.execute_script("window.scrollBy(0, arguments[0]);", DOWN_STEP_PX_EFF)
            time.sleep(DOWN_DELAY_S)
            cur_h2 = _get_scroll_h()
            if cur_h2 > prev_h:
                grew = True
                grew_px = cur_h2 - prev_h
                prev_h = cur_h2

        stall_cycles = 0 if grew else (stall_cycles + 1)

        # 주기적 CDP drain
        cdpNew = 0
        if (burst_idx % DESCENT_CDP_LOG_INTERVAL) == 0:
            cdpNew = update_cdp_seen_from_logs(cdp_seen_keys, cdp_url_by_key)
            cdp_peak = max(cdp_peak, len(cdp_seen_keys))
            print(
                f"debug: downBurst={burst_idx}, perBurstScrolls={DOWN_SCROLL_BURST}, "
                f"yOffset={cur_y}, deltaY={delta_y}, scrollHeight={prev_h}, grewPx={grew_px}, grew={int(grew)}, "
                f"heightStallSeq={stall_cycles}/{DOWN_STALL_TOLERANCE}, yStallSeq={y_stall_seq}/{YOFFSET_STALL_BURSTS}, "
                f"cdpKeys={len(cdp_seen_keys)}, cdpNew={cdpNew}"
            )
        else:
            print(
                f"debug: downBurst={burst_idx}, perBurstScrolls={DOWN_SCROLL_BURST}, "
                f"yOffset={cur_y}, deltaY={delta_y}, scrollHeight={prev_h}, grewPx={grew_px}, grew={int(grew)}, "
                f"heightStallSeq={stall_cycles}/{DOWN_STALL_TOLERANCE}, yStallSeq={y_stall_seq}/{YOFFSET_STALL_BURSTS}"
            )

        if stall_cycles >= DOWN_STALL_TOLERANCE or y_stall_seq >= YOFFSET_STALL_BURSTS:
            stop_reason = ("height-stall" if stall_cycles >= DOWN_STALL_TOLERANCE else "yoffset-stall")
            break

    # 최종 drain(마지막 남은 이벤트 수거)
    _ = update_cdp_seen_from_logs(cdp_seen_keys, cdp_url_by_key)

    # near-bottom 프리로드 루프(±0.5*vh 왕복)
    nb_step = max(50, int(NB_STEP_RATIO * vh))
    total_added_nb = 0
    base_y = _get_scroll_y()
    print(f"message: near-bottom preload start: swings={NB_SWINGS}, nb_step={nb_step}, wait={NB_WAIT_S}s")

    for swing in range(1, NB_SWINGS + 1):
        # 위로 약간
        driver.execute_script("window.scrollBy(0, arguments[0]);", -nb_step)
        time.sleep(NB_WAIT_S)
        a1 = update_cdp_seen_from_logs(cdp_seen_keys, cdp_url_by_key)
        total_added_nb += a1
        print(f"debug: near-bottom swing#{swing} up, yOffset={_get_scroll_y()}, cdpNew={a1}, cdpKeys={len(cdp_seen_keys)}")

        # 아래로 약간
        driver.execute_script("window.scrollBy(0, arguments[0]);", nb_step)
        time.sleep(NB_WAIT_S)
        a2 = update_cdp_seen_from_logs(cdp_seen_keys, cdp_url_by_key)
        total_added_nb += a2
        print(f"debug: near-bottom swing#{swing} down, yOffset={_get_scroll_y()}, cdpNew={a2}, cdpKeys={len(cdp_seen_keys)}")

    # 최종적으로 바닥으로 위치 고정
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(max(0.05, NB_WAIT_S / 2))
    _ = update_cdp_seen_from_logs(cdp_seen_keys, cdp_url_by_key)

    print(f"message: full-descent finished. stopReason={stop_reason}, yOffset={_get_scroll_y()}, scrollHeight={_get_scroll_h()}, "
          f"cdpKeys={len(cdp_seen_keys)}, vh={vh}, stepEff={DOWN_STEP_PX_EFF}, nbAdded={total_added_nb}")
    return cdp_peak, vh, DOWN_STEP_PX_EFF, total_added_nb

# -----------------------------------------------------------------------------
# SAFE: 업스크롤 수집(Union 확장 및 메타 확보; IO 제외, DOM 스냅샷만)
# -----------------------------------------------------------------------------
def _dom_snapshot_collect(pad: int) -> Tuple[List[Dict[str, str]], int]:
    """DOM 스냅샷 1회 호출. (items, jsCalls=1)"""
    try:
        dom_raw = driver.execute_script(JS_COLLECT_SNIPPET, pad) or []
    except Exception:
        dom_raw = []
    items: List[Dict[str, str]] = []
    for d in dom_raw:
        try:
            url = canon_media_url(d.get("url") or "")
            items.append({
                "url": url,
                "uploader_name": d.get("uploader_name") or "",
                "upload_time": d.get("upload_time") or "",
            })
        except Exception:
            continue
    return items, 1

def _poll_until_settled_union(target_keys: set[str],
                              union_keys: set[str],
                              union_url_by_key: Dict[str, str],
                              meta_map: Dict[str, Dict[str, str]]) -> Tuple[int,int,int,int]:
    """
    DOM 스냅샷을 짧게 여러 번 호출하여 안정화.
    - target_keys: 초기 CDP 타깃(집계용)
    - union_keys, union_url_by_key, meta_map: DOM에서 발견되면 즉시 편입/갱신
    반환: (new_total, dup_total, last_batch_size, js_calls)
    """
    STEP_MIN_SETTLE_S = 0.30
    idle_seq = 0
    t0 = time.time()
    new_total = 0
    dup_total = 0
    last_batch_size = 0
    js_calls = 0

    while True:
        items, jc = _dom_snapshot_collect(VIEWPORT_PAD)
        js_calls += jc
        last_batch_size = len(items)

        # 병합/집계
        step_new = 0
        step_dup = 0
        for it in items:
            url = it["url"]
            mk = normalize_media_key(url)
            if not mk:
                continue
            # union 확장
            existed = (mk in union_keys)
            if not existed and SAFE_ALLOW_UNION_EXPAND:
                union_keys.add(mk)
                union_url_by_key[mk] = url
                meta_map.setdefault(mk, {"uploader_name": "", "upload_time": ""})
                # 메타 갱신
                if it.get("uploader_name"):
                    meta_map[mk]["uploader_name"] = it["uploader_name"]
                if it.get("upload_time"):
                    meta_map[mk]["upload_time"] = it["upload_time"]
                step_new += 1
            else:
                # 이미 있던 키라도 메타가 비어있으면 채움
                if mk in union_keys:
                    meta_map.setdefault(mk, {"uploader_name": "", "upload_time": ""})
                    if it.get("uploader_name") and not meta_map[mk].get("uploader_name"):
                        meta_map[mk]["uploader_name"] = it["uploader_name"]
                    if it.get("upload_time") and not meta_map[mk].get("upload_time"):
                        meta_map[mk]["upload_time"] = it["upload_time"]
                    step_dup += 1

        new_total += step_new
        dup_total += step_dup

        if step_new == 0:
            idle_seq += 1
        else:
            idle_seq = 0

        if (idle_seq >= 2) and ((time.time() - t0) >= STEP_MIN_SETTLE_S):
            break
        time.sleep(max(UP_DELAY_S * 0.5, 0.02))

    return new_total, dup_total, last_batch_size, js_calls

def safe_upward_collect_union(initial_target_keys: set[str],
                              target_url_by_key: Dict[str, str]) -> Tuple[Dict[str, Dict[str, str]], set[str], List[str], Dict[str, str], set[str]]:
    """
    SAFE 업스크롤(Union 확장). 반환:
      - meta_map: 키 -> {url,uploader_name,upload_time}
      - confirmed_keys: 업스크롤 동안 DOM에서 실제 관측된 키 집합
      - missing_keys_against_initial: 초기 CDP 타깃 대비 미싱 키
      - union_url_by_key: 최종 Union의 url 맵
      - union_keys: 최종 Union 키 집합
    """
    # 초기 집합 준비
    union_keys: set[str] = set(initial_target_keys)  # 시작은 CDP 타깃
    union_url_by_key: Dict[str, str] = dict(target_url_by_key)  # CDP url
    meta_map: Dict[str, Dict[str, str]] = {}  # 키별 메타(업로더/시간)
    confirmed_keys: set[str] = set()

    # 스텝/커버리지 계산
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

    TargetTotalSeen = len(initial_target_keys)

    step = 0
    top_stall_seq = 0
    crawl_t0 = time.time()

    while True:
        step += 1
        curr_y = _get_scroll_y()

        if curr_y <= 2:
            a, d, b, js_calls = _poll_until_settled_union(initial_target_keys, union_keys, union_url_by_key, meta_map)
            print(f"debug: [MODE=SAFE] scrollstep={step}(final), newURL={a}, dupURL={d}, batchSize={b}, jsCalls={js_calls}, yOffset={_get_scroll_y()}, "
                  f"TargetTotalSeen={TargetTotalSeen}, CurrentTotalSeen={len(confirmed_keys)}, UnionTotalSeen={len(union_keys)}")
            break

        # 현재 뷰포트에서 스냅샷 → union 확장/메타 채우기
        new_total, dup_total, last_batch_size, total_js_calls = _poll_until_settled_union(initial_target_keys, union_keys, union_url_by_key, meta_map)
        # DOM에서 관측된 것 중 이번 스텝에서 새로 들어온 키를 confirmed로 간주(간단화)
        confirmed_keys.update(union_keys)  # 보수적: union으로 추가된 키를 관측된 것으로 처리
        print(f"debug: [MODE=SAFE] scrollstep={step}, newURL={new_total}, dupURL={dup_total}, batchSize={last_batch_size}, jsCalls={total_js_calls}, "
              f"yOffset={curr_y}, TargetTotalSeen={TargetTotalSeen}, CurrentTotalSeen={len(confirmed_keys)}, UnionTotalSeen={len(union_keys)}")

        prev_y = _get_scroll_y()
        driver.execute_script("window.scrollBy(0, arguments[0]);", -int(move_px))
        time.sleep(UP_DELAY_S)
        cur_y = _get_scroll_y()

        if cur_y >= prev_y - 1:
            top_stall_seq += 1
        else:
            top_stall_seq = 0

        if cur_y <= 2 and top_stall_seq >= 3:
            a, d, b, js_calls = _poll_until_settled_union(initial_target_keys, union_keys, union_url_by_key, meta_map)
            print(f"debug: [MODE=SAFE] scrollstep={step}(final), newURL={a}, dupURL={d}, batchSize={b}, jsCalls={js_calls}, yOffset={_get_scroll_y()}, "
                  f"TargetTotalSeen={TargetTotalSeen}, CurrentTotalSeen={len(confirmed_keys)}, UnionTotalSeen={len(union_keys)}")
            break

    elapsed = time.time() - crawl_t0
    missing_keys = [k for k in initial_target_keys if k not in union_keys]  # 초기 CDP 기준의 미싱(Union이 더 크면 0)
    print(f"message: SAFE upward collection finished in {elapsed:.2f} seconds")
    print(f"message: TargetTotalSeen(initial CDP)={TargetTotalSeen}, UnionTotalSeen(final)={len(union_keys)}, MissingAgainstInitial={len(missing_keys)}")

    # meta_map에 url도 채워서 반환(다운로드 편의)
    meta_out: Dict[str, Dict[str, str]] = {}
    for k in union_keys:
        meta = meta_map.get(k, {"uploader_name": "", "upload_time": ""})
        meta_out[k] = {
            "url": union_url_by_key.get(k, ""),
            "uploader_name": meta.get("uploader_name", ""),
            "upload_time": meta.get("upload_time", ""),
        }
    return meta_out, confirmed_keys, missing_keys, union_url_by_key, union_keys

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

def download_one(index: int, data: Dict[str, str], out_dir: str) -> Tuple[bool, str | None, str, str | None]:
    url = data["url"]
    uploader_name = data.get("uploader_name", "")
    upload_time = data.get("upload_time", "")

    filename = make_deterministic_filename(url, uploader_name, upload_time)
    file_path = os.path.join(out_dir, filename)

    if SKIP_IF_EXISTS and os.path.exists(file_path):
        return True, os.path.basename(file_path), url, None

    session = make_session()
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(resp.content)
        return True, os.path.basename(file_path), url, None
    except Exception as e:
        return False, None, url, str(e)
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

# -----------------------------------------------------------------------------
# Main flow
# -----------------------------------------------------------------------------
# 1) 완전 하강(+near-bottom 프리로드)
cdp_seen_keys: set[str] = set()
cdp_url_by_key: Dict[str, str] = {}
cdp_peak, vh, step_eff, nb_added = full_descent_and_preload(cdp_seen_keys, cdp_url_by_key)

# 2) 하강 직후 CDP를 한 번 더 drain -> 타깃 최종 확정(CDP 기준)
_ = update_cdp_seen_from_logs(cdp_seen_keys, cdp_url_by_key)
target_url_by_key: Dict[str, str] = dict(cdp_url_by_key)  # key -> canonical URL
initial_target_keys = set(target_url_by_key.keys())
print(f"message: CDP drain after descent: TargetTotalSeen={len(initial_target_keys)}, cdpPeakDuringDescent={cdp_peak}, nbAdded={nb_added}")

# 3) 수집/다운로드 엔트리 구성
image_entries: List[Dict[str, str]] = []

if mode == "CDP_ONLY":
    # 업스크롤 없이 즉시 다운로드 (메타 없음)
    for k in initial_target_keys:
        image_entries.append({
            "url": target_url_by_key[k],
            "uploader_name": "",
            "upload_time": ""
        })
else:
    # SAFE 모드: 업스크롤 수집(Union 확장 포함)
    meta_map, confirmed_keys, missing_keys, union_url_by_key, union_keys = safe_upward_collect_union(initial_target_keys, target_url_by_key)
    # 엔트리 구성: 최종 Union 기준
    for k in union_keys:
        m = meta_map.get(k, {"url": union_url_by_key.get(k, ""), "uploader_name": "", "upload_time": ""})
        if not m.get("url"):
            m["url"] = union_url_by_key.get(k, "")
        image_entries.append({
            "url": m["url"],
            "uploader_name": m.get("uploader_name", ""),
            "upload_time": m.get("upload_time", "")
        })
    # 리포트
    print(f"message: SAFE summary: InitialTarget={len(initial_target_keys)}, FinalUnion={len(union_keys)}, MissingAgainstInitial={len(missing_keys)}")
    if missing_keys:
        print("message: Missing detail (initial CDP but not seen in union):")
        for k in missing_keys[:50]:
            print(f"message: MISSING key={k} url={target_url_by_key.get(k, '')}")
        if len(missing_keys) > 50:
            print(f"message: ...and {len(missing_keys)-50} more")

# 4) 다운로드
print("message: Start downloading images...")
result_file_path = os.path.join(new_folder_path, "result.txt")
write_lock = threading.Lock()
pass_count = 0
ok_count = 0
non_meta_ok_count = 0  # 메타 없이 저장된 개수(성공분 기준)

from concurrent.futures import Future
with open(result_file_path, "w", encoding="utf-8") as result_file:
    future_to_entry: Dict[Future, Dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for idx_, entry in enumerate(image_entries, start=1):
            fu = ex.submit(download_one, idx_, entry, new_folder_path)
            future_to_entry[fu] = entry
        for fu in tqdm(as_completed(list(future_to_entry.keys())), total=len(future_to_entry), desc="Downloading", unit="file"):
            ok, fname, url, err = fu.result()
            entry = future_to_entry[fu]
            has_meta = bool(entry.get("uploader_name") or entry.get("upload_time"))
            with write_lock:
                if ok:
                    ok_count += 1
                    if not has_meta:
                        non_meta_ok_count += 1
                    mk = normalize_media_key(url) or ""
                    meta_flag = "" if has_meta else " MISSING_META"
                    result_file.write(f"{fname} , URL: {url} , KEY: {mk}{meta_flag}\n")
                else:
                    pass_count += 1
                    print(f"message: Failed to download {url}: {err}")

successful_downloads = ok_count
print(f"message: Total number of successfully downloaded images: {successful_downloads}")
print(f"message: Saved non-meta(raw) images: {non_meta_ok_count}")

# 5) 후처리: 중복 파일 정리 로그(선택)
dup_count, deleted_files = move_duplicate_images(new_folder_path)
print(f"message: Number of duplicate files: {dup_count}")
print(f"message: Number of deleted duplicate files: {deleted_files}")
print(f"message: Number of failed downloads: {pass_count}")
print(f"message: Number of errors: {0}")

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

# X bookmark backuptool by qus20000
# Windows / Python 3.9+
#
# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------
# 본 스크립트는 X(트위터) 북마크 페이지에서 이미지 URL을 백업한다.
# - 공통: "완전 하강(북마크 페이지에서 스크롤 최하단까지 탐지)"방식을 통해 모든 트윗이 로드된 상태를 확보
# - 모드:
#   (1) CDP_ONLY: 하강 직후 CDP 성능로그에서 pbs.twimg.com/media/... URL만으로 즉시 다운로드
#                 업스크롤/DOM 파싱/IntersectionObserver 미사용 → 가장 빠름
#                 업로더/시간 메타데이터는 획득이 제한적이므로 폴백 파일명 사용
#   (2) SAFE    : 하강 직후 CDP Target(고유 미디어키 집합) 확정
#                 업스크롤하면서 DOM+IO로 메타데이터를 확보해 Target과 매칭(누락 방지/품질↑)
#                 최종적으로 Target 대비 Current 매칭율과 미매칭 수(실패수)를 리포트
#
# 로그 정책:
#   - 하강 단계: 매 버스트마다 스크롤 진행 디버그( yOffset / deltaY / scrollHeight / grewPx / stall 시퀀스 )
#               + DESCENT_CDP_LOG_INTERVAL 버스트마다 CDP 수집 현황(cdpKeys) 병기
#   - 업스크롤 단계(SAFE): 한 줄 포맷으로만 출력
#     debug: [MODE=SAFE] scrollstep=1324, newURL=0, dupURL=56, batchSize=7, jsCalls=8, yOffset=4292, TargetTotalSeen=4830, CurrentTotalSeen=4805
#
# 배포/튜닝:
#   - CONFIG 섹션만 수정해서 환경/성능 튜닝 가능
#   - URL 정규화는 pbs.twimg.com/media/<MEDIA_KEY> 기준(쿼리/파라미터 무시)으로 통일
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

    # 하강(완전 바닥 탐지)
    "DOWN_SCROLL_BURST": 60,         # 버스트당 스크롤 횟수
    "DOWN_STEP_PX": 1800,            # 스크롤 1회 픽셀
    "DOWN_DELAY_S": 0.02,            # 스크롤 호출 사이 대기
    "DOWN_BUFFER_CHECKS": 6,         # 버스트 후 scrollHeight 증가 체크 횟수
    "DOWN_BUFFER_SLEEP_S": 0.18,     # 각 체크 간 대기
    "DOWN_STALL_TOLERANCE": 3,       # scrollHeight 증가 정지 연속 허용 횟수
    "YOFFSET_STALL_BURSTS": 5,       # 연속 버스트 동안 yOffset 변화 없음 허용 횟수(네트워크 지연 대비)
    "YOFFSET_EPS": 8,                # yOffset 변화 유효성 오차(px)
    "DESCENT_CDP_LOG_INTERVAL": 5,   # 하강 중 CDP drain 및 cdpKeys 로그 주기(버스트 단위)

    # 업스크롤(SAFE 모드에서만 사용)
    "UP_STEP_PX": 4000,              # 의도 스텝 상한(실제 stepPxEff는 뷰포트 기반으로 클램프)
    "UP_DELAY_S": 0.04,              # 배치 폴링/스크롤 settle 대기 최소단위
    "VIEWPORT_PAD": 300,             # 수집 패딩(px)
    "SAFE_OVERLAP_RATIO": 0.50,      # 커버리지 대비 최소 겹침 비율(0.4~0.6 권장)

    # 다운로드
    "MAX_WORKERS": 10,               # 이미지 병렬 다운로드 스레드 수
}

# 편의 바인딩
DEBUGGER_ADDRESS     = CONFIG["DEBUGGER_ADDRESS"]
USER_DATA_DIR        = CONFIG["USER_DATA_DIR"]
PROFILE_DIR_NAME     = CONFIG["PROFILE_DIR_NAME"]

DOWN_SCROLL_BURST    = CONFIG["DOWN_SCROLL_BURST"]
DOWN_STEP_PX         = CONFIG["DOWN_STEP_PX"]
DOWN_DELAY_S         = CONFIG["DOWN_DELAY_S"]
DOWN_BUFFER_CHECKS   = CONFIG["DOWN_BUFFER_CHECKS"]
DOWN_BUFFER_SLEEP_S  = CONFIG["DOWN_BUFFER_SLEEP_S"]
DOWN_STALL_TOLERANCE = CONFIG["DOWN_STALL_TOLERANCE"]
YOFFSET_STALL_BURSTS = CONFIG["YOFFSET_STALL_BURSTS"]
YOFFSET_EPS          = CONFIG["YOFFSET_EPS"]
DESCENT_CDP_LOG_INTERVAL = CONFIG["DESCENT_CDP_LOG_INTERVAL"]

UP_STEP_PX           = CONFIG["UP_STEP_PX"]
UP_DELAY_S           = CONFIG["UP_DELAY_S"]
VIEWPORT_PAD         = CONFIG["VIEWPORT_PAD"]
SAFE_OVERLAP_RATIO   = CONFIG["SAFE_OVERLAP_RATIO"]

MAX_WORKERS          = CONFIG["MAX_WORKERS"]

# -----------------------------------------------------------------------------
# Dependencies: pip auto-install
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
from PIL import Image  # 유지(향후 후처리 대비)
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

# CDP Network.enable
try:
    driver.execute_cdp_cmd("Network.enable", {})
except Exception as _e:
    print(f"message: Network.enable failed or not supported: {_e}")

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
# Mode select (1=CDP_ONLY, 2=SAFE)
# -----------------------------------------------------------------------------
print("\n============================================================")
print("Select mode:")
print("  1) CDP_ONLY : Fastest. After full descent, drain CDP media URLs and download immediately.")
print("                No upward scan / No DOM parsing / No IntersectionObserver.")
print("                Uploader/Time metadata may be missing (fallback filename).")
print("  2) SAFE     : Recommended. After full descent, fix CDP Target, then upward scan.")
print("                Use DOM+IntersectionObserver to enrich metadata and match Target.")
print("                Reports Target vs Current (matched) and missing count.")
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
# JS Collectors (SAFE 모드에서 사용)
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

    function __xPush(url, uploader, dt) {
      if (!url) return;
      url = url.replace(/name=[^&]+/, 'name=orig');
      if (window.__xSeen.has(url)) return;
      window.__xSeen.add(url);
      window.__xBuf.push({url, uploader_name: uploader || '', upload_time: dt || ''});
    }

    function __xExtractFromArticle(art) {
      try {
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
          const src = im.getAttribute('src') || '';
          if (src) __xPush(src, uploader, dt);
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
    """
    pbs.twimg.com/media/<MEDIA_KEY>[.ext]?... → MEDIA_KEY 추출
    쿼리/파라미터는 무시. 못 찾으면 None.
    """
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
    """name=orig 로 정규화(기존 파라미터는 유지)."""
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

def drain_cdp_media() -> List[str]:
    """CDP 성능로그에서 media URL 추출(고유 URL, orig 정규화)."""
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
    """
    CDP 로그를 drain 해서 현재까지의 고유 미디어키 집합/URL 맵을 갱신한다.
    반환: 이번 호출에서 새로 추가된 key 개수
    """
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
# Descent: 완전 바닥 탐지 (+ 간헐 CDP 수집 로그)
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

def full_descent(cdp_seen_keys: set[str], cdp_url_by_key: Dict[str, str]):
    time.sleep(2)
    print("message: full-descent mode start...")

    prev_h = _get_scroll_h()
    stall_cycles = 0
    burst_idx = 0
    last_y = _get_scroll_y()
    y_stall_seq = 0
    stop_reason = "unknown"

    # CDP 진행 현황
    cdp_key_count = len(cdp_seen_keys)

    while True:
        burst_idx += 1

        for _ in range(DOWN_SCROLL_BURST):
            driver.execute_script(f"window.scrollBy(0, {DOWN_STEP_PX});")
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
            driver.execute_script("window.scrollBy(0, arguments[0]);", DOWN_STEP_PX)
            time.sleep(DOWN_DELAY_S)
            cur_h2 = _get_scroll_h()
            if cur_h2 > prev_h:
                grew = True
                grew_px = cur_h2 - prev_h
                prev_h = cur_h2

        stall_cycles = 0 if grew else (stall_cycles + 1)

        # 주기적으로 CDP drain 및 로그
        if (burst_idx % DESCENT_CDP_LOG_INTERVAL) == 0:
            new_added = update_cdp_seen_from_logs(cdp_seen_keys, cdp_url_by_key)
            cdp_key_count = len(cdp_seen_keys)
            print(
                f"debug: downBurst={burst_idx}, perBurstScrolls={DOWN_SCROLL_BURST}, "
                f"yOffset={cur_y}, deltaY={delta_y}, scrollHeight={prev_h}, grewPx={grew_px}, grew={int(grew)}, "
                f"heightStallSeq={stall_cycles}/{DOWN_STALL_TOLERANCE}, yStallSeq={y_stall_seq}/{YOFFSET_STALL_BURSTS}, "
                f"cdpKeys={cdp_key_count}, cdpNew={new_added}"
            )
        else:
            print(
                f"debug: downBurst={burst_idx}, perBurstScrolls={DOWN_SCROLL_BURST}, "
                f"yOffset={cur_y}, deltaY={delta_y}, scrollHeight={prev_h}, grewPx={grew_px}, grew={int(grew)}, "
                f"heightStallSeq={stall_cycles}/{DOWN_STALL_TOLERANCE}, yStallSeq={y_stall_seq}/{YOFFSET_STALL_BURSTS}"
            )

        if stall_cycles >= DOWN_STALL_TOLERANCE:
            stop_reason = f"height-stall x{stall_cycles}"
            break
        if y_stall_seq >= YOFFSET_STALL_BURSTS:
            stop_reason = f"yoffset-stall x{y_stall_seq}"
            break

    print(f"message: full-descent finished. stopReason={stop_reason}, yOffset={_get_scroll_y()}, scrollHeight={_get_scroll_h()}, cdpKeys={len(cdp_seen_keys)}")

# -----------------------------------------------------------------------------
# SAFE: 업스크롤 수집(타겟 매칭 및 메타 확보)
# -----------------------------------------------------------------------------
def bootstrap_observers():
    try:
        ok = driver.execute_script(JS_OBSERVER_BOOTSTRAP)
        print(f"message: observer bootstrap: {'ok' if ok else 'failed'}")
    except Exception as e:
        print(f"message: observer bootstrap error: {e}")

def flush_io_buffer() -> List[Dict[str, str]]:
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
            })
        except Exception:
            continue
    return out

def safe_upward_collect(target_keys: set[str], target_url_by_key: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    confirmed: Dict[str, Dict[str, str]] = {}
    confirmed_keys: set[str] = set()

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
    CurrentTotalSeen = 0

    step = 0
    top_stall_seq = 0
    crawl_t0 = time.time()

    while True:
        step += 1
        curr_y = _get_scroll_y()

        if curr_y <= 2:
            a, d, b, js_calls = _poll_once_and_confirm(target_keys, confirmed_keys, confirmed)
            CurrentTotalSeen = len(confirmed_keys)
            print(f"debug: [MODE=SAFE] scrollstep={step}(final), newURL={a}, dupURL={d}, batchSize={b}, jsCalls={js_calls}, yOffset={_get_scroll_y()}, TargetTotalSeen={TargetTotalSeen}, CurrentTotalSeen={CurrentTotalSeen}")
            break

        new_total, dup_total, last_batch_size, total_js_calls = _poll_until_settled(target_keys, confirmed_keys, confirmed)
        CurrentTotalSeen = len(confirmed_keys)
        print(f"debug: [MODE=SAFE] scrollstep={step}, newURL={new_total}, dupURL={dup_total}, batchSize={last_batch_size}, jsCalls={total_js_calls}, yOffset={curr_y}, TargetTotalSeen={TargetTotalSeen}, CurrentTotalSeen={CurrentTotalSeen}")

        prev_y = _get_scroll_y()
        driver.execute_script("window.scrollBy(0, arguments[0]);", -int(move_px))
        time.sleep(UP_DELAY_S)
        cur_y = _get_scroll_y()

        if cur_y >= prev_y - 1:
            top_stall_seq += 1
        else:
            top_stall_seq = 0

        if cur_y <= 2 and top_stall_seq >= 3:
            a, d, b, js_calls = _poll_once_and_confirm(target_keys, confirmed_keys, confirmed)
            CurrentTotalSeen = len(confirmed_keys)
            print(f"debug: [MODE=SAFE] scrollstep={step}(final), newURL={a}, dupURL={d}, batchSize={b}, jsCalls={js_calls}, yOffset={_get_scroll_y()}, TargetTotalSeen={TargetTotalSeen}, CurrentTotalSeen={CurrentTotalSeen}")
            break

    result: Dict[str, Dict[str, str]] = {}
    for k in target_keys:
        meta = confirmed.get(k)
        if meta:
            result[k] = meta
        else:
            result[k] = {
                "url": target_url_by_key.get(k, ""),
                "uploader_name": "",
                "upload_time": "",
            }

    elapsed = time.time() - crawl_t0
    print(f"message: SAFE upward collection finished in {elapsed:.2f} seconds")
    print(f"message: TargetTotalSeen={TargetTotalSeen}, CurrentTotalSeen={CurrentTotalSeen}, Missing={TargetTotalSeen - CurrentTotalSeen}")
    return result

def _poll_once_and_confirm(target_keys: set[str], confirmed_keys: set[str], confirmed: Dict[str, Dict[str, str]]) -> Tuple[int,int,int,int]:
    jsCalls = 0

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
            })
        except Exception:
            continue

    merged: Dict[str, Dict[str, str]] = {}
    for src in (io_items, dom_items):
        for it in src:
            u = it["url"]
            if u not in merged:
                merged[u] = it

    new_cnt = 0
    dup_cnt = 0
    for u, it in merged.items():
        mk = normalize_media_key(u)
        if not mk:
            continue
        if mk in target_keys:
            if mk not in confirmed_keys:
                confirmed_keys.add(mk)
                confirmed[mk] = {
                    "url": u,
                    "uploader_name": it.get("uploader_name", ""),
                    "upload_time": it.get("upload_time", ""),
                }
                new_cnt += 1
            else:
                dup_cnt += 1
    return new_cnt, dup_cnt, len(merged), jsCalls

def _poll_until_settled(target_keys: set[str], confirmed_keys: set[str], confirmed: Dict[str, Dict[str, str]]) -> Tuple[int,int,int,int]:
    STEP_MIN_SETTLE_S = 0.30
    idle_seq = 0
    t0 = time.time()
    new_total = 0
    dup_total = 0
    last_batch_size = 0
    js_calls = 0

    while True:
        a, d, b, c = _poll_once_and_confirm(target_keys, confirmed_keys, confirmed)
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
    safe_uploader_name = re.sub(r'[\\/*?:"<>|]', "", uploader_name)
    safe_upload_time = re.sub(r'[\\/*?:"<>|]', "", upload_time)

    mk = normalize_media_key(url) or "unknown"
    url_path = url.split("?")[0]
    _, ext = os.path.splitext(url_path)
    if not ext:
        ext = ".png"
    if safe_uploader_name or safe_upload_time:
        base = f"{safe_uploader_name}_{safe_upload_time}".strip("_")
    else:
        base = f"{mk}"
    if not base:
        base = mk

    file_path = os.path.join(out_dir, f"{base}{ext}")
    stem, suf = os.path.splitext(file_path)
    c = 1
    while os.path.exists(file_path):
        file_path = f"{stem}_{c}{suf}"
        c += 1

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
# 1) 완전 하강 (도중에 간헐적으로 CDP drain 및 cdpKeys 로그)
cdp_seen_keys: set[str] = set()
cdp_url_by_key: Dict[str, str] = {}
full_descent(cdp_seen_keys, cdp_url_by_key)

# 2) 하강 직후 CDP를 한 번 더 drain → 타겟 최종 확정
_ = update_cdp_seen_from_logs(cdp_seen_keys, cdp_url_by_key)
target_url_by_key: Dict[str, str] = dict(cdp_url_by_key)  # key -> canonical URL
target_keys = set(target_url_by_key.keys())
TargetTotalSeen = len(target_keys)
print(f"message: CDP drain after descent: TargetTotalSeen={TargetTotalSeen}")

image_entries: List[Dict[str, str]] = []

if mode == "CDP_ONLY":
    # 업스크롤 없이 즉시 다운로드
    for k in target_keys:
        image_entries.append({
            "url": target_url_by_key[k],
            "uploader_name": "",     # CDP_ONLY: 메타 불명
            "upload_time": ""
        })
else:
    # SAFE 모드: IO 부트스트랩 → 업스크롤 수집/매칭
    def bootstrap_observers_wrapper():
        bootstrap_observers()
    bootstrap_observers_wrapper()

    def safe_collect_wrapper():
        return safe_upward_collect(target_keys, target_url_by_key)

    confirmed_map = safe_collect_wrapper()

    for k in target_keys:
        image_entries.append({
            "url": confirmed_map[k]["url"],
            "uploader_name": confirmed_map[k]["uploader_name"],
            "upload_time": confirmed_map[k]["upload_time"],
        })

# 3) 다운로드
print("message: Start downloading images...")
result_file_path = os.path.join(new_folder_path, "result.txt")
write_lock = threading.Lock()
pass_count = 0
ok_count = 0

with open(result_file_path, "w", encoding="utf-8") as result_file:
    futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for idx_, d in enumerate(image_entries, start=1):
            futures.append(ex.submit(download_one, idx_, d, new_folder_path))
        for fu in tqdm(as_completed(futures), total=len(futures), desc="Downloading", unit="file"):
            ok, fname, url, err = fu.result()
            with write_lock:
                if ok:
                    ok_count += 1
                    mk = normalize_media_key(url) or ""
                    meta_flag = "" if (d.get("uploader_name") or d.get("upload_time")) else " MISSING_META"
                    result_file.write(f"{fname} , URL: {url} , KEY: {mk}{meta_flag}\n")
                else:
                    pass_count += 1
                    print(f"message: Failed to download {url}: {err}")

successful_downloads = ok_count
print(f"message: Total number of successfully downloaded images: {successful_downloads}")

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

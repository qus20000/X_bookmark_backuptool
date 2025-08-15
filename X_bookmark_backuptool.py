# X bookmark backuptool by qus20000
# Windows / Python 3.9.x ~ 3.10.x 권장
#
# -----------------------------------------------------------------------------
# Overview
# -----------------------------------------------------------------------------
# 본 스크립트는 X(트위터) 북마크 페이지에서 이미지 URL을 수집하고 이미지를 다운로드하는 도구입니다.
#
# 공통 동작:
# - 1) 완전 하강(북마크 페이지에서 스크롤 최하단까지 탐지)"방식을 통해 모든 트윗이 로드된 상태를 확보 
#   2) CDP(Network.* 성능 로그)에서 pbs.twimg.com/media/... 요청을 추출 → Target 집합 구성
#   3) 모드에 따라:
#      - CDP_ONLY: 하강 직후 바로 다운로드(업스크롤 없음). 하강 중 IO(IntersectionObserver)로
#                  수집해둔 메타(업로더/시간)가 있으면 파일명에 반영, 없으면 MEDIA_KEY로 저장.
#      - SAFE:    업스크롤 하며 DOM+IO로 Target을 하나씩 “확정”하고 메타를 최대한 채움.
#                  확정되지 않은(Target - Current) 키는 “메타 없이”라도 URL로 저장.
#
# 로그 정책:
#   - 하강 단계: 매 Burst 진행 상황과 CDP/IO 누적치 로그
#   - 업스크롤 단계: 단일 라인 포맷만 출력
#   - Save시 디버그타입 : debug: [MODE=SAFE] scrollstep=..., newURL=..., dupURL=..., batchSize=..., jsCalls=..., yOffset=..., TargetTotalSeen=..., CurrentTotalSeen=...
#   - backup 종료시 TargetTotalSeen, CurrentTotalSeen, Missing 수를 출력하고,
#                Missing 키/URL 상세를 log.txt에 기록(다운로드는 “키만”으로 진행)
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

# -----------------------------------------------------------------------------
# CONFIG: 사용자 튜닝 파라미터
# -----------------------------------------------------------------------------
CONFIG = {
    # 연결/프로필
    "DEBUGGER_ADDRESS": "127.0.0.1:9222",
    "USER_DATA_DIR": os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "User Data"),
    "PROFILE_DIR_NAME": "Default",

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

CDP_DISABLE_CACHE          = CONFIG["CDP_DISABLE_CACHE"]
HARD_RELOAD_ON_BOOKMARKS   = CONFIG["HARD_RELOAD_ON_BOOKMARKS"]
CDP_ONLY_AUTOFALLBACK      = CONFIG["CDP_ONLY_AUTOFALLBACK"]
CDP_ONLY_MIN_KEYS          = CONFIG["CDP_ONLY_MIN_KEYS"]
CDP_ONLY_MIN_RATIO_OF_PEAK = CONFIG["CDP_ONLY_MIN_RATIO_OF_PEAK"]

UP_STEP_PX           = CONFIG["UP_STEP_PX"]
UP_DELAY_S           = CONFIG["UP_DELAY_S"]
VIEWPORT_PAD         = CONFIG["VIEWPORT_PAD"]
SAFE_OVERLAP_RATIO   = CONFIG["SAFE_OVERLAP_RATIO"]

MAX_WORKERS          = CONFIG["MAX_WORKERS"]

FILENAME_MODE        = CONFIG["FILENAME_MODE"]
SKIP_IF_EXISTS       = CONFIG["SKIP_IF_EXISTS"]

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
print("  1) CDP_ONLY : Fastest. After full descent, drain CDP media URLs and download immediately.")
print("                No upward scan / No heavy DOM parse. During descent, a lightweight")
print("                IntersectionObserver collects metadata opportunistically.")
print("  2) SAFE     : Recommended. After full descent, fix CDP Target, then upward scan.")
print("                Use DOM+IntersectionObserver to enrich metadata and match Target.")
print("                Reports Target vs Current and logs Missing keys/urls.")
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
            })
        except Exception:
            continue
    return out

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
            meta_map[mk] = {"uploader_name": it.get("uploader_name", ""), "upload_time": it.get("upload_time", "")}
            new_keys += 1
        else:
            # 빈 필드는 채움
            if not prev.get("uploader_name") and it.get("uploader_name"):
                prev["uploader_name"] = it.get("uploader_name", "")
            if not prev.get("upload_time") and it.get("upload_time"):
                prev["upload_time"] = it.get("upload_time", "")
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
                 desc_meta_by_key: Dict[str, Dict[str, str]]) -> int:
     
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
            new_added = update_cdp_seen_from_logs(cdp_seen_keys, cdp_url_by_key)
            cdp_peak = max(cdp_peak, len(cdp_seen_keys))
            io_items = flush_io_buffer()
            ioNew = merge_into_meta_map(desc_meta_by_key, io_items)
            print(
                f"debug: downBurst={burst_idx}, perBurstScrolls={DOWN_SCROLL_BURST}, "
                f"yOffset={cur_y}, deltaY={delta_y}, scrollHeight={prev_h}, grewPx={grew_px}, grew={int(grew)}, "
                f"heightStallSeq={stall_cycles}/{DOWN_STALL_TOLERANCE}, yStallSeq={y_stall_seq}/{YOFFSET_STALL_BURSTS}, "
                f"cdpKeys={len(cdp_seen_keys)}, cdpNew={new_added}, ioKeys={len(desc_meta_by_key)}, ioNew={ioNew}"
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

    # 최종 drain(마지막 남은 이벤트/버퍼 수거)
    _ = update_cdp_seen_from_logs(cdp_seen_keys, cdp_url_by_key)
    io_items = flush_io_buffer()
    _ = merge_into_meta_map(desc_meta_by_key, io_items)

    print(f"message: full-descent finished. stopReason={stop_reason}, yOffset={_get_scroll_y()}, scrollHeight={_get_scroll_h()}, cdpKeys={len(cdp_seen_keys)}, ioKeys={len(desc_meta_by_key)}")
    return cdp_peak

# -----------------------------------------------------------------------------
# SAFE: 업스크롤 수집(타겟 매칭 및 메타 확보)
# -----------------------------------------------------------------------------
def _poll_once_and_confirm(target_keys: set[str], confirmed_keys: set[str], confirmed: Dict[str, Dict[str, str]]) -> Tuple[int,int,int,int]:
    # IO 버퍼+DOM 스냅샷을 병합하여 Target에 해당하는 키를 확정. 
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
    # 단계 내에서 수집이 안정될 때까지 짧게 폴링. 
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

# === MOD: SAFE 상승 전(bottom 근처) jiggle 1회 수행
def pre_upward_jiggle(cdp_seen_keys: set[str], cdp_url_by_key: Dict[str, str]) -> None:
    vh = _get_vh()
    delta = max(80, int(0.45 * vh))
    _jiggle_once(delta, 0.08)
    added = update_cdp_seen_from_logs(cdp_seen_keys, cdp_url_by_key)
    print(f"message: pre-upward jiggle done (vh={vh}, delta={delta}, cdpNew={added})")

def safe_upward_collect(target_keys: set[str], target_url_by_key: Dict[str, str]) -> Tuple[Dict[str, Dict[str, str]], set[str], List[str]]:
    # SAFE 업스크롤 수집. 반환: (확정맵, 확정키집합, Missing 키 List) 
    confirmed: Dict[str, Dict[str, str]] = {}
    confirmed_keys: set[str] = set()

    # === MOD: 상승 시작 직전에 jiggle 1회로 바닥 부근 로딩 유도
    pre_upward_jiggle(cdp_seen_keys, cdp_url_by_key)

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

    step = 0
    top_stall_seq = 0
    crawl_t0 = time.time()

    while True:
        step += 1
        curr_y = _get_scroll_y()

        if curr_y <= 2:
            a, d, b, js_calls = _poll_once_and_confirm(target_keys, confirmed_keys, confirmed)
            print(f"debug: [MODE=SAFE] scrollstep={step}(final), newURL={a}, dupURL={d}, batchSize={b}, jsCalls={js_calls}, yOffset={_get_scroll_y()}, TargetTotalSeen={TargetTotalSeen}, CurrentTotalSeen={len(confirmed_keys)}")
            break

        new_total, dup_total, last_batch_size, total_js_calls = _poll_until_settled(target_keys, confirmed_keys, confirmed)
        print(f"debug: [MODE=SAFE] scrollstep={step}, newURL={new_total}, dupURL={dup_total}, batchSize={last_batch_size}, jsCalls={total_js_calls}, yOffset={curr_y}, TargetTotalSeen={TargetTotalSeen}, CurrentTotalSeen={len(confirmed_keys)}")

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
            print(f"debug: [MODE=SAFE] scrollstep={step}(final), newURL={a}, dupURL={d}, batchSize={b}, jsCalls={js_calls}, yOffset={_get_scroll_y()}, TargetTotalSeen={TargetTotalSeen}, CurrentTotalSeen={len(confirmed_keys)}")
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
# IO/MutationObserver는 하강 시작 전에 주입(두 모드 공통으로 메타를 최대한 확보)
bootstrap_observers()

# === MOD: 하강 시작 전에 top jiggle 1회로 초기 로딩 유도
pre_descent_jiggle(set(), {})  # 초기 CDP 누계 의미 없으므로 더미로 호출, 메시지 용도

# 1) 완전 하강 (도중에 주기적으로 CDP/IO drain 및 로그)
cdp_seen_keys: set[str] = set()
cdp_url_by_key: Dict[str, str] = {}
desc_meta_by_key: Dict[str, Dict[str, str]] = {}  # 하강 중 IO로 모은 메타(키 -> 메타)
cdp_peak = full_descent(cdp_seen_keys, cdp_url_by_key, desc_meta_by_key)

# 2) 하강 직후 CDP를 한 번 더 drain -> 타겟 최종 확정
_ = update_cdp_seen_from_logs(cdp_seen_keys, cdp_url_by_key)
target_url_by_key: Dict[str, str] = dict(cdp_url_by_key)  # key -> canonical URL
target_keys = set(target_url_by_key.keys())
TargetTotalSeen = len(target_keys)
print(f"message: CDP drain after descent: TargetTotalSeen={TargetTotalSeen}, cdpPeakDuringDescent={cdp_peak}")

# 2.5) CDP_ONLY 자동 폴백 판단
if mode == "CDP_ONLY" and CDP_ONLY_AUTOFALLBACK:
    min_allowed = max(CDP_ONLY_MIN_KEYS, int(cdp_peak * CDP_ONLY_MIN_RATIO_OF_PEAK))
    if TargetTotalSeen < min_allowed:
        print(f"message: CDP_ONLY target too low (Target={TargetTotalSeen} < MinAllowed={min_allowed}). Auto-fallback to SAFE.")
        mode = "SAFE"

# 3) 수집/다운로드 엔트리 구성
image_entries: List[Dict[str, str]] = []
missing_keys: List[str] = []  # SAFE에서만 의미 있음

if mode == "CDP_ONLY":
    # 업스크롤 없이 즉시 다운로드
    # 하강 중 IO에서 모아둔 메타(desc_meta_by_key)가 있으면 반영
    for k in target_keys:
        meta = desc_meta_by_key.get(k, {"uploader_name": "", "upload_time": ""})
        image_entries.append({
            "url": target_url_by_key[k],
            "uploader_name": meta.get("uploader_name", ""),
            "upload_time": meta.get("upload_time", "")
        })
else:
    # SAFE 모드: 업스크롤 수집/매칭
    confirmed_map, confirmed_keys, missing_keys = safe_upward_collect(target_keys, target_url_by_key)
    # 엔트리 구성: 확정된 것은 confirmed_map 메타, 미싱은 하강-IO 메타로 보강(없으면 빈값)
    for k in target_keys:
        if k in confirmed_map:
            image_entries.append({
                "url": confirmed_map[k]["url"],
                "uploader_name": confirmed_map[k]["uploader_name"],
                "upload_time": confirmed_map[k]["upload_time"],
            })
        else:
            fallback_meta = desc_meta_by_key.get(k, {"uploader_name": "", "upload_time": ""})
            image_entries.append({
                "url": target_url_by_key[k],
                "uploader_name": fallback_meta.get("uploader_name", ""),
                "upload_time": fallback_meta.get("upload_time", ""),
            })

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

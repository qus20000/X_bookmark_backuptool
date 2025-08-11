# X bookmark backuptool by qus20000
# Windows / Python 3.9+
# 기능 요약:
# - 패키지 자동 설치(ensure_packages)
# - Chrome 연결 전략(우선순위):
#   1) 디버깅 포트(9222)로 이미 로그인된 크롬에 붙기
#   2) 사용자 프로필(User Data) 재사용해 새 창 띄우기
#   3) 새 세션 띄우고 로그인 페이지 + Enter 대기
# - ChromeDriver 자동 매칭: Selenium Manager → webdriver-manager 폴백
# - GPU 초기화 이슈(ANGLE) 완화 옵션
# - 북마크: 끝까지 빠르게 스크롤 ↓, 위로 올라오며(JS 배치) 정밀 수집 ↑
# - 멀티스레드 이미지 다운로드 + tqdm 진행바
# - 로깅(터미널 + 파일 동시)
# - 업 스크롤 시 dupURL=0이면 stepPx/4씩 4회 재수집 후 다음 스텝

import sys
import os
import subprocess
import importlib.util
import time
import re
import hashlib
import shutil
from typing import Dict, List, Tuple
import msvcrt  # Windows 전용 콘솔 키 입력
import threading

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

import requests
from PIL import Image
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

DEBUGGER_ADDRESS = "127.0.0.1:9222"
USER_DATA_DIR = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "User Data")
PROFILE_DIR_NAME = "Default"

DOWN_SCROLL_BURST = 60
DOWN_STEP_PX = 1800
DOWN_DELAY_S = 0.02
DOWN_BUFFER_CHECKS = 6
DOWN_BUFFER_SLEEP_S = 0.18
DOWN_STALL_TOLERANCE = 3

# stepPx를 크게 설정
UP_STEP_PX = 900
UP_DELAY_S = 0.04
VIEWPORT_PAD = 300

MAX_WORKERS = 8

folder_base_name = "images"
new_folder_path = folder_base_name
idx = 0
while os.path.exists(new_folder_path):
    new_folder_path = f"{folder_base_name}{idx}"
    idx += 1
os.mkdir(new_folder_path)

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

log_file_path = os.path.join(new_folder_path, "log.txt")
sys.stdout = Logger(log_file_path)

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

time.sleep(2)
print("message: fast scroll down to the bottom (no collection during descent)...")
prev_scroll_height = driver.execute_script("return document.body.scrollHeight")
stall_cycles = 0
while True:
    for _ in range(DOWN_SCROLL_BURST):
        driver.execute_script(f"window.scrollBy(0, {DOWN_STEP_PX});")
        time.sleep(DOWN_DELAY_S)
    grew = False
    for _ in range(DOWN_BUFFER_CHECKS):
        time.sleep(DOWN_BUFFER_SLEEP_S)
        cur_h = driver.execute_script("return document.body.scrollHeight")
        if cur_h > prev_scroll_height:
            prev_scroll_height = cur_h
            grew = True
            break
    if not grew:
        stall_cycles += 1
    else:
        stall_cycles = 0
    if stall_cycles >= DOWN_STALL_TOLERANCE:
        print("message: reached the bottom (no more height growth).")
        break


crawl_start_time = time.time()

print("message: collecting while scrolling up in small steps...")
UP_IDLE_CONFIRMS = 1
UP_TOP_STALL_CONFIRMS = 3
UP_TOP_EPS = 2

image_data: List[Dict[str, str]] = []
seen_urls = set()

def get_scrollY() -> int:
    try:
        y = driver.execute_script(
            "return Math.max(window.pageYOffset || 0, "
            "document.documentElement.scrollTop || 0, "
            "document.body.scrollTop || 0);"
        )
        return int(y) if y is not None else 0
    except Exception:
        return 0

def collect_upward_batch() -> Tuple[int, int, int]:
    batch = driver.execute_script(JS_COLLECT_SNIPPET, VIEWPORT_PAD) or []
    added = 0
    dups = 0
    for item in batch:
        url = item.get("url") or ""
        if not url:
            continue
        if url in seen_urls:
            dups += 1
            continue
        image_data.append({
            "url": url,
            "uploader_name": item.get("uploader_name") or "",
            "upload_time": item.get("upload_time") or "",
        })
        seen_urls.add(url)
        added += 1
    return added, dups, len(batch)

step = 0
top_stall_seq = 0
while True:
    step += 1
    idle_seq = 0
    local_dup = 0
    local_new = 0
    local_batch_calls = 0
    last_batch_size = 0
    while True:
        added, dups, batch_size = collect_upward_batch()
        local_batch_calls += 1
        local_dup += dups
        local_new += added
        last_batch_size = batch_size
        if added == 0:
            idle_seq += 1
        else:
            idle_seq = 0
        if idle_seq >= UP_IDLE_CONFIRMS:
            break
        time.sleep(max(UP_DELAY_S * 0.5, 0.02))

    # dupURL=0이면 stepPx/4씩 4회 재수집
    if local_dup == 0:
        quarter_step = int(UP_STEP_PX / 4)
        for _ in range(4):
            driver.execute_script("window.scrollBy(0, arguments[0]);", -quarter_step)
            time.sleep(UP_DELAY_S)
            collect_upward_batch()
        continue

    print(
        f"debug: scrollstep={step}, newURL={local_new}, dupURL={local_dup}, "
        f"batchSize={last_batch_size}, jsCalls={local_batch_calls}, "
        f"yOffset={get_scrollY()}, totalSeen={len(seen_urls)}, stepPx={UP_STEP_PX}"
    )

    prev_y = get_scrollY()
    driver.execute_script("window.scrollBy(0, arguments[0]);", -int(UP_STEP_PX))
    time.sleep(UP_DELAY_S)
    cur_y = get_scrollY()

    if cur_y >= prev_y - 1:
        top_stall_seq += 1
    else:
        top_stall_seq = 0

    if cur_y <= UP_TOP_EPS and top_stall_seq >= UP_TOP_STALL_CONFIRMS:
        _added, _dups, _batch_size = collect_upward_batch()
        print(
            f"debug: scrollstep={step}(final), newURL={_added}, dupURL={_dups}, "
            f"batchSize={_batch_size}, jsCalls=1, yOffset={get_scrollY()}, "
            f"totalSeen={len(seen_urls)}, stepPx={UP_STEP_PX}"
        )
        break
crawl_end_time = time.time()
elapsed = crawl_end_time - crawl_start_time
print(f"message: URL crawling completed in {elapsed:.2f} seconds")

print(f"message: Final total number of collected URLs: {len(image_data)}")
print("message: Start downloading images...")

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
    uploader_name = data["uploader_name"]
    upload_time = data["upload_time"]
    safe_uploader_name = re.sub(r'[\\/*?:"<>|]', "", uploader_name)
    safe_upload_time = re.sub(r'[\\/*?:"<>|]', "", upload_time)
    url_path = url.split("?")[0]
    _, ext = os.path.splitext(url_path)
    if not ext:
        ext = ".png"
    base = f"{safe_uploader_name}_{safe_upload_time}".strip("_")
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

result_file_path = os.path.join(new_folder_path, "result.txt")
write_lock = threading.Lock()
pass_count = 0
ok_count = 0

with open(result_file_path, "w", encoding="utf-8") as result_file:
    futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for idx_, d in enumerate(image_data, start=1):
            futures.append(ex.submit(download_one, idx_, d, new_folder_path))
        for fu in tqdm(as_completed(futures), total=len(futures), desc="Downloading", unit="file"):
            ok, fname, url, err = fu.result()
            with write_lock:
                if ok:
                    ok_count += 1
                    result_file.write(f"{fname} , URL : {url}\n")
                else:
                    pass_count += 1
                    print(f"\nmessage: Failed to download {url}: {err}")

successful_downloads = ok_count
print(f"\nmessage: Total number of successfully downloaded images: {successful_downloads}")

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
                    print(f"message: Duplicate files detail: {file_path}")
                    shutil.move(file_path, os.path.join(duplicate_folder, filename))
                    duplicate_count += 1
                else:
                    hash_dict[h] = file_path
            except Exception as e:
                print(f"message: error occured: {file_path}, {e}")
                continue
    try:
        deleted_files = len(os.listdir(duplicate_folder))
        shutil.rmtree(duplicate_folder)
    except Exception as e:
        print(f"message: Error deleting duplicate folder: {e}")
        deleted_files = 0
    return duplicate_count, deleted_files

dup_count, deleted_files = move_duplicate_images(new_folder_path)
print(f"\nmessage: Number of duplicate files: {dup_count}")
print(f"message: Number of deleted duplicate files: {deleted_files}")
print(f"message: Number of failed downloads: {pass_count}")
print(f"message: Number of errors: {0}")

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

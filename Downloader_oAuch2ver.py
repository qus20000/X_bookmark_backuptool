# download_from_items.py
# items.ndjson(메타 포함) -> 파일명 정책으로 다운로드 + open_by_tid 오프너 생성
#
# 파일명 예:
# @xxxxxxxx_2025-01-06T164753_asdfasdfasdf_tid_1234567890.png
#
# 오프너 생성:
# - open_by_tid.py : 드래그&드롭 + 그냥 실행하면 파일 선택창(tkinter)
# - open_by_tid.bat: .py 드롭이 안 먹는 환경 대비(배치 파일에 드롭하면 Fullsize 인자 전달)
#
# 실행:
# - python download_from_items.py
# - 기본 입력: bookmark_meta/items.ndjson
# - 출력 폴더: downloaded_images

import os
import json
import re
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import requests
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 사용자 설정
# ---------------------------------------------------------------------------
IN_PATH = os.path.join("bookmark_meta", "items.ndjson")

OUT_DIR = "downloaded_images"
MAX_WORKERS = 10
TIMEOUT = 20
SKIP_IF_EXISTS = True

# 파일명에 트윗ID 포함 태그(오프너도 이 태그로 파싱)
TID_TAG = "_tid_"

# 오프너 파일명
OPENER_PY = "open_by_tid.py"
OPENER_BAT = "open_by_tid.bat"


def ensure_out_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def slug(s: str) -> str:
    """
    파일명에 안전한 문자만 남긴다.
    """
    s = (s or "").strip()
    s = re.sub(r'[\\/*?:"<>|]+', "", s)
    s = re.sub(r"\s+", "_", s)
    return s


def guess_ext_from_url(url: str) -> str:
    """
    url 쿼리에서 format=png 같은 확장자를 우선 사용.
    없으면 path 확장자 / 기본 .jpg
    """
    if "format=" in url:
        try:
            fmt = url.split("format=")[1].split("&")[0].strip().lower()
            if fmt:
                return "." + fmt
        except Exception:
            pass

    parsed = urlparse(url)
    _, ext = os.path.splitext(parsed.path)
    if ext:
        return ext.lower()

    return ".jpg"


def build_filename(item: Dict) -> str:
    """
    @_handle_YYYY-mm-ddTHHMMSS_mediaKey_tid_<tweetid>.<ext>
    - author는 "@rei_vrc" 또는 "@uid_..." 형태로 들어온다고 가정
    - author가 "@rei_vrc"면 "@_rei_vrc"로 맞춰서 기존 스타일 유지
    """
    author = item.get("author") or "@unknown"
    created_norm = item.get("created_at_norm") or ""
    media_key = item.get("media_key") or "unknown"
    tweet_id = item.get("tweet_id") or "0"
    url = item.get("url") or ""

    ext = guess_ext_from_url(url)

    author_fmt = author
    if author.startswith("@") and not author.startswith("@_"):
        author_fmt = "@_" + author[1:]

    name = f"{author_fmt}_{created_norm}_{media_key}{TID_TAG}{tweet_id}{ext}"
    return slug(name)


def read_items(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise RuntimeError(f"input file not found: {path}")

    items: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def download_one(item: Dict) -> Tuple[bool, str, str]:
    url = item.get("url") or ""
    if not url:
        return False, "", "missing url"

    filename = build_filename(item)
    save_path = os.path.join(OUT_DIR, filename)

    if SKIP_IF_EXISTS and os.path.exists(save_path):
        return True, filename, "skip_exists"

    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(r.content)
        return True, filename, "ok"
    except Exception as e:
        return False, filename, str(e)


def write_opener_py(path: str) -> None:
    """
    - 드래그&드롭(인자 전달) 지원
    - 그냥 실행하면 파일 선택창(tkinter) 띄움
    - 파일명에서 _tid_<digits> 추출 후 https://x.com/i/web/status/<digits> 오픈
    """
    content = f"""\
import sys
import re
import webbrowser

TID_TAG = {TID_TAG!r}

def extract_tid(path: str) -> str | None:
    m = re.search(re.escape(TID_TAG) + r"(\\d+)", path)
    if not m:
        return None
    return m.group(1)

def pick_file_gui() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    path = filedialog.askopenfilename(
        title="Select a file (filename must include _tid_<id>)",
        filetypes=[
            ("All files", "*.*"),
            ("Images", "*.png;*.jpg;*.jpeg;*.webp;*.gif;*.bmp"),
        ],
    )
    root.destroy()

    if not path:
        return None
    return path

def main():
    if len(sys.argv) >= 2:
        path = sys.argv[1]
    else:
        path = pick_file_gui()
        if not path:
            print("message: no file selected. exit.")
            return

    tid = extract_tid(path)
    if not tid:
        print("message: tid not found in filename:", path)
        print(f"message: expected pattern: {TID_TAG}<digits>")
        return

    url = f"https://x.com/i/web/status/{{tid}}"
    print("message: open:", url)
    webbrowser.open(url)

if __name__ == "__main__":
    main()
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def write_opener_bat(path: str) -> None:
    """
    .py 드롭이 안 먹는 환경 대비.
    이미지 파일을 이 .bat 위로 드롭하면 100% 인자로 전달된다.
    """
    content = f"""@echo off
setlocal

set SCRIPT_DIR=%~dp0
python "%SCRIPT_DIR%{OPENER_PY}" "%~1"

endlocal
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    ensure_out_dir()

    opener_py_path = os.path.join(OUT_DIR, OPENER_PY)
    opener_bat_path = os.path.join(OUT_DIR, OPENER_BAT)
    write_opener_py(opener_py_path)
    write_opener_bat(opener_bat_path)

    items = read_items(IN_PATH)
    print(f"message: items_loaded={len(items)}")
    print(f"message: opener_created={opener_py_path}")
    print(f"message: opener_created={opener_bat_path}")

    ok = 0
    fail = 0
    skipped = 0

    # 결과 로그(나중에 검증/재다운용)
    result_path = os.path.join(OUT_DIR, "download_result.txt")
    lock = None  # 단일 스레드 write만 해서 락 생략

    with open(result_path, "w", encoding="utf-8") as rf:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(download_one, it) for it in items]

            with tqdm(total=len(futures), desc="Downloading", unit="file") as pbar:
                for fu in as_completed(futures):
                    success, fname, status = fu.result()
                    if success:
                        ok += 1
                        if status == "skip_exists":
                            skipped += 1
                        rf.write(f"OK  file={fname} status={status}\n")
                    else:
                        fail += 1
                        rf.write(f"FAIL file={fname} err={status}\n")
                    pbar.update(1)

        rf.flush()

    print("--------------------------------------------------")
    print(f"message: ok={ok} skipped={skipped} fail={fail}")
    print(f"message: out_dir={OUT_DIR}")
    print(f"message: result_log={result_path}")
    print(f"message: drag an image onto {OPENER_PY} or {OPENER_BAT} to open tweet page")


if __name__ == "__main__":
    main()

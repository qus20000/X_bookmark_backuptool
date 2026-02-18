import sys
import re
import webbrowser

TID_TAG = '_tid_'

def extract_tid(path: str) -> str | None:
    m = re.search(re.escape(TID_TAG) + r"(\d+)", path)
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
        print(f"message: expected pattern: _tid_<digits>")
        return

    url = f"https://x.com/i/web/status/{tid}"
    print("message: open:", url)
    webbrowser.open(url)

if __name__ == "__main__":
    main()

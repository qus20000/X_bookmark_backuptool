import base64
import hashlib
import json
import secrets
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests


# ---------------------------------------------------------------------------
# 사용자 설정
# ---------------------------------------------------------------------------
CLIENT_ID = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # 여기에 너의 OAuth2 Client ID 넣기
REDIRECT_URI = "http://127.0.0.1:8080/callback"

# 북마크 조회에 필요한 최소 스코프
# offline.access를 포함하면 refresh_token이 내려와서 만료 후 재발급 자동화가 가능해짐.
SCOPES = ["tweet.read", "users.read", "bookmark.read", "offline.access"]

TOKEN_SAVE_PATH = "x_oauth_token.json"

AUTH_URL = "https://x.com/i/oauth2/authorize"
TOKEN_URL = "https://api.x.com/2/oauth2/token"


# ---------------------------------------------------------------------------
# PKCE 유틸
# ---------------------------------------------------------------------------
def b64url_no_pad(raw: bytes) -> str:
    # base64url 인코딩 후 패딩("=") 제거
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def make_code_verifier() -> str:
    # RFC 7636: code_verifier는 충분히 랜덤/긴 문자열이 필요
    return b64url_no_pad(secrets.token_bytes(32))


def make_code_challenge(verifier: str) -> str:
    # S256: BASE64URL(SHA256(verifier))
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return b64url_no_pad(digest)


# ---------------------------------------------------------------------------
# 로컬 콜백 서버: redirect로 들어오는 ?code=... 수신
# ---------------------------------------------------------------------------
class CallbackHandler(BaseHTTPRequestHandler):
    oauth_code = None
    oauth_state = None
    oauth_error = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)

        if "error" in qs:
            CallbackHandler.oauth_error = qs.get("error", [""])[0]

        if "state" in qs:
            CallbackHandler.oauth_state = qs.get("state", [""])[0]

        if "code" in qs:
            CallbackHandler.oauth_code = qs.get("code", [""])[0]

        # 브라우저에 완료 안내 표시
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(
            b"<html><body><h3>OK</h3><p>You can close this tab and return to the terminal.</p></body></html>"
        )

    def log_message(self, format, *args):
        # http.server 기본 로그 제거(터미널 지저분해지는 것 방지)
        return


def run_local_server(host: str, port: int) -> HTTPServer:
    httpd = HTTPServer((host, port), CallbackHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd


# ---------------------------------------------------------------------------
# OAuth URL 생성
# ---------------------------------------------------------------------------
def build_authorize_url(client_id: str, redirect_uri: str, scopes: list[str], state: str, code_challenge: str) -> str:
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": " ".join(scopes),
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    return AUTH_URL + "?" + urllib.parse.urlencode(params)


def exchange_code_for_token(client_id: str, redirect_uri: str, code: str, code_verifier: str) -> dict:
    # X 문서 기준: x-www-form-urlencoded로 POST
    data = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code": code,
        "code_verifier": code_verifier,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    r = requests.post(TOKEN_URL, data=data, headers=headers, timeout=30)
    if r.status_code >= 400:
        # 토큰 교환 실패 시 원인 파악을 위해 본문 출력
        print("HTTP", r.status_code)
        print(r.text)
        r.raise_for_status()
    return r.json()


def main():
    if "여기에_CLIENT_ID" in CLIENT_ID:
        raise RuntimeError("CLIENT_ID를 먼저 넣어야 합니다.")

    # redirect_uri에서 host/port 파싱
    u = urllib.parse.urlparse(REDIRECT_URI)
    host = u.hostname or "127.0.0.1"
    port = u.port or 8080

    httpd = run_local_server(host, port)

    code_verifier = make_code_verifier()
    code_challenge = make_code_challenge(code_verifier)

    state = b64url_no_pad(secrets.token_bytes(16))

    url = build_authorize_url(CLIENT_ID, REDIRECT_URI, SCOPES, state, code_challenge)
    print("Open this URL in your browser (if it doesn't open automatically):")
    print(url)

    try:
        webbrowser.open(url)
    except Exception:
        pass

    print("\nWaiting for redirect with authorization code...")
    while True:
        if CallbackHandler.oauth_error:
            httpd.shutdown()
            raise RuntimeError(f"OAuth error: {CallbackHandler.oauth_error}")

        if CallbackHandler.oauth_code:
            if CallbackHandler.oauth_state != state:
                httpd.shutdown()
                raise RuntimeError("State mismatch (possible CSRF). Aborting.")
            code = CallbackHandler.oauth_code
            break

    httpd.shutdown()

    token = exchange_code_for_token(CLIENT_ID, REDIRECT_URI, code, code_verifier)

    with open(TOKEN_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(token, f, ensure_ascii=False, indent=2)

    print("\nSaved:", TOKEN_SAVE_PATH)
    print("keys:", ", ".join(token.keys()))
    print("access_token:", "OK" if token.get("access_token") else "MISSING")
    print("refresh_token:", "OK" if token.get("refresh_token") else "MISSING")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a static Netlify publish directory for the Scalpel marketing site.

Stdlib-only (no FastAPI/torch) so Netlify's Python build can succeed.

The FastAPI API (signup DB, jobs, console) must run separately with
DATABASE_URL=postgres://… . Netlify hosts the site and proxies /v1/* and /app
to that API (see netlify.toml).

Usage:
  SCALPEL_API_ORIGIN=https://api.yourdomain.com python scripts/build_netlify.py
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "scalpel" / "api" / "static"
OUT = ROOT / "netlify-dist"

API_ORIGIN = (
    os.environ.get("SCALPEL_API_ORIGIN")
    or os.environ.get("SCALPEL_PUBLIC_API_URL")
    or ""
).rstrip("/")

_NAV_LINKS = (
    ("product", "/product", "Product"),
    ("developers", "/developers", "Developers"),
    ("security", "/security", "Security"),
)

_FOOTER = """<footer class="site-footer">
  <div class="foot-grid">
    <div class="foot-brand">
      <div class="nav-brand"><span></span>Scalpel</div>
      <p>Bias surgery for adtech teams that ship creative models under brand audit.</p>
    </div>
    <div>
      <h4>Product</h4>
      <a href="/product">How it works</a>
      <a href="/developers">Developers</a>
      <a href="/security">Security</a>
    </div>
    <div>
      <h4>Platform</h4>
      <a href="/app">Console</a>
      <a href="/docs">API reference</a>
      <a href="/signup">Create workspace</a>
      <a href="/login">Log in</a>
    </div>
    <div>
      <h4>Company</h4>
      <a href="mailto:sales@scalpel.ai">sales@scalpel.ai</a>
      <a href="/privacy">Privacy</a>
      <a href="/terms">Terms</a>
    </div>
  </div>
  <div class="foot-base">
    <span>© Scalpel AI</span>
    <span>Bias ops · rank-one edits · compliance artifacts</span>
  </div>
</footer>"""


def _version() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    return m.group(1) if m else "0.0.0"


def _nav(active: str) -> str:
    links = []
    for key, href, label in _NAV_LINKS:
        cls = ' class="active"' if key == active else ""
        links.append(f'<a href="{href}"{cls}>{label}</a>')
    mid = "\n        ".join(links)
    return f"""<header class="site-nav">
  <a class="nav-brand" href="/"><span></span>Scalpel</a>
  <nav class="nav-mid">
        {mid}
  </nav>
  <div class="nav-actions" data-auth-slot>
    <a class="btn btn-ghost" href="/app">Console</a>
    <a class="btn btn-line" href="/login">Log in</a>
    <a class="btn btn-solid" href="/signup">Get started</a>
  </div>
</header>"""


def _render(
    name: str,
    *,
    title: str,
    description: str,
    active: str = "",
    api_base: str = "",
) -> str:
    html = (STATIC / name).read_text()
    css_v = int((STATIC / "site.css").stat().st_mtime)
    session_v = int((STATIC / "session.js").stat().st_mtime)
    live = STATIC / "live-demo.js"
    live_v = int(live.stat().st_mtime) if live.exists() else 0
    og = f"""<title>{title}</title>
<meta name="description" content="{description}">
<meta property="og:type" content="website">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{description}">
<meta property="og:image" content="/static/og.svg">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{title}">
<meta name="twitter:description" content="{description}">
<meta name="twitter:image" content="/static/og.svg">
<link rel="icon" href="/static/favicon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/static/favicon.svg">
<link rel="stylesheet" href="/static/site.css?v={css_v}">"""
    api_boot = (
        f"<script>window.SCALPEL_API_BASE={api_base!r};</script>\n"
        f'<script src="/static/session.js?v={session_v}"></script>\n'
        f'<script src="/static/live-demo.js?v={live_v}"></script>'
    )
    html = html.replace("__HEAD_META__", og)
    html = html.replace("__NAV__", _nav(active))
    html = html.replace("__FOOTER__", _FOOTER)
    html = html.replace("</body>", f"{api_boot}\n</body>")
    html = html.replace("__SCALPEL_VERSION__", _version())
    html = html.replace("__API_BASE__", api_base)
    return html


def main() -> int:
    if not STATIC.is_dir():
        print(f"ERROR: static dir missing at {STATIC}", file=sys.stderr)
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)
    out_static = OUT / "static"
    out_static.mkdir(parents=True)

    # Browser calls stay same-origin; Netlify proxies /v1 and /app to the API.
    api_base = ""

    pages = {
        "index.html": (
            "landing.html",
            "Scalpel — Bias surgery for adtech models",
            "For adtech and brand-safety teams: measure creative association "
            "bias in open CLIP, cut it from weights you own, ship a PDF brands accept.",
            "",
        ),
        "product.html": (
            "product.html",
            "Scalpel — For adtech & brand safety",
            "Surgical bias editing for ML and Trust leads who run open creative models.",
            "product",
        ),
        "developers.html": (
            "developers.html",
            "Scalpel — Developers",
            "API quickstart for edit jobs and audit reports.",
            "developers",
        ),
        "security.html": (
            "security.html",
            "Scalpel — Security",
            "Tenant isolation and compliance artifacts.",
            "security",
        ),
        "privacy.html": (
            "privacy.html",
            "Scalpel — Privacy",
            "How Scalpel handles account data.",
            "",
        ),
        "terms.html": (
            "terms.html",
            "Scalpel — Terms",
            "Terms of service.",
            "",
        ),
    }

    for out_name, (src, title, desc, active) in pages.items():
        (OUT / out_name).write_text(
            _render(src, title=title, description=desc, active=active, api_base=api_base)
        )

    auth_src = (STATIC / "auth.html").read_text()
    for mode, title in (("login", "Log in"), ("signup", "Sign up")):
        html = (
            auth_src.replace("__AUTH_MODE__", mode)
            .replace("__AUTH_TITLE__", title)
            .replace("__API_BASE__", api_base)
        )
        (OUT / f"{mode}.html").write_text(html)

    for name in ("site.css", "session.js", "live-demo.js", "favicon.svg", "og.svg"):
        src = STATIC / name
        if src.exists():
            shutil.copy2(src, out_static / name)

    # Console fallback if /app proxy is misconfigured (static shell only).
    console = STATIC / "index.html"
    if console.exists():
        console_html = console.read_text().replace("__SCALPEL_VERSION__", _version())
        boot = "<script>window.SCALPEL_API_BASE='';</script>\n"
        if "SCALPEL_API_BASE" not in console_html:
            console_html = console_html.replace("</head>", f"{boot}</head>", 1)
        (OUT / "app.html").write_text(console_html)

    api = API_ORIGIN
    redirect_lines = [
        "# Pretty marketing paths",
        "/product       /product.html       200",
        "/developers    /developers.html    200",
        "/security      /security.html      200",
        "/privacy       /privacy.html       200",
        "/terms         /terms.html         200",
        "/login         /login.html         200",
        "/signup        /signup.html        200",
        "",
    ]
    if api:
        redirect_lines.extend(
            [
                "# Proxy API + console + OpenAPI to the Scalpel Python service",
                f"/v1/*          {api}/v1/:splat     200",
                f"/r/*           {api}/r/:splat      200",
                f"/app           {api}/app           200",
                f"/app/*         {api}/app           200",
                f"/docs          {api}/docs          200",
                f"/docs/*        {api}/docs/:splat   200",
                f"/redoc         {api}/redoc         200",
                f"/openapi.json  {api}/openapi.json  200",
                f"/health        {api}/health        200",
                f"/ready         {api}/ready         200",
                "",
            ]
        )
    else:
        # Static console shell only — OpenAPI / live jobs need SCALPEL_API_ORIGIN.
        redirect_lines.append("/app           /app.html           200")
        redirect_lines.append("")
    (OUT / "_redirects").write_text("\n".join(redirect_lines))

    index = OUT / "index.html"
    if not index.is_file() or index.stat().st_size < 100:
        print("ERROR: index.html was not written — publish would 404", file=sys.stderr)
        return 1

    print(f"Wrote {OUT} (API origin for proxies: {api or 'UNSET'})")
    print(f"index.html bytes={index.stat().st_size}")
    if not api:
        print(
            "WARNING: Set SCALPEL_API_ORIGIN in Netlify env to your FastAPI host "
            "(Postgres DATABASE_URL must be set on that host). "
            "Without it, /docs, live surgery, signup, and the live /app console "
            "cannot reach the API (static marketing pages still work)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

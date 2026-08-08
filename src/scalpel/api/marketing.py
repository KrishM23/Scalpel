"""Shared chrome and helpers for public marketing pages."""

from __future__ import annotations

from pathlib import Path

from fastapi.responses import HTMLResponse

import scalpel

_STATIC = Path(__file__).parent / "static"

_NAV_LINKS = (
    ("product", "/product", "Product"),
    ("developers", "/developers", "Developers"),
    ("security", "/security", "Security"),
)


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


_FOOTER = """<footer class="site-footer">
  <div class="foot-grid">
    <div class="foot-brand">
      <div class="nav-brand"><span></span>Scalpel</div>
      <p>Surgical model editing for teams that ship under audit.</p>
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


def render_marketing_page(
    name: str,
    *,
    title: str,
    description: str,
    active: str = "",
    replacements: dict[str, str] | None = None,
) -> HTMLResponse:
    html = (_STATIC / name).read_text()
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
<link rel="stylesheet" href="/static/site.css?v={int((_STATIC / 'site.css').stat().st_mtime)}">"""
    import os

    reps = dict(replacements or {})
    api_base = reps.pop(
        "__API_BASE__",
        (
            os.environ.get("SCALPEL_PUBLIC_API_URL")
            or os.environ.get("PUBLIC_API_URL")
            or ""
        ).rstrip("/"),
    )
    session_v = int((_STATIC / "session.js").stat().st_mtime)
    live_demo = _STATIC / "live-demo.js"
    live_v = int(live_demo.stat().st_mtime) if live_demo.exists() else 0
    api_boot = (
        f"<script>window.SCALPEL_API_BASE={api_base!r};</script>\n"
        f'<script src="/static/session.js?v={session_v}"></script>\n'
        f'<script src="/static/live-demo.js?v={live_v}"></script>'
    )
    html = html.replace("__HEAD_META__", og)
    html = html.replace("__NAV__", _nav(active))
    html = html.replace("__FOOTER__", _FOOTER)
    if "__SESSION_SCRIPT__" in html:
        html = html.replace("__SESSION_SCRIPT__", api_boot)
    else:
        html = html.replace("</body>", f"{api_boot}\n</body>")
    html = html.replace("__SCALPEL_VERSION__", scalpel.__version__)
    html = html.replace("__API_BASE__", api_base)
    for key, value in reps.items():
        html = html.replace(key, value)
    return HTMLResponse(
        html,
        headers={"Cache-Control": "no-cache"},
    )

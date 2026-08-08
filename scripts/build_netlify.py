#!/usr/bin/env python3
"""Build a static Netlify publish directory for the Scalpel marketing site.

The FastAPI API (signup DB, jobs, console) must run separately with
DATABASE_URL=postgres://… so accounts persist. Netlify hosts the site and
proxies /v1/* and /app to that API (see netlify.toml).

Usage:
  SCALPEL_API_ORIGIN=https://api.yourdomain.com python scripts/build_netlify.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "scalpel" / "api" / "static"
OUT = ROOT / "netlify-dist"

# Origin of the Python API (no trailing slash). Empty = same-origin proxy.
API_ORIGIN = (
    os.environ.get("SCALPEL_API_ORIGIN")
    or os.environ.get("SCALPEL_PUBLIC_API_URL")
    or ""
).rstrip("/")


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from scalpel.api.marketing import render_marketing_page

    if OUT.exists():
        shutil.rmtree(OUT)
    out_static = OUT / "static"
    out_static.mkdir(parents=True)

    pages = {
        "index.html": ("landing.html", "Scalpel — Surgical model editing",
                       "Locate latent bias circuits. Erase them with calibrated rank-one edits.",
                       ""),
        "product.html": ("product.html", "Scalpel — Product",
                         "Measure, locate, cut, and prove.", "product"),
        "developers.html": ("developers.html", "Scalpel — Developers",
                            "API quickstart for edit jobs and audit reports.", "developers"),
        "security.html": ("security.html", "Scalpel — Security",
                          "Tenant isolation and compliance artifacts.", "security"),
        "privacy.html": ("privacy.html", "Scalpel — Privacy",
                         "How Scalpel handles account data.", ""),
        "terms.html": ("terms.html", "Scalpel — Terms",
                       "Terms of service.", ""),
    }

    # Browser calls stay same-origin; Netlify proxies /v1 and /app to the API.
    api_base = ""
    os.environ["SCALPEL_PUBLIC_API_URL"] = api_base

    for out_name, (src, title, desc, active) in pages.items():
        resp = render_marketing_page(
            src, title=title, description=desc, active=active,
            replacements={"__API_BASE__": api_base},
        )
        (OUT / out_name).write_text(resp.body.decode("utf-8"))

    auth_src = (STATIC / "auth.html").read_text()
    for mode, title in (("login", "Log in"), ("signup", "Sign up")):
        html = (
            auth_src.replace("__AUTH_MODE__", mode)
            .replace("__AUTH_TITLE__", title)
            .replace("__API_BASE__", api_base)
        )
        (OUT / f"{mode}.html").write_text(html)

    for name in (
        "site.css",
        "session.js",
        "live-demo.js",
        "favicon.svg",
        "og.svg",
        "index.html",  # console also copied for /app fallback if not proxied
    ):
        src = STATIC / name
        if src.exists():
            if name == "index.html":
                # Console stays on the API host via redirect; keep a copy as app.html
                text = src.read_text().replace("__SCALPEL_VERSION__", "netlify")
                (OUT / "app.html").write_text(text)
            else:
                shutil.copy2(src, out_static / name)

    # Pretty-URL fallbacks as duplicate paths for file-based hosts.
    redirects = OUT / "_redirects"
    api = API_ORIGIN or "https://YOUR_API_HOST"
    redirects.write_text(
        "\n".join(
            [
                "# Pretty marketing paths",
                "/product       /product.html       200",
                "/developers    /developers.html    200",
                "/security      /security.html      200",
                "/privacy       /privacy.html       200",
                "/terms         /terms.html         200",
                "/login         /login.html         200",
                "/signup        /signup.html        200",
                "",
                "# Proxy API + console to the Scalpel Python service",
                f"/v1/*          {api}/v1/:splat     200",
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
    )

    print(f"Wrote {OUT} (API origin for proxies: {api})")
    if "YOUR_API_HOST" in api:
        print(
            "WARNING: Set SCALPEL_API_ORIGIN to your FastAPI host before deploying "
            "(Postgres DATABASE_URL must be set on that host)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

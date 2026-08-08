/** Shared browser session for marketing + auth (localStorage API key). */
(function () {
  const KEYS = ["scalpel_key", "scalpel_user", "scalpel_tenant", "scalpel_plan"];

  function getSession() {
    const apiKey = (localStorage.getItem("scalpel_key") || "").trim();
    if (!apiKey) return null;
    return {
      apiKey,
      name: (localStorage.getItem("scalpel_user") || "").trim(),
      tenant: (localStorage.getItem("scalpel_tenant") || "").trim(),
      plan: (localStorage.getItem("scalpel_plan") || "").trim(),
    };
  }

  function clearSession() {
    KEYS.forEach((k) => localStorage.removeItem(k));
  }

  function signOut(redirect) {
    clearSession();
    location.href = redirect || "/";
  }

  function label(session) {
    return session.name || session.tenant || "Workspace";
  }

  function hydrateNav() {
    const actions = document.querySelector(".nav-actions");
    if (!actions) return;
    const session = getSession();
    if (!session) {
      actions.innerHTML = [
        '<a class="btn btn-ghost" href="/app">Console</a>',
        '<a class="btn btn-line" href="/login">Log in</a>',
        '<a class="btn btn-solid" href="/signup">Get started</a>',
      ].join("");
      return;
    }
    const who = label(session).replace(/[<>&"]/g, "");
    actions.innerHTML = [
      `<span class="nav-user" title="${who}">${who}</span>`,
      '<a class="btn btn-solid" href="/app">Console</a>',
      '<button type="button" class="btn btn-line" id="navSignOut">Sign out</button>',
    ].join("");
    document.getElementById("navSignOut")?.addEventListener("click", () => signOut("/"));
  }

  function hydrateFooter() {
    const session = getSession();
    if (!session) return;
    const foot = document.querySelector(".site-footer");
    if (!foot) return;
    foot.querySelectorAll("a").forEach((a) => {
      const href = a.getAttribute("href");
      if (href === "/login") {
        a.textContent = "Sign out";
        a.setAttribute("href", "#signout");
        a.addEventListener("click", (e) => {
          e.preventDefault();
          signOut("/");
        });
      }
      if (href === "/signup") {
        a.textContent = "Open console";
        a.setAttribute("href", "/app");
      }
    });
  }

  window.ScalpelSession = { getSession, clearSession, signOut, hydrateNav, hydrateFooter };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      hydrateNav();
      hydrateFooter();
    });
  } else {
    hydrateNav();
    hydrateFooter();
  }
})();

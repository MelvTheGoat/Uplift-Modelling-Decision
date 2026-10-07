/**
 * Router, data loading and chrome.
 *
 * Hash routing rather than path routing, deliberately. Path routing needs the
 * server to rewrite unknown paths back to index.html, which GitHub Pages will
 * not do — a visitor who reloads on /money gets a 404. Hashes work on any
 * static host with no configuration at all, which is the whole point of
 * building the site this way.
 */

import { el, replaceChildren } from "./dom.js";
import { GLOSSARY } from "./copy.js";

const PAGES = [
  { id: "", title: "Start here", module: () => import("./pages/overview.js") },
  {
    id: "analyse",
    title: "Analyse your own campaign",
    module: () => import("./pages/analyse.js"),
    tool: true,
  },
  { id: "did-it-work", title: "Did the email work?", module: () => import("./pages/didItWork.js") },
  { id: "targeting", title: "Can we pick who to email?", module: () => import("./pages/targeting.js") },
  { id: "money", title: "What is it worth?", module: () => import("./pages/money.js") },
  { id: "trust", title: "Can we trust it?", module: () => import("./pages/trust.js") },
  { id: "sleeping-dogs", title: "Who should we leave alone?", module: () => import("./pages/dogs.js") },
  { id: "plan-a-test", title: "Plan your own test", module: () => import("./pages/planTest.js") },
  { id: "about", title: "How this was built", module: () => import("./pages/about.js") },
];

/** The study bundle, fetched once. */
let data = null;

/** Fetch `data.json`, which carries every number on the site. */
async function loadData() {
  const response = await fetch("data.json", { cache: "no-cache" });
  if (!response.ok) {
    throw new Error(
      `data.json could not be loaded (HTTP ${response.status}). ` +
        "Regenerate it with `python scripts/build_web_data.py`.",
    );
  }
  return response.json();
}

/** Build the sidebar links. */
function renderNav(activeId) {
  const nav = document.getElementById("nav");
  // The tool sits outside the numbered walkthrough. Numbering it as a step
  // would imply it comes after reading the study, when it is the thing most
  // visitors came for and should be reachable without reading anything.
  let step = 0;
  replaceChildren(
    nav,
    PAGES.map((page) => {
      let marker;
      if (page.tool) marker = "★";
      else if (page.id === "") marker = "—";
      else {
        step += 1;
        marker = String(step);
      }
      return el(
        "a",
        {
          href: `#/${page.id}`,
          class: page.tool ? "nav-tool" : null,
          "aria-current": page.id === activeId ? "page" : null,
        },
        [el("span", { class: "nav-step", text: marker }), page.title],
      );
    }),
  );
}

/** Build the sidebar glossary once. */
function renderGlossary() {
  const list = document.getElementById("glossary");
  const children = [];
  for (const [term, definition] of Object.entries(GLOSSARY)) {
    children.push(el("dt", { text: term }));
    children.push(el("dd", { html: definition }));
  }
  replaceChildren(list, children);
}

/** Read the page id out of the hash. */
function currentId() {
  const hash = window.location.hash.replace(/^#\/?/, "").replace(/\/$/, "");
  return PAGES.some((page) => page.id === hash) ? hash : "";
}

/** Load and render the page the hash names. */
async function renderRoute() {
  const id = currentId();
  const page = PAGES.find((entry) => entry.id === id) ?? PAGES[0];
  const view = document.getElementById("view");

  renderNav(id);
  document.title = id === "" ? "Who should get the email?" : `${page.title} · Who should get the email?`;

  try {
    const module = await page.module();
    replaceChildren(view, module.render(data));
  } catch (error) {
    replaceChildren(view, [
      el("div", { class: "error-box" }, [
        el("h2", { text: "That page could not be rendered" }),
        el("p", { text: String(error && error.message ? error.message : error) }),
        el("p", {
          text:
            "This is a bug rather than something you did. The other pages should still work.",
        }),
      ]),
    ]);
    // Keep the real stack in the console; the box above is for the visitor.
    console.error(error);
  }

  document.getElementById("main").scrollTo({ top: 0 });
  window.scrollTo({ top: 0 });
  closeNav();
}

// --------------------------------------------------------------- chrome

/** Close the slide-over navigation and remove its backdrop. */
function closeNav() {
  const sidebar = document.getElementById("sidebar");
  const toggle = document.getElementById("nav-toggle");
  sidebar.dataset.open = "false";
  toggle.setAttribute("aria-expanded", "false");
  document.querySelector(".scrim")?.remove();
}

/** Wire the hamburger, the backdrop and Escape. */
function wireNavToggle() {
  const sidebar = document.getElementById("sidebar");
  const toggle = document.getElementById("nav-toggle");

  toggle.addEventListener("click", () => {
    const open = sidebar.dataset.open === "true";
    if (open) {
      closeNav();
      return;
    }
    sidebar.dataset.open = "true";
    toggle.setAttribute("aria-expanded", "true");
    const scrim = el("div", { class: "scrim", onclick: closeNav });
    document.body.append(scrim);
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeNav();
  });
}

/**
 * Wire the theme toggle.
 *
 * Three states, which is the part people get wrong: an explicit light choice,
 * an explicit dark choice, and no choice at all — where the OS setting should
 * win. Removing the attribute is what hands control back to the media query,
 * so the toggle cycles through all three rather than flipping between two.
 */
function wireThemeToggle() {
  const button = document.getElementById("theme-toggle");

  const label = () => {
    const explicit = document.documentElement.dataset.theme;
    if (explicit === "dark") return "Theme: dark";
    if (explicit === "light") return "Theme: light";
    return "Theme: follow system";
  };

  button.textContent = label();
  button.addEventListener("click", () => {
    const explicit = document.documentElement.dataset.theme;
    const next = explicit === "light" ? "dark" : explicit === "dark" ? null : "light";

    if (next) document.documentElement.dataset.theme = next;
    else document.documentElement.removeAttribute("data-theme");

    try {
      if (next) localStorage.setItem("theme", next);
      else localStorage.removeItem("theme");
    } catch (error) {
      // Private browsing or blocked storage: the choice just will not persist.
    }

    button.textContent = label();
    // The charts read their colours from CSS at draw time, so they have to be
    // redrawn rather than restyled.
    renderRoute();
  });
}

// --------------------------------------------------------------- boot

async function main() {
  renderGlossary();
  wireNavToggle();
  wireThemeToggle();

  try {
    data = await loadData();
  } catch (error) {
    replaceChildren(document.getElementById("view"), [
      el("div", { class: "error-box" }, [
        el("h2", { text: "The findings could not be loaded" }),
        el("p", { text: String(error.message ?? error) }),
        el("p", {
          html:
            "If you are running this locally, note that ES modules and " +
            "<code>fetch</code> both need a real web server — opening " +
            "<code>index.html</code> straight off disk will not work. " +
            "From the repository root: <code>python -m http.server -d web 8000</code>",
        }),
      ]),
    ]);
    return;
  }

  window.addEventListener("hashchange", renderRoute);
  await renderRoute();
}

main();

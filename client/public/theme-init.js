// Applies the saved theme before the app loads, so a dark-mode user doesn't
// see a flash of light page. Same storage key and rules as src/lib/theme.ts.
// A file rather than an inline script so a strict CSP (script-src 'self')
// still allows it.
(() => {
  let theme = "system";
  try {
    theme = localStorage.getItem("warden-theme") || "system";
  } catch {
    // Storage blocked: follow the system.
  }
  const dark =
    theme === "dark" ||
    (theme !== "light" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.classList.toggle("dark", dark);
  document.documentElement.style.colorScheme = dark ? "dark" : "light";
})();

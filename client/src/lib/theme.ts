import { createContext } from "react";

export type Theme = "light" | "dark" | "system";
export type ResolvedTheme = "light" | "dark";

// Also read by public/theme-init.js, which applies the theme before React
// loads. Change both together.
const STORAGE_KEY = "warden-theme";
export const DARK_QUERY = "(prefers-color-scheme: dark)";

const isTheme = (value: unknown): value is Theme =>
  value === "light" || value === "dark" || value === "system";

// Storage can throw (private mode, blocked site data); fall back to system.
export function readStoredTheme(): Theme {
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    return isTheme(value) ? value : "system";
  } catch {
    return "system";
  }
}

export function storeTheme(theme: Theme) {
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // The choice still applies for this visit.
  }
}

export const isThemeStorageKey = (key: string | null) => key === STORAGE_KEY;

export function systemTheme(): ResolvedTheme {
  return window.matchMedia(DARK_QUERY).matches ? "dark" : "light";
}

// Tailwind's `dark:` variant and the .dark variables in index.css key off this
// class on <html>. color-scheme makes native scrollbars and form controls match.
export function applyTheme(theme: ResolvedTheme) {
  const root = document.documentElement;
  root.classList.toggle("dark", theme === "dark");
  root.style.colorScheme = theme;
}

export type ThemeContextValue = {
  // What the user picked.
  theme: Theme;
  // What's on screen: "system" resolved to light or dark.
  resolvedTheme: ResolvedTheme;
  setTheme: (theme: Theme) => void;
};

export const ThemeContext = createContext<ThemeContextValue | null>(null);

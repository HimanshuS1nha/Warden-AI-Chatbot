import { useEffect, useLayoutEffect, useMemo, useState } from "react";

import {
  DARK_QUERY,
  type ResolvedTheme,
  type Theme,
  ThemeContext,
  applyTheme,
  isThemeStorageKey,
  readStoredTheme,
  storeTheme,
  systemTheme,
} from "@/lib/theme";

/**
 * Light, dark or follow the system. The choice is saved in localStorage;
 * with "system" (the default) the page follows the OS setting as it changes.
 */
export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(readStoredTheme);
  const [system, setSystem] = useState<ResolvedTheme>(systemTheme);
  const resolvedTheme = theme === "system" ? system : theme;

  // Follow the OS setting.
  useEffect(() => {
    const query = window.matchMedia(DARK_QUERY);
    const onChange = () => setSystem(query.matches ? "dark" : "light");
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);

  // Follow a choice made in another tab.
  useEffect(() => {
    const onStorage = (event: StorageEvent) => {
      if (isThemeStorageKey(event.key)) setThemeState(readStoredTheme());
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  // Before paint, so a change never flashes the old theme.
  useLayoutEffect(() => applyTheme(resolvedTheme), [resolvedTheme]);

  const value = useMemo(
    () => ({
      theme,
      resolvedTheme,
      setTheme: (next: Theme) => {
        storeTheme(next);
        setThemeState(next);
      },
    }),
    [theme, resolvedTheme],
  );

  return <ThemeContext value={value}>{children}</ThemeContext>;
}

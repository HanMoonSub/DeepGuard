import { useEffect, useState } from 'react';

const STORAGE_KEY = 'dg-theme';

const readTheme = () => {
  try {
    return localStorage.getItem(STORAGE_KEY) === 'dark' ? 'dark' : 'light';
  } catch (e) {
    return 'light';
  }
};

/**
 * Light / Dark theme state.
 * Sets <html data-theme> only while staying on a page that uses this hook, and removes it on leaving.
 * → Pages that don't support theming (main, login, detail, etc.) always appear in light mode.
 */
const useTheme = () => {
  const [theme, setThemeState] = useState(readTheme);

  useEffect(() => {
    const root = document.documentElement;
    root.dataset.theme = theme;
    return () => { delete root.dataset.theme; };
  }, [theme]);

  const setTheme = (next) => {
    setThemeState(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch (e) {
      // Ignore save failures (private browsing, etc.) — the theme is kept for the current screen only
    }
  };

  return [theme, setTheme];
};

export default useTheme;

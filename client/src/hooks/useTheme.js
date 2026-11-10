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
 * Light / Dark 테마 상태.
 * 이 훅을 쓰는 페이지에 머무는 동안에만 <html data-theme>을 붙이고, 떠나면 제거한다.
 * → 테마를 지원하지 않는 페이지(메인·로그인·상세 등)는 항상 라이트로 보인다.
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
      // 저장 실패(사생활 보호 모드 등)는 무시 — 현재 화면에서만 유지
    }
  };

  return [theme, setTheme];
};

export default useTheme;

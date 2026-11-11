import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

import Button from './Button';
import logoMark from '../assets/logo-shield.svg';
import './Navbar.css';

// variant: dark (transparent over the hero → dark background on scroll) | light (white background)
// links: same-page anchor menu [{ href: '#bench', label: 'Benchmark' }]
// logoutRedirect: path to go to after logout
// solid: keep the dark background even before scrolling in the dark variant (for use over a white hero)
// theme / onThemeChange: when passed, shows a sun/moon theme toggle left of login (use with the useTheme hook)
const Navbar = ({ variant = 'dark', sessionUser, onLogout, links = [], logoutRedirect = '/main', solid = false, theme, onThemeChange }) => {
  const navigate = useNavigate();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  // Navigate with the router without a reload (href is for accessibility/opening in a new tab)
  const go = (path) => (e) => {
    e.preventDefault();
    navigate(path);
  };

  const handleLogoutClick = async () => {
    if (!window.confirm("로그아웃 하시겠습니까?")) return;
    try {
      await axios.get('/auth/logout');
      if (onLogout) onLogout();
      alert("성공적으로 로그아웃되었습니다.");
      navigate(logoutRedirect);
    } catch (error) {
      alert("로그아웃 처리 중 오류가 발생했습니다.");
    }
  };

  const subtleBtn = variant === 'dark' ? 'ghost' : 'outline';

  return (
    <nav className={`dg-nav dg-nav--${variant}${solid ? ' dg-nav--solid' : ''}${scrolled ? ' is-scrolled' : ''}`}>
      <div className="dg-nav__inner">
        <a className="dg-nav__logo" href="/main" onClick={go('/main')}>
          <img className="dg-nav__mark" src={logoMark} alt="" aria-hidden="true" />DeepGuard
        </a>

        <div className="dg-nav__menu">
          <a className="dg-nav__link" href="/analysis" onClick={go('/analysis')}>
            이미지 분석
          </a>
          <a className="dg-nav__link" href="/video-analysis" onClick={go('/video-analysis')}>
            동영상 분석<span className="dg-nav__badge">HOT</span>
          </a>
          {links.map(({ href, label }) => (
            <a key={href} className="dg-nav__link dg-nav__anchor" href={href}>{label}</a>
          ))}
        </div>

        <div className="dg-nav__actions">
          {onThemeChange && (
            <button type="button" className="dg-nav__theme"
              onClick={() => onThemeChange(theme === 'dark' ? 'light' : 'dark')}
              aria-label={theme === 'dark' ? 'Light 모드로 전환' : 'Dark 모드로 전환'}
              title={theme === 'dark' ? 'Light 모드' : 'Dark 모드'}>
              {theme === 'dark' ? (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></svg>
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" /></svg>
              )}
            </button>
          )}
          {sessionUser ? (
            <>
              <span className="dg-nav__user"><strong>{sessionUser.name}</strong>님</span>
              <Button variant={subtleBtn} onClick={handleLogoutClick}>로그아웃</Button>
            </>
          ) : (
            <Button variant={subtleBtn} href="/login" onClick={go('/login')}>로그인</Button>
          )}
          <Button variant="primary" className="dg-nav__cta" href="/analysis" onClick={go('/analysis')}>Try Now</Button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;

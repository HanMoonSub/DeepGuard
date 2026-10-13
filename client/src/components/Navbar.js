import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

import Button from './Button';
import './Navbar.css';

// variant: dark(히어로 위 투명 → 스크롤 시 어두운 배경) | light(흰 배경)
// links: 같은 페이지 앵커 메뉴 [{ href: '#bench', label: 'Benchmark' }]
// logoutRedirect: 로그아웃 후 이동 경로
const Navbar = ({ variant = 'dark', sessionUser, onLogout, links = [], logoutRedirect = '/main' }) => {
  const navigate = useNavigate();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  // 새로고침 없이 라우터로 이동 (href는 접근성/새 탭 열기용)
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
    <nav className={`dg-nav dg-nav--${variant}${scrolled ? ' is-scrolled' : ''}`}>
      <div className="dg-nav__inner">
        <a className="dg-nav__logo" href="/main" onClick={go('/main')}>
          <i aria-hidden="true" />DeepGuard
        </a>

        <div className="dg-nav__menu">
          <a className="dg-nav__link" href="/analysis" onClick={go('/analysis')}>이미지 분석</a>
          <a className="dg-nav__link" href="/video-analysis" onClick={go('/video-analysis')}>동영상 분석</a>
          {links.map(({ href, label }) => (
            <a key={href} className="dg-nav__link dg-nav__anchor" href={href}>{label}</a>
          ))}
        </div>

        <div className="dg-nav__actions">
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

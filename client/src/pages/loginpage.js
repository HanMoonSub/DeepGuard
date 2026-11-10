import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

import Button from '../components/Button';
import Card from '../components/Card';
import Eclipse from '../components/Eclipse';
import logoMark from '../assets/logo-shield.svg';
import './AuthPage.css';



const LoginPage = ({ setSessionUser }) => {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    try {
      await axios.post('/auth/login', { email, password });
      
      const homeRes = await axios.get('/home');
      if (homeRes.data.session_user) {
        setSessionUser(homeRes.data.session_user);
        alert(`${homeRes.data.session_user.name}님 환영합니다!`);
        navigate('/analysis');
      }
    } catch (error) {
      const detail = error.response?.data?.detail || "로그인 정보를 확인해주세요.";
      alert(`로그인 실패: ${detail}`);
    } finally {
      setIsLoading(false);
    }
  };

  // 새로고침 없이 라우터로 이동 (href는 접근성/새 탭 열기용)
  const go = (path) => (e) => {
    e.preventDefault();
    navigate(path);
  };

  return (
    <div className="dg-auth">
      <div className="dg-auth__split">
        <aside className="dg-auth__hero">
          <a className="dg-auth__logo" href="/main" onClick={go('/main')}>
            <img className="dg-auth__mark" src={logoMark} alt="" aria-hidden="true" />DeepGuard
          </a>
          <div className="dg-auth__copy dg-auth__copy--centered">
            {/* 메인 히어로와 같은 이클립스 링 (장식용) — 문구 블록 중앙에 맞춤 */}
            <Eclipse />
            <h1 className="dg-auth__title">딥페이크,<br /><em>눈으로 확인</em>하세요</h1>
            <p className="dg-auth__lead">
              이미지와 영상 속 위조 흔적을 탐지하고, 어디가 왜 가짜인지 시각적 근거까지 제공하는 딥페이크 탐지 서비스.
            </p>
          </div>
        </aside>

        <main className="dg-auth__main">
          <Card className="dg-card--static dg-auth__card" title="로그인">
            <p className="dg-auth__sub">Deep Guard 서비스를 이용하려면 로그인하세요.</p>
            <form className="dg-auth__form" onSubmit={handleLogin}>
              <label className="dg-auth__field">
                <span className="dg-auth__label">이메일 주소</span>
                <input className="dg-auth__input" type="email" placeholder="이메일 주소" value={email} onChange={(e) => setEmail(e.target.value)} required />
              </label>
              <label className="dg-auth__field">
                <span className="dg-auth__label">비밀번호</span>
                <input className="dg-auth__input" type="password" placeholder="비밀번호" value={password} onChange={(e) => setPassword(e.target.value)} required />
              </label>
              <Button type="submit" variant="primary" className="dg-btn--block dg-auth__submit" disabled={isLoading}>
                {isLoading ? "인증 중..." : "로그인 하기"}
              </Button>
            </form>
            <p className="dg-auth__switch">
              아직 계정이 없으신가요?
              <button type="button" className="dg-auth__link" onClick={() => navigate('/register')}>회원가입</button>
            </p>
          </Card>
        </main>
      </div>
    </div>
  );
};

export default LoginPage;
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

import Button from '../components/Button';
import Card from '../components/Card';
import Footer from '../components/Footer';
import './AuthPage.css';

const SignupPage = () => {
  const navigate = useNavigate();

  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isAgreed, setIsAgreed] = useState(false); 
  const [isLoading, setIsLoading] = useState(false);

  const handleSignup = async (e) => {
    e.preventDefault();
    
    if (password.length < 8) {
      alert("비밀번호는 8자 이상이어야 합니다.");
      return;
    }

    if (!isAgreed) {
      alert("이용 수칙에 동의해 주세요.");
      return;
    }

    setIsLoading(true);

    try {
      const response = await axios.post('/auth/register', {
        name,
        email,
        password
      });

      if (response.status === 200 || response.status === 201) {
        alert(response.data.message || "회원가입이 완료되었습니다! 로그인해 주세요."); 
        navigate('/login');
      }
    } catch (error) {
      if (error.response && error.response.data) {
        const { error_type, title_message, detail } = error.response.data;

        if (error_type === "valid") {
          alert(`[입력 오류] ${detail}`);
        } else {
          alert(`[${title_message}] ${detail}`);
        }
      } else {
        alert("서버와 통신할 수 없습니다. 네트워크 상태를 확인해주세요.");
      }
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
            <i aria-hidden="true" />DeepGuard
          </a>
          <div className="dg-auth__copy">
            <p className="dg-auth__eyebrow">GET STARTED</p>
            <h1 className="dg-auth__title">Deep Guard에 오신 걸 환영합니다!</h1>
            <p className="dg-auth__lead">계정을 생성하고 딥가드를 시작하세요</p>
          </div>
        </aside>

        <main className="dg-auth__main">
          <Card className="dg-card--static dg-auth__card" title="회원가입">
            <form className="dg-auth__form" onSubmit={handleSignup}>
              <label className="dg-auth__field">
                <span className="dg-auth__label">이름</span>
                <input className="dg-auth__input" type="text" placeholder="이름을 입력하세요" value={name} onChange={(e) => setName(e.target.value)} required />
              </label>
              <label className="dg-auth__field">
                <span className="dg-auth__label">이메일 주소</span>
                <input className="dg-auth__input" type="email" placeholder="example@gmail.com" value={email} onChange={(e) => setEmail(e.target.value)} required />
              </label>
              <label className="dg-auth__field">
                <span className="dg-auth__label">비밀번호</span>
                <input className="dg-auth__input" type="password" placeholder="8자 이상 입력해 주세요." value={password} onChange={(e) => setPassword(e.target.value)} required />
              </label>

              <div className="dg-auth__check">
                <input
                  type="checkbox"
                  id="terms"
                  checked={isAgreed}
                  onChange={(e) => setIsAgreed(e.target.checked)}
                />
                <label htmlFor="terms">이용 수칙에 동의합니다.</label>
              </div>

              <Button type="submit" variant="primary" className="dg-btn--block dg-auth__submit" disabled={isLoading}>
                {isLoading ? "처리 중..." : "회원 가입"}
              </Button>
            </form>
            <p className="dg-auth__switch">
              이미 계정이 있으신가요?
              <button type="button" className="dg-auth__link" onClick={() => navigate('/login')}>로그인</button>
            </p>
          </Card>
        </main>
      </div>
      <Footer />
    </div>
  );
};

export default SignupPage;
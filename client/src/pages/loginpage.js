import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios'; 

import logo from '../assets/logo.svg';

const LoginPage = ({ setSessionUser }) => {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      const response = await axios.post('http://localhost:8000/auth/login', { email, password });
      
      if (response.status === 200) {
        const token = response.data.access_token;
        localStorage.setItem('token', token); 

        const homeRes = await axios.get('http://localhost:8000/home', {
          headers: { Authorization: `Bearer ${token}` }
        });

        if (homeRes.data.session_user) {
          setSessionUser(homeRes.data.session_user);
          alert(`${homeRes.data.session_user.name}님 환영합니다!`);
        }
        
        navigate('/analysis');
      }
    } catch (error) {
      if (error.response && error.response.data) {
        const { title_message, detail } = error.response.data;
        alert(`[${title_message || '오류'}] ${detail || '로그인에 실패했습니다.'}`);
      } else {
        alert("서버와 통신할 수 없습니다.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const containerStyle = {
    backgroundColor: '#000',
    height: '100vh',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    color: 'white',
    fontFamily: 'sans-serif'
  };

  const loginBoxStyle = {
    width: '400px',
    padding: '40px',
    backgroundColor: '#111',
    borderRadius: '20px',
    border: '1px solid #222',
    textAlign: 'center'
  };

  const inputStyle = {
    width: '100%',
    padding: '12px 15px',
    marginBottom: '15px',
    borderRadius: '8px',
    border: '1px solid #333',
    backgroundColor: '#000',
    color: 'white',
    boxSizing: 'border-box',
    fontSize: '15px',
    outline: 'none'
  };

  const loginBtnStyle = {
    width: '100%',
    padding: '14px',
    backgroundColor: isLoading ? '#555' : '#39FF14',
    color: '#000',
    border: 'none',
    borderRadius: '8px',
    fontWeight: 'bold',
    fontSize: '16px',
    cursor: isLoading ? 'not-allowed' : 'pointer',
    marginTop: '10px',
    transition: '0.3s'
  };

  return (
    <div style={containerStyle}>
      <img 
        src={logo} 
        alt="Deep Guard" 
        style={{ height: '40px', marginBottom: '30px', cursor: 'pointer' }} 
        onClick={() => navigate('/main')}
      />
      
      <div style={loginBoxStyle}>
        <h2 style={{ marginBottom: '10px', fontSize: '26px' }}>로그인</h2>
        <p style={{ color: '#666', fontSize: '14px', marginBottom: '30px' }}>
          Deep Guard 서비스를 이용하려면 로그인하세요.
        </p>
        
        <form onSubmit={handleLogin}>
          <input
            type="email"
            placeholder="이메일 주소"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            style={inputStyle}
            required
          />
          <input
            type="password"
            placeholder="비밀번호"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            style={inputStyle}
            required
          />
          
          <button type="submit" style={loginBtnStyle} disabled={isLoading}>
            {isLoading ? "인증 중..." : "로그인 하기"}
          </button>
        </form>
        
        <div style={{ marginTop: '25px', fontSize: '14px', color: '#888' }}>
          아직 계정이 없으신가요?{' '}
          <span 
            onClick={() => navigate('/signup')} 
            style={{ color: '#39FF14', cursor: 'pointer', fontWeight: 'bold', textDecoration: 'underline' }}
          >
            회원가입
          </span>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
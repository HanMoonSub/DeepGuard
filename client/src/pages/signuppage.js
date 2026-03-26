import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

import logo from '../assets/logo.svg';
import medicalIcon from '../assets/MedicalIcons.svg';
import ellipse from '../assets/Ellipse638.svg';

const SignupPage = () => {
  const navigate = useNavigate();

  // 입력값 상태 관리
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  
  // 이용약관 동의 및 로딩 상태
  const [isAgreed, setIsAgreed] = useState(false); 
  const [isLoading, setIsLoading] = useState(false);

  // 회원가입 처리 함수
  const handleSignup = async (e) => {
    e.preventDefault();
    
    // 보안: 1차 유효성 검사
    if (password.length < 8) {
      alert("비밀번호는 8자 이상이어야 합니다.");
      return;
    }

    if (!isAgreed) {
      alert("이용 수칙에 동의해 주세요.");
      return;
    }

    setIsLoading(true); // 로딩 시작 (버튼 비활성화)

    try {
      const response = await axios.post('http://localhost:8000/signup', {
        name,
        email,
        password
      });

      if (response.status === 200 || response.status === 201) {
        alert("회원가입 성공! 로그인 페이지로 이동합니다.");
        navigate('/login');
      }
    } catch (error) {
      if (error.response) {
        const { title_message, detail } = error.response.data;
        alert(`[${title_message}] ${detail}`);
      } else {
        alert("서버와 통신할 수 없습니다.");
      }
    } finally {
      setIsLoading(false); // 성공하든 실패하든 로딩 종료
    }
  };

  // 스타일 정의
  const containerStyle = {
    backgroundColor: '#000000',
    height: '100vh',
    width: '100vw',
    display: 'flex',
    justifyContent: 'space-around',
    alignItems: 'center',
    color: 'white',
    fontFamily: 'sans-serif',
    overflow: 'hidden',
    position: 'relative',
  };

  const formSideStyle = { zIndex: 2, display: 'flex', flexDirection: 'column', width: '600px' };

  const inputStyle = {
    backgroundColor: '#050505', 
    border: '1px solid #333',
    borderRadius: '8px',
    padding: '15px',
    color: 'white',
    marginTop: '5px',
    width: '100%',
    boxSizing: 'border-box'
  };

  const buttonStyle = {
    backgroundColor: isLoading ? '#555' : '#39FF14', // 로딩 중일 때 색상 변경
    color: 'black',
    padding: '15px',
    border: 'none',
    borderRadius: '8px',
    cursor: isLoading ? 'not-allowed' : 'pointer',
    fontWeight: 'bold',
    fontSize: '16px',
    marginTop: '30px',
    width: '100%',
    opacity: isLoading ? 0.7 : 1
  };

  return (
    <div style={containerStyle}>
      <img src={ellipse} alt="" style={{ position: 'absolute', bottom: '-15%', right: '-5%', width: '70%', zIndex: 1 }} />

      <div style={formSideStyle}>
        <div style={{ marginBottom: '30px' }}>
          <img src={logo} alt="Deep Guard" style={{ height: '35px' }} />
        </div>

        <h1 style={{ fontSize: '36px', marginBottom: '10px', fontWeight: 'bold' }}>Deep Guard에 오신 걸 환영합니다!</h1>
        <p style={{ color: '#aaa', marginBottom: '40px' }}>계정을 생성하고 딥가드를 시작하세요</p>

        <form onSubmit={handleSignup} style={{ display: 'flex', flexDirection: 'column' }}>
          <label style={{ color: '#39FF14', fontSize: '14px', marginTop: '15px' }}>이름</label>
          <input type="text" placeholder="Lee yesol" style={inputStyle} value={name} onChange={(e) => setName(e.target.value)} required />

          <label style={{ color: '#39FF14', fontSize: '14px', marginTop: '15px' }}>이메일 주소</label>
          <input type="email" placeholder="yesol@example.com" style={inputStyle} value={email} onChange={(e) => setEmail(e.target.value)} required />

          <label style={{ color: '#39FF14', fontSize: '14px', marginTop: '15px' }}>비밀번호</label>
          <input type="password" placeholder="8자 이상 입력해 주세요." style={inputStyle} value={password} onChange={(e) => setPassword(e.target.value)} required />

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '20px' }}>
            {/* 리액트 방식으로 관리되는 체크박스 */}
            <input 
              type="checkbox" 
              id="terms" 
              checked={isAgreed} 
              onChange={(e) => setIsAgreed(e.target.checked)} 
              style={{ accentColor: '#39FF14' }} 
            />
            <label htmlFor="terms" style={{ fontSize: '13px', color: '#ccc' }}>이용 수칙에 동의합니다.</label>
          </div>

          <button type="submit" style={buttonStyle} disabled={isLoading}>
            {isLoading ? "처리 중..." : "회원 가입"}
          </button>
        </form>
        
        <p style={{ marginTop: '25px', fontSize: '14px', color: '#888', textAlign: 'center' }}>
          Already have an Account? <span style={{ color: '#39FF14', cursor: 'pointer', fontWeight: 'bold' }} onClick={() => navigate('/login')}>Login</span>
        </p>
      </div>

      <div style={{ zIndex: 2 }}>
        <img src={medicalIcon} alt="Medical Icon" style={{ width: '450px', marginTop: '-180px' }} />
      </div>
    </div>
  );
};

export default SignupPage;
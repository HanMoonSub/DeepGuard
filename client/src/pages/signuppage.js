import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

import logo from '../assets/logo.svg';
import medicalIcon from '../assets/MedicalIcons.svg';
import ellipse from '../assets/Ellipse638.svg';

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
      const response = await axios.post('http://localhost:8000/auth/register', {
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
    backgroundColor: isLoading ? '#555' : '#39FF14',
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
          <input type="text" placeholder="이름을 입력하세요" style={inputStyle} value={name} onChange={(e) => setName(e.target.value)} required />

          <label style={{ color: '#39FF14', fontSize: '14px', marginTop: '15px' }}>이메일 주소</label>
          <input type="email" placeholder="example@gmail.com" style={inputStyle} value={email} onChange={(e) => setEmail(e.target.value)} required />

          <label style={{ color: '#39FF14', fontSize: '14px', marginTop: '15px' }}>비밀번호</label>
          <input type="password" placeholder="8자 이상 입력해 주세요." style={inputStyle} value={password} onChange={(e) => setPassword(e.target.value)} required />

          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginTop: '20px' }}>
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
          이미 계정이 있으신가요? <span style={{ color: '#39FF14', cursor: 'pointer', fontWeight: 'bold' }} onClick={() => navigate('/login')}>로그인</span>
        </p>
      </div>

      <div style={{ zIndex: 2 }}>
        <img src={medicalIcon} alt="Decoration" style={{ width: '450px', marginTop: '-180px' }} />
      </div>
    </div>
  );
};

export default SignupPage;
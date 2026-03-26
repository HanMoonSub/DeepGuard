import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios'; // axios 라이브러리 설치 필요

const LoginPage = () => {
  const navigate = useNavigate();

  //변수명으로 상태 정의
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState(''); // hashed_password 아님

  //로그인 처리 함수 (비동기)
  const handleLogin = async (e) => {
    e.preventDefault();
    
    try {
      // 금요일에 연동할 백엔드 주소
      const response = await axios.post('http://localhost:8000/login', {
        email,
        password
      });

      if (response.status === 200) {
        alert("로그인 성공!");
        navigate('/main'); // 로그인 성공 시 메인으로 이동
      }
    } catch (error) {
      // 백엔드 에러 규격 (status_code, title_message, detail) 처리
      if (error.response) {
        const { title_message, detail } = error.response.data;
        alert(`[${title_message}] ${detail}`);
      } else {
        alert("서버와 통신할 수 없습니다.");
      }
    }
  };

  // 스타일 정의 (기존 유지)
  const containerStyle = {
    backgroundColor: '#000000',
    height: '100vh',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    color: 'white',
    fontFamily: 'sans-serif',
  };

  const inputStyle = {
    width: '300px',
    padding: '12px',
    margin: '8px 0',
    backgroundColor: '#1E1E1E',
    border: '1px solid #333',
    borderRadius: '4px',
    color: 'white',
  };

  const buttonStyle = {
    width: '325px',
    padding: '12px',
    backgroundColor: '#39FF14',
    border: 'none',
    borderRadius: '4px',
    fontWeight: 'bold',
    cursor: 'pointer',
    marginTop: '20px',
  };

  return (
    <div style={containerStyle}>
      <h2 style={{ marginBottom: '30px', fontSize: '2rem' }}>로그인</h2>
      
      {/* form 태그로 감싸서 Enter 키로도 로그인이 가능하게 */}
      <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column' }}>
        <label style={{ fontSize: '0.8rem', color: '#888' }}>이메일</label>
        <input 
          type="email" 
          placeholder="xxxxx@gmail.com" 
          style={inputStyle} 
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        
        <label style={{ fontSize: '0.8rem', color: '#888', marginTop: '10px' }}>비밀번호</label>
        <input 
          type="password" 
          placeholder="xxxxxxxx" 
          style={inputStyle} 
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        
        <button type="submit" style={buttonStyle}>로그인</button>
      </form>

      <div style={{ marginTop: '20px', fontSize: '0.9rem' }}>
        계정이 없나요?{' '}
        {/*클릭 시 회원가입 페이지로 이동 */}
        <span 
          style={{ color: '#39FF14', cursor: 'pointer', fontWeight: 'bold' }} 
          onClick={() => navigate('/signup')}
        >
          회원 가입
        </span>
      </div>
    </div>
  );
};

export default LoginPage;
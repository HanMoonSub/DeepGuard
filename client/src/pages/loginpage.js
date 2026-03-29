import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios'; 

const LoginPage = () => {
  const navigate = useNavigate();

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState(''); 

  const handleLogin = async (e) => {
    e.preventDefault();
    
    try {
      // 엔드포인트 수정: /auth/login
      const response = await axios.post('http://localhost:8000/auth/login', {
        email,
        password
      });

      // 백엔드에서 성공 응답(200)이 왔을 때
      if (response.status === 200) {
        // 서버에서 준 response.data.message를 alert로 띄움
        alert(response.data.message); 
        
        // 성공 시 리액트 내부가 아닌 백엔드 홈(http://localhost:8000)으로 강제 이동
        window.location.href = 'http://localhost:8000'; 
      }
    } catch (error) {
      if (error.response) {
        // 서버가 에러 규격에 맞춰준 데이터 출력
        const { title_message, detail } = error.response.data;
        alert(`[${title_message}] ${detail}`);
      } else {
        alert("서버와 통신할 수 없습니다.");
      }
    }
  };

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
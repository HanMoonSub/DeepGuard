import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios'; 

const LoginPage = ({ setSessionUser }) => {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState(''); 

  const handleLogin = async (e) => {
    e.preventDefault();
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
        alert(`[${title_message}] ${detail}`);
      } else {
        alert("서버 통신 실패");
      }
    }
  };

  const containerStyle = { backgroundColor: '#000000', height: '100vh', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', color: 'white', fontFamily: 'sans-serif' };
  const inputStyle = { width: '300px', padding: '12px', margin: '8px 0', backgroundColor: '#1E1E1E', border: '1px solid #333', borderRadius: '4px', color: 'white' };
  const buttonStyle = { width: '325px', padding: '12px', backgroundColor: '#39FF14', border: 'none', borderRadius: '4px', fontWeight: 'bold', cursor: 'pointer', marginTop: '20px', color: 'black' };

  return (
    <div style={containerStyle}>
      <h2 style={{ marginBottom: '30px', fontSize: '2rem' }}>로그인</h2>
      <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column' }}>
        <label style={{ fontSize: '0.8rem', color: '#888' }}>이메일</label>
        <input type="email" placeholder="xxxxx@gmail.com" style={inputStyle} value={email} onChange={(e) => setEmail(e.target.value)} required />
        <label style={{ fontSize: '0.8rem', color: '#888', marginTop: '10px' }}>비밀번호</label>
        <input type="password" placeholder="xxxxxxxx" style={inputStyle} value={password} onChange={(e) => setPassword(e.target.value)} required />
        <button type="submit" style={buttonStyle}>로그인</button>
      </form>
      <div style={{ marginTop: '20px', fontSize: '0.9rem' }}>
        계정이 없나요? <span style={{ color: '#39FF14', cursor: 'pointer', fontWeight: 'bold' }} onClick={() => navigate('/signup')}>회원 가입</span>
      </div>
    </div>
  );
};

export default LoginPage;
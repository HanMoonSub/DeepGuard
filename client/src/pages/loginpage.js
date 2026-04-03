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
      }
    }
  };

  return (
    <div style={{ backgroundColor: '#000', height: '100vh', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', color: 'white' }}>
      <h2 style={{ marginBottom: '30px', fontSize: '2rem' }}>로그인</h2>
      <form onSubmit={handleLogin} style={{ display: 'flex', flexDirection: 'column' }}>
        <input type="email" placeholder="이메일" style={{ width: '300px', padding: '12px', margin: '8px 0', backgroundColor: '#1E1E1E', border: '1px solid #333', color: 'white' }} value={email} onChange={(e) => setEmail(e.target.value)} required />
        <input type="password" placeholder="비밀번호" style={{ width: '300px', padding: '12px', margin: '8px 0', backgroundColor: '#1E1E1E', border: '1px solid #333', color: 'white' }} value={password} onChange={(e) => setPassword(e.target.value)} required />
        <button type="submit" style={{ width: '325px', padding: '12px', backgroundColor: '#39FF14', border: 'none', fontWeight: 'bold', marginTop: '20px', cursor:'pointer' }}>로그인</button>
      </form>
    </div>
  );
};
export default LoginPage;
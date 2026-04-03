import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import axios from 'axios';

import MainPage from './pages/mainpage';
import LoginPage from './pages/loginpage';
import SignupPage from './pages/signuppage';
import AnalysisPage from './pages/analysispage'; 

function App() {
  const [sessionUser, setSessionUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const checkLoggedIn = async () => {
      const token = localStorage.getItem('token');
      
      if (token) {
        try {
          const response = await axios.get('http://localhost:8000/home', {
            headers: { Authorization: `Bearer ${token}` }
          });

          if (response.data.session_user) {
            setSessionUser(response.data.session_user);
          }
        } catch (error) {
          // 토큰이 만료되었거나 유효하지 않은 경우 로그아웃 처리
          console.error("세션 만료 또는 인증 에러:", error);
          handleLogout();
        }
      }
      setIsLoading(false);
    };

    checkLoggedIn();
  }, []);

  // 로그아웃 로직 
  const handleLogout = () => {
    localStorage.removeItem('token'); 
    setSessionUser(null);             
    alert("로그아웃 되었습니다.");
    window.location.href = '/main';
  };

  if (isLoading) {
    return (
      <div style={{ 
        backgroundColor: '#000', 
        height: '100vh', 
        color: '#39FF14', 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center',
        fontFamily: 'sans-serif'
      }}>
        Loading...
      </div>
    );
  }

  return (
    <Router>
      <Routes>
        <Route path="/" element={<Navigate replace to="/main" />} />
        
        <Route 
          path="/main" 
          element={<MainPage sessionUser={sessionUser} onLogout={handleLogout} />} 
        />
        
        <Route 
          path="/login" 
          element={<LoginPage setSessionUser={setSessionUser} />} 
        />
        
        <Route path="/signup" element={<SignupPage />} />

        <Route 
          path="/analysis" 
          element={<AnalysisPage sessionUser={sessionUser} />} 
        />
      </Routes>
    </Router>
  );
}

export default App;
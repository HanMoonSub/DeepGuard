import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import axios from 'axios';

import MainPage from './pages/mainpage';
import LoginPage from './pages/loginpage';
import RegisterPage from './pages/signuppage'; 
import AnalysisPage from './pages/analysispage';
import AnalysisDetailPage from './pages/AnalysisDetailPage';
import VideoAnalysisPage from './pages/VideoAnalysisPage';

// 쿠키 공유 설정 유지
axios.defaults.withCredentials = true;

function App() {
  const [sessionUser, setSessionUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // 세션 확인 로직 유지
  useEffect(() => {
    const checkSession = async () => {
      try {
        const response = await axios.get('/auth/check');
        if (response.data && response.data.user) {
          setSessionUser(response.data.user);
        }
      } catch (error) {
        setSessionUser(null);
      } finally {
        setLoading(false);
      }
    };
    checkSession();
  }, []);

  const handleLogout = () => {
    setSessionUser(null); 
  };

  if (loading) return <div style={{ backgroundColor: '#000', height: '100vh' }}></div>;

  return (
    <Router>
      <Routes>
        <Route path="/" element={<Navigate to="/main" />} />
        <Route path="/main" element={<MainPage sessionUser={sessionUser} onLogout={handleLogout} />} />
        
        {/* 이미지 분석 (기존 유지) */}
        <Route path="/analysis" element={<AnalysisPage sessionUser={sessionUser} onLogout={handleLogout} setSessionUser={setSessionUser} />} />
        
        {/* 비디오 분석 (신규 추가) */}
        <Route path="/video-analysis" element={<VideoAnalysisPage sessionUser={sessionUser} onLogout={handleLogout} setSessionUser={setSessionUser} />} />
        
        {/* 공용 상세 페이지 */}
        <Route path="/analysis-detail" element={<AnalysisDetailPage sessionUser={sessionUser} />} />
        
        <Route path="/login" element={<LoginPage setSessionUser={setSessionUser} />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="*" element={<Navigate to="/main" />} />
      </Routes>
    </Router>
  );
}

export default App;
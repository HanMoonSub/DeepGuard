import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import MainPage from './pages/mainpage';
import LoginPage from './pages/loginpage';
import SignupPage from './pages/signuppage';

function App() {
  return (
    <Router>
      <Routes>
        {/* 첫 접속 시 메인 페이지가 나오도록 설정 */}
        <Route path="/" element={<MainPage />} />
        
        {/* 각 페이지별 주소 */}
        <Route path="/main" element={<MainPage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/signup" element={<SignupPage />} />
      </Routes>
    </Router>
  );
}

export default App;
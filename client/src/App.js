import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import axios from 'axios';

import MainPage from './pages/mainpage';
import LoginPage from './pages/loginpage';
import SignupPage from './pages/signuppage';
import AnalysisPage from './pages/analysispage'; 

axios.defaults.withCredentials = true;

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
          handleLogout();
        }
      }
      setIsLoading(false);
    };
    checkLoggedIn();
  }, []);

  const handleLogout = async () => {
    try {
      const response = await axios.get('http://localhost:8000/auth/logout');
      if (response.data.status === "success") {
        localStorage.removeItem('token');
        setSessionUser(null);
        alert(response.data.message);
        window.location.href = '/main';
      }
    } catch (error) {
      localStorage.removeItem('token');
      setSessionUser(null);
      window.location.href = '/main';
    }
  };

  if (isLoading) return <div style={{backgroundColor:'#000', color:'#39FF14', height:'100vh', display:'flex', justifyContent:'center', alignItems:'center'}}>Loading...</div>;

  return (
    <Router>
      <Routes>
        <Route path="/" element={<Navigate replace to="/main" />} />
        <Route path="/main" element={<MainPage sessionUser={sessionUser} onLogout={handleLogout} />} />
        <Route path="/login" element={<LoginPage setSessionUser={setSessionUser} />} />
        <Route path="/signup" element={<SignupPage />} />
        <Route path="/analysis" element={<AnalysisPage sessionUser={sessionUser} />} />
      </Routes>
    </Router>
  );
}

export default App;
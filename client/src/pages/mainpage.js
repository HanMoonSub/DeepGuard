import React from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

import logo from '../assets/logo.svg'; 
import circle from '../assets/circle.svg';
import bgCurve from '../assets/line.svg'; 

const MainPage = ({ sessionUser, onLogout }) => {
  const navigate = useNavigate();

  // 이미지 분석 페이지로 이동
  const handleBasicAnalysis = () => navigate('/analysis');
  
  // 동영상 분석 페이지로 이동 (신규)
  const handleVideoAnalysis = () => navigate('/video-analysis');

  const handleProAnalysis = () => {
    if (sessionUser) {
      navigate('/analysis');
    } else {
      alert("Pro 모델 분석은 로그인이 필요합니다.");
      navigate('/login');
    }
  };

  const handleLogoutClick = async () => {
    if (!window.confirm("로그아웃 하시겠습니까?")) return;
    try {
      await axios.get('/auth/logout');
      if (onLogout) onLogout(); 
      alert("성공적으로 로그아웃되었습니다.");
      navigate('/main');
    } catch (error) {
      alert("로그아웃 처리 중 오류가 발생했습니다.");
    }
  };

  const containerStyle = { backgroundColor: '#000000', minHeight: '100vh', width: '100vw', color: 'white', fontFamily: 'sans-serif', display: 'flex', flexDirection: 'column', position: 'relative', overflowX: 'hidden' };
  const navStyle = { position: 'sticky', top: 0, width: '100%', padding: '20px 80px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: 'rgba(0, 0, 0, 0.9)', backdropFilter: 'blur(10px)', zIndex: 100, boxSizing: 'border-box', borderBottom: '1px solid #111' };
  const sectionStyle = { display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '100px 20px', textAlign: 'center', position: 'relative', zIndex: 10 };
  const analysisBoxStyle = { width: '420px', height: '500px', backgroundColor: '#111', borderRadius: '24px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', padding: '40px', boxSizing: 'border-box', border: '1px solid #222', transition: '0.3s' };
  const modelBtnBase = { width: '100%', padding: '14px 20px', borderRadius: '30px', fontSize: '15px', fontWeight: 'bold', cursor: 'pointer', border: 'none', display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '12px', color: 'white' };

  return (
    <div style={containerStyle}>
      <img src={bgCurve} alt="" style={{ position: 'absolute', bottom: '0', right: '0', width: '80%', opacity: 0.6, zIndex: 1 }} />

      <nav style={navStyle}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <img src={logo} alt="Deep Guard" style={{ height: '32px', marginRight: '40px', cursor: 'pointer' }} onClick={() => navigate('/main')} />
          {sessionUser && (
            <span style={{ fontSize: '18px', color: '#ccc', marginLeft: '10px' }}>
              <strong style={{ color: '#39FF14' }}>{sessionUser.name}</strong>님 환영합니다
            </span>
          )}
        </div>
        
        <div style={{ display: 'flex', gap: '30px', alignItems: 'center' }}>
          <button onClick={handleBasicAnalysis} style={{ background: 'none', border: 'none', color: '#fff', cursor: 'pointer', fontWeight: 'bold' }}>이미지 분석</button>
          <button onClick={handleVideoAnalysis} style={{ background: 'none', border: 'none', color: '#fff', cursor: 'pointer', fontWeight: 'bold' }}>동영상 분석</button>
          
          {sessionUser ? (
            <button onClick={handleLogoutClick} style={{ backgroundColor: '#FF4B4B', color: 'white', padding: '8px 25px', borderRadius: '20px', border: 'none', cursor: 'pointer', fontWeight: 'bold' }}>로그아웃</button>
          ) : (
            <button onClick={() => navigate('/login')} style={{ backgroundColor: '#333', color: 'white', padding: '8px 25px', borderRadius: '20px', border: 'none', cursor: 'pointer', fontWeight: 'bold' }}>로그인</button>
          )}
        </div>
      </nav>

      <section style={{ ...sectionStyle, minHeight: '85vh' }}>
        <div style={{ border: '1px solid #333', padding: '6px 16px', borderRadius: '20px', fontSize: '13px', color: '#39FF14', marginBottom: '25px', letterSpacing: '1px' }}>DeepGuard AI System</div>
        <h1 style={{ fontSize: '52px', marginBottom: '30px', fontWeight: 'bold', lineHeight: '1.3' }}>
          Deep Guard는 사용자가 업로드한<br/>
          이미지 또는 비디오의 <span style={{ color: '#ffffff' }}>딥페이크 변조 여부</span>와<br/>
          상세 정보를 제공합니다.
        </h1>
        <p style={{ color: '#888', marginBottom: '50px', fontSize: '18px' }}>정밀한 AI 모델을 통해 지금 바로 분석을 시작해보세요.</p>
        <button style={{ backgroundColor: '#39FF14', color: '#000', padding: '16px 45px', borderRadius: '40px', fontSize: '18px', fontWeight: 'bold', cursor: 'pointer', border: 'none' }} onClick={handleBasicAnalysis}>무료 분석 시작 ❯</button>
      </section>

      <section style={{ ...sectionStyle, flexDirection: 'row', gap: '40px', paddingBottom: '150px' }}>
        {/* 이미지 분석 박스 */}
        <div style={analysisBoxStyle} onMouseOver={(e) => e.currentTarget.style.borderColor = '#39FF14'} onMouseOut={(e) => e.currentTarget.style.borderColor = '#222'}>
          <h3 style={{ fontSize: '28px', marginBottom: '25px' }}>이미지 분석</h3>
          <img src={circle} alt="" style={{ width: '140px', marginBottom: '40px', opacity: 0.8 }} />
          <button style={{ ...modelBtnBase, backgroundColor: '#333' }} onClick={handleBasicAnalysis}>Fast 모델 <span>❯</span></button>
          <button style={{ ...modelBtnBase, backgroundColor: '#000', border: '1px solid #444' }} onClick={handleProAnalysis}>
            <span style={{ color: 'white' }}>Pro 정밀 모델</span>
            <span style={{ color: '#39FF14' }}>❯</span>
          </button>
        </div>
        
        {/* 비디오 분석 박스 */}
        <div style={analysisBoxStyle} onMouseOver={(e) => e.currentTarget.style.borderColor = '#39FF14'} onMouseOut={(e) => e.currentTarget.style.borderColor = '#222'}>
          <h3 style={{ fontSize: '28px', marginBottom: '25px' }}>비디오 분석</h3>
          <img src={circle} alt="" style={{ width: '140px', marginBottom: '40px', opacity: 0.8 }} />
          <button style={{ ...modelBtnBase, backgroundColor: '#333' }} onClick={handleVideoAnalysis}>Fast 모델 <span>❯</span></button>
          <button style={{ ...modelBtnBase, backgroundColor: '#000', border: '1px solid #444' }} onClick={handleVideoAnalysis}>
            <span style={{ color: 'white' }}>Pro 정밀 모델</span>
            <span style={{ color: '#39FF14' }}>❯</span>
          </button>
        </div>
      </section>
    </div>
  );
};

export default MainPage;
import React from 'react';
import { useNavigate } from 'react-router-dom';

import logo from '../assets/logo.svg'; 
import circle from '../assets/circle.svg';
import bgCurve from '../assets/line.svg'; 

const MainPage = ({ sessionUser, onLogout }) => {
  const navigate = useNavigate();

  // Basic 모델 분석 (Fast) -> 바로 업로드 페이지로
  const handleBasicAnalysis = () => {
    navigate('/analysis');
  };

  // Pro 모델 분석 -> 로그인 체크 후 이동
  const handleProAnalysis = () => {
    if (sessionUser) {
      navigate('/analysis');
    } else {
      alert("Pro 모델 분석은 로그인이 필요합니다.");
      navigate('/login');
    }
  };

  const handleHomeRedirect = () => {
    navigate('/main');
  };

  const containerStyle = { backgroundColor: '#000000', minHeight: '200vh', width: '100vw', color: 'white', fontFamily: 'sans-serif', display: 'flex', flexDirection: 'column', position: 'relative', overflowX: 'hidden' };
  const navStyle = { position: 'sticky', top: 0, width: '100%', padding: '20px 80px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: 'rgba(0, 0, 0, 0.9)', backdropFilter: 'blur(10px)', zIndex: 100, boxSizing: 'border-box', borderBottom: '1px solid #222' };
  const sectionStyle = { display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '120px 20px', textAlign: 'center', position: 'relative', zIndex: 2 };
  
  const analysisBoxStyle = { width: '450px', height: '520px', backgroundColor: '#111', borderRadius: '24px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', padding: '40px', boxSizing: 'border-box', border: '1px solid #222' };
  
  const modelBtnBase = { width: '100%', padding: '14px 24px', borderRadius: '30px', fontSize: '16px', fontWeight: 'bold', cursor: 'pointer', border: 'none', display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '12px' };
  const basicBtnStyle = { ...modelBtnBase, backgroundColor: '#2A2A2A', color: 'white' };
  const proBtnStyle = { ...modelBtnBase, backgroundColor: '#1E1E1E', color: '#888' };
  const navBtnStyle = { color: '#39FF14', cursor: 'pointer', fontSize: '18px', fontWeight: 'bold', background: 'none', border: 'none', marginRight: '30px' };

  return (
    <div style={containerStyle}>
      <img src={bgCurve} alt="" style={{ position: 'absolute', bottom: '0', right: '0', width: '90%', opacity: 0.6, zIndex: 1 }} />

      {/* 네비게이션 바 */}
      <nav style={navStyle}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <img src={logo} alt="Deep Guard" style={{ height: '35px', marginRight: '50px', cursor: 'pointer' }} onClick={handleHomeRedirect} />
          {sessionUser && (
            <span style={{ color: '#39FF14', fontWeight: 'bold', fontSize: '18px' }}>
              {sessionUser.name}님 환영합니다
            </span>
          )}
        </div>
        <div>
          <button style={navBtnStyle} onClick={handleBasicAnalysis}>이미지 분석</button>
          <button style={navBtnStyle} onClick={handleBasicAnalysis}>동영상 분석</button>
          {sessionUser ? (
            <button onClick={onLogout} style={{ ...navBtnStyle, marginRight: 0, color: 'white', backgroundColor: '#FF4B4B', padding: '10px 20px', borderRadius: '20px' }}>로그아웃</button>
          ) : (
            <button onClick={() => navigate('/login')} style={{ ...navBtnStyle, marginRight: 0, color: 'white', backgroundColor: '#333', padding: '10px 20px', borderRadius: '20px' }}>로그인</button>
          )}
        </div>
      </nav>

      <section style={{ ...sectionStyle, minHeight: '90vh' }}>
        <h1 style={{ fontSize: '64px', marginBottom: '25px', fontWeight: 'bold', lineHeight: '1.2' }}>
          Deep Guard는 사용자가 업로드한<br/>
          이미지 또는 비디오의 딥페이크 변조 여부와<br/>
          상세 정보를 제공합니다.
        </h1>
        <p style={{ color: '#aaa', fontSize: '22px', marginBottom: '20px' }}>
          지금 바로 분석을 시작해보세요.
        </p>
        <button 
          style={{ backgroundColor: '#39FF14', color: 'black', padding: '18px 50px', borderRadius: '40px', fontSize: '22px', fontWeight: 'bold', cursor: 'pointer', border: 'none', marginTop: '50px', boxShadow: '0 0 20px rgba(57, 255, 20, 0.3)' }} 
          onClick={handleProAnalysis}
        >
          분석 시작 ❯
        </button>
      </section>

      {/* 분석 카드 섹션 */}
      <section style={{ ...sectionStyle, backgroundColor: '#050505', minHeight: '100vh', flexDirection: 'row', gap: '40px' }}>
        
        {/* 이미지 분석 카드 */}
        <div style={analysisBoxStyle}>
          <h3 style={{ fontSize: '36px', color: 'white', marginBottom: '30px' }}>이미지 분석</h3>
          <img src={circle} alt="이미지 분석 도구" style={{ width: '180px', marginBottom: '40px' }} />
          <button style={basicBtnStyle} onClick={handleBasicAnalysis}>
            Basic 모델로 분석 <span>❯</span>
          </button>
          <button style={proBtnStyle} onClick={handleProAnalysis}>
            Pro 모델로 분석 <span>❯</span>
          </button>
        </div>

        {/* 비디오 분석 카드 */}
        <div style={analysisBoxStyle}>
          <h3 style={{ fontSize: '36px', color: 'white', marginBottom: '30px' }}>비디오 분석</h3>
          <img src={circle} alt="비디오 분석 도구" style={{ width: '180px', marginBottom: '40px' }} />
          <button style={basicBtnStyle} onClick={handleBasicAnalysis}>
            Basic 모델로 분석 <span>❯</span>
          </button>
          <button style={proBtnStyle} onClick={handleProAnalysis}>
            Pro 모델로 분석 <span>❯</span>
          </button>
        </div>

      </section>

      {/* 푸터 */}
      <footer style={{ padding: '80px', textAlign: 'center', backgroundColor: '#000', borderTop: '1px solid #222', zIndex: 2, position: 'relative' }}>
        <h2 onClick={handleHomeRedirect} style={{ color: '#39FF14', marginBottom: '20px', fontSize: '24px', cursor: 'pointer' }}>Deep Guard</h2>
        <p style={{ color: '#555', fontSize: '16px' }}>
          © 2026 Deep Guard. All rights reserved. <br/>
          딥페이크 탐지 시스템 | 충북대학교 소프트웨어학부
        </p>
      </footer>
    </div>
  );
};

export default MainPage;
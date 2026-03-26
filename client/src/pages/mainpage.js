import React from 'react';
import { useNavigate } from 'react-router-dom';

// 확장자를 .png에서 .svg로 수정했습니다.
import logo from '../assets/logo.svg'; 
import circle from '../assets/circle.svg'; // 분석 영역 이미지 (기존 analysisVisual 대신 사용)
import bgCurve from '../assets/line.svg'; 

const MainPage = () => {
  const navigate = useNavigate();

  // 스타일 정의 (기존과 동일)
  const containerStyle = {
    backgroundColor: '#000000',
    minHeight: '200vh',
    width: '100vw',
    color: 'white',
    fontFamily: 'sans-serif',
    display: 'flex',
    flexDirection: 'column',
    position: 'relative',
    overflowX: 'hidden',
  };

  const navStyle = {
    position: 'sticky',
    top: 0,
    width: '100%',
    padding: '20px 80px',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
    backdropFilter: 'blur(10px)',
    zIndex: 100,
    boxSizing: 'border-box',
    borderBottom: '1px solid #222'
  };

  const sectionStyle = {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: '120px 20px',
    textAlign: 'center',
    position: 'relative',
    zIndex: 2
  };

  const analysisBoxStyle = {
    width: '500px',
    height: '400px',
    backgroundColor: '#161616',
    borderRadius: '20px',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    padding: '40px',
    boxSizing: 'border-box',
    border: '1px solid #333'
  };

  const navBtnStyle = {
    color: '#39FF14',
    cursor: 'pointer',
    fontSize: '18px',
    fontWeight: 'bold',
    background: 'none',
    border: 'none',
    marginRight: '30px'
  };

  const startBtnStyle = {
    backgroundColor: '#39FF14',
    color: 'black',
    padding: '18px 50px',
    borderRadius: '40px',
    fontSize: '22px',
    fontWeight: 'bold',
    cursor: 'pointer',
    border: 'none',
    marginTop: '50px',
    boxShadow: '0 0 20px rgba(57, 255, 20, 0.3)'
  };

  return (
    <div style={containerStyle}>
      {/* 배경 곡선 문양 */}
      <img src={bgCurve} alt="" style={{
        position: 'absolute',
        bottom: '0',
        right: '0',
        width: '90%',
        opacity: 0.6,
        zIndex: 1
      }} />

      {/* 상단 네비게이션 */}
      <nav style={navStyle}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <img src={logo} alt="Deep Guard" style={{ height: '35px', marginRight: '50px' }} />
          <span style={{ color: '#aaa', marginRight: '30px', cursor: 'pointer' }}>시스템 특징</span>
          <span style={{ color: '#aaa', marginRight: '30px', cursor: 'pointer' }}>사용 방법</span>
          <span style={{ color: '#aaa', marginRight: '30px', cursor: 'pointer' }}>FAQ</span>
        </div>
        <div>
          <button style={navBtnStyle}>이미지 분석</button>
          <button style={navBtnStyle}>동영상 분석</button>
          <button 
            onClick={() => navigate('/login')} 
            style={{ ...navBtnStyle, marginRight: 0, color: 'white', backgroundColor: '#333', padding: '10px 20px', borderRadius: '20px' }}
          >
            로그인
          </button>
        </div>
      </nav>

      {/* 메인 섹션 */}
      <section style={{ ...sectionStyle, minHeight: '90vh' }}>
        <h1 style={{ fontSize: '64px', marginBottom: '25px', fontWeight: 'bold', lineHeight: '1.2' }}>
          Deep Guard는 사용자가 업로드한<br/>
          이미지 또는 비디오의 딥페이크 변조 여부와<br/>
          상세 정보를 제공합니다.
        </h1>
        <p style={{ color: '#aaa', fontSize: '22px', marginBottom: '20px' }}>
          지금 바로 분석을 시작해보세요.
        </p>
        <button style={startBtnStyle}>분석 시작 ❯</button>
      </section>

      {/* 분석 섹션 */}
      <section style={{ ...sectionStyle, backgroundColor: '#050505', minHeight: '100vh', flexDirection: 'row', justifyContent: 'center', gap: '60px' }}>
        <div style={analysisBoxStyle}>
  <h3 style={{ fontSize: '32px', color: '#39FF14', marginBottom: '20px' }}>이미지 분석</h3>
  <p style={{ color: '#aaa', marginBottom: '30px', fontSize: '18px' }}>
    다양한 포맷의 이미지 파일을 분석합니다.
  </p>
  {/* alt에서 Image 단어를 삭제했습니다. */}
  <img src={circle} alt="Detailed Analysis View" style={{ width: '250px', marginBottom: '30px' }} />
  <button style={{ ...startBtnStyle, fontSize: '18px', padding: '12px 30px', marginTop: 0 }}>업로드 ❯</button>
</div>

        <div style={analysisBoxStyle}>
          <h3 style={{ fontSize: '32px', color: '#39FF14', marginBottom: '20px' }}>동영상 분석</h3>
          <p style={{ color: '#aaa', marginBottom: '30px', fontSize: '18px' }}>
            영상의 변조 구간을 상세히 탐지합니다.
          </p>
          <img src={circle} alt="Video Analysis" style={{ width: '250px', marginBottom: '30px' }} />
          <button style={{ ...startBtnStyle, fontSize: '18px', padding: '12px 30px', marginTop: 0 }}>업로드 ❯</button>
        </div>
      </section>

      {/* 하단 푸터 */}
      <footer style={{ padding: '80px', textAlign: 'center', backgroundColor: '#000', borderTop: '1px solid #222', zIndex: 2, position: 'relative' }}>
        <h2 style={{ color: '#39FF14', marginBottom: '20px', fontSize: '24px' }}>Deep Guard</h2>
        <p style={{ color: '#555', fontSize: '16px', lineHeight: '1.6' }}>
          © 2026 Deep Guard. All rights reserved. <br/>
          딥페이크 탐지 시스템 | 충북대학교 소프트웨어학부
        </p>
      </footer>
    </div>
  );
};

export default MainPage;
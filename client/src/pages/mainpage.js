import React from 'react';
import { useNavigate } from 'react-router-dom';
import logo from '../assets/logo.svg'; 
import circle from '../assets/circle.svg';
import bgCurve from '../assets/line.svg'; 

const MainPage = ({ sessionUser, onLogout }) => {
  const navigate = useNavigate();

  const handleBasicAnalysis = () => navigate('/analysis');
  const handleProAnalysis = () => {
    if (sessionUser) navigate('/analysis');
    else {
      alert("Pro 모델 분석은 로그인이 필요합니다.");
      navigate('/login');
    }
  };

  const containerStyle = { backgroundColor: '#000000', minHeight: '200vh', width: '100vw', color: 'white', fontFamily: 'sans-serif', display: 'flex', flexDirection: 'column', position: 'relative', overflowX: 'hidden' };
  const navStyle = { position: 'sticky', top: 0, width: '100%', padding: '20px 80px', display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: 'rgba(0, 0, 0, 0.9)', backdropFilter: 'blur(10px)', zIndex: 100, boxSizing: 'border-box', borderBottom: '1px solid #222' };
  const sectionStyle = { display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '120px 20px', textAlign: 'center', position: 'relative', zIndex: 2 };
  const analysisBoxStyle = { width: '450px', height: '520px', backgroundColor: '#111', borderRadius: '24px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', padding: '40px', boxSizing: 'border-box', border: '1px solid #222' };
  const modelBtnBase = { width: '100%', padding: '14px 24px', borderRadius: '30px', fontSize: '16px', fontWeight: 'bold', cursor: 'pointer', border: 'none', display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '12px' };

  return (
    <div style={containerStyle}>
      <img src={bgCurve} alt="" style={{ position: 'absolute', bottom: '0', right: '0', width: '90%', opacity: 0.6, zIndex: 1 }} />
      <nav style={navStyle}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <img src={logo} alt="Deep Guard" style={{ height: '35px', marginRight: '50px', cursor: 'pointer' }} onClick={() => navigate('/main')} />
          {sessionUser && <span style={{ color: '#39FF14', fontWeight: 'bold', fontSize: '18px' }}>{sessionUser.name}님 환영합니다</span>}
        </div>
        <div>
          <button style={{color:'#39FF14', cursor:'pointer', background:'none', border:'none', marginRight:'30px', fontWeight:'bold'}} onClick={handleBasicAnalysis}>이미지 분석</button>
          <button style={{color:'#39FF14', cursor:'pointer', background:'none', border:'none', marginRight:'30px', fontWeight:'bold'}} onClick={handleBasicAnalysis}>동영상 분석</button>
          {sessionUser ? (
            <button onClick={onLogout} style={{ color: 'white', backgroundColor: '#FF4B4B', padding: '10px 20px', borderRadius: '20px', border:'none', cursor:'pointer', fontWeight:'bold' }}>로그아웃</button>
          ) : (
            <button onClick={() => navigate('/login')} style={{ color: 'white', backgroundColor: '#333', padding: '10px 20px', borderRadius: '20px', border:'none', cursor:'pointer', fontWeight:'bold' }}>로그인</button>
          )}
        </div>
      </nav>

      <section style={{ ...sectionStyle, minHeight: '90vh' }}>
        <h1 style={{ fontSize: '64px', marginBottom: '25px', fontWeight: 'bold', lineHeight: '1.2' }}>
          Deep Guard는 사용자가 업로드한<br/>이미지 또는 비디오의 딥페이크 변조 여부와<br/>상세 정보를 제공합니다.
        </h1>
        <button style={{ backgroundColor: '#39FF14', color: 'black', padding: '18px 50px', borderRadius: '40px', fontSize: '22px', fontWeight: 'bold', cursor: 'pointer', border: 'none', marginTop: '50px' }} onClick={handleProAnalysis}>분석 시작 ❯</button>
      </section>

      <section style={{ ...sectionStyle, backgroundColor: '#050505', minHeight: '100vh', flexDirection: 'row', gap: '40px' }}>
        <div style={analysisBoxStyle}>
          <h3 style={{ fontSize: '36px', color: 'white', marginBottom: '30px' }}>이미지 분석</h3>
          <img src={circle} alt="분석 도구" style={{ width: '180px', marginBottom: '40px' }} />
          <button style={{...modelBtnBase, backgroundColor:'#2A2A2A', color:'white'}} onClick={handleBasicAnalysis}>Basic 모델로 분석 <span>❯</span></button>
          <button style={{...modelBtnBase, backgroundColor:'#1E1E1E', color:'#888'}} onClick={handleProAnalysis}>Pro 모델로 분석 <span>❯</span></button>
        </div>
        <div style={analysisBoxStyle}>
          <h3 style={{ fontSize: '36px', color: 'white', marginBottom: '30px' }}>비디오 분석</h3>
          <img src={circle} alt="비디오 분석 도구" style={{ width: '180px', marginBottom: '40px' }} />
          <button style={{...modelBtnBase, backgroundColor:'#2A2A2A', color:'white'}} onClick={handleBasicAnalysis}>Basic 모델로 분석 <span>❯</span></button>
          <button style={{...modelBtnBase, backgroundColor:'#1E1E1E', color:'#888'}} onClick={handleProAnalysis}>Pro 모델로 분석 <span>❯</span></button>
        </div>
      </section>
    </div>
  );
};
export default MainPage;
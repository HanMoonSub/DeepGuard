import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

const AnalysisDetailPage = ({ sessionUser }) => {
  const { state: data } = useLocation();
  const navigate = useNavigate();

  // 1. 데이터 자체가 없는 경우에 대한 확실한 방어
  if (!data) {
    return (
      <div style={{ backgroundColor: '#000', height: '100vh', display: 'flex', justifyContent: 'center', alignItems: 'center', color: '#fff' }}>
        <p>기록을 찾을 수 없습니다.</p>
        <button onClick={() => navigate('/analysis')} style={{ marginLeft: '10px', color: '#39FF14', background: 'none', border: 'none', cursor: 'pointer', textDecoration: 'underline' }}>돌아가기</button>
      </div>
    );
  }

  // 2. 데이터 구조 분해 할당 및 기본값 설정 (undefined 에러 방지 핵심)
  // 데이터가 없을 경우를 대비해 기본값 0을 설정합니다.
  const analysis = data.analysis || data || {}; 
  const prob = analysis.prob ?? 0;
  const face_conf = analysis.face_conf ?? analysis.conf ?? 0;
  const face_ratio = analysis.face_ratio ?? 0;
  const face_brightness = analysis.face_brightness ?? 0;

  const isUnknown = prob === -1;
  const brightnessColor = face_brightness < 20 ? '#FF4B4B' : '#39FF14';
  const ratioColor = face_ratio >= 3 ? '#39FF14' : '#FF4B4B';

  // 스타일 정의 (기존과 동일)
  const containerStyle = { backgroundColor: '#000', minHeight: '100vh', width: '100vw', color: 'white', fontFamily: "'Inter', sans-serif", padding: '40px 80px', boxSizing: 'border-box' };
  const headerStyle = { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '40px' };
  const mainContentStyle = { display: 'flex', gap: '40px', maxWidth: '1400px', margin: '0 auto' };
  const imageCardStyle = { flex: 1.2, backgroundColor: '#0A0A0A', borderRadius: '28px', border: '1px solid #1A1A1A', display: 'flex', justifyContent: 'center', alignItems: 'center', overflow: 'hidden', minHeight: '600px' };
  const infoPanelStyle = { flex: 0.8, display: 'flex', flexDirection: 'column', gap: '24px' };
  const resultCardStyle = { padding: '40px', backgroundColor: '#0D0D0D', borderRadius: '28px', border: '1px solid #1A1A1A', textAlign: 'center' };
  const metricBoxStyle = { padding: '24px', backgroundColor: '#0D0D0D', borderRadius: '20px', border: '1px solid #1A1A1A' };

  return (
    <div style={containerStyle}>
      <header style={headerStyle}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <button onClick={() => navigate(-1)} style={{ color: '#888', background: 'none', border: 'none', cursor: 'pointer', fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px', marginRight: '20px' }}>
            <span style={{ fontSize: '20px' }}>←</span> 뒤로가기
          </button>
        </div>
        {sessionUser && (
          <div style={{ backgroundColor: '#111', padding: '8px 16px', borderRadius: '12px', border: '1px solid #222' }}>
            <span style={{ color: '#888', fontSize: '13px' }}>분석 담당</span>
            <span style={{ color: '#39FF14', fontWeight: 'bold', marginLeft: '8px' }}>{sessionUser.name}</span>
          </div>
        )}
      </header>

      <div style={mainContentStyle}>
        <div style={imageCardStyle}>
          {/* 이미지 경로 처리 보안 */}
          <img 
            src={data.image_loc ? (data.image_loc.startsWith('blob') ? data.image_loc : `http://localhost:8000${data.image_loc}`) : ''} 
            alt="Analyzed" 
            style={{ maxWidth: '90%', maxHeight: '90%', objectFit: 'contain', borderRadius: '12px' }} 
          />
        </div>

        <div style={infoPanelStyle}>
          <div style={resultCardStyle}>
            <p style={{ color: '#555', fontSize: '14px', letterSpacing: '2px', marginBottom: '8px' }}>Final Analysis</p>
            <h1 style={{ fontSize: '72px', fontWeight: '900', color: data.label === 'FAKE' ? '#FF4B4B' : (isUnknown ? '#444' : '#39FF14'), margin: '0' }}>
              {data.label || 'PENDING'}
            </h1>
            <div style={{ marginTop: '20px' }}>
              <p style={{ fontSize: '28px', fontWeight: '600', color: '#FFF', margin: '0' }}>
                {/* .toFixed() 실행 전 숫자인지 한 번 더 체크 */}
                {isUnknown ? 'N/A' : (Number(prob) * 100).toFixed(1) + '%'}
              </p>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
            <div style={metricBoxStyle}>
              <p style={{ color: '#555', fontSize: '12px', marginBottom: '12px', fontWeight: '600' }}>BRIGHTNESS</p>
              <h2 style={{ fontSize: '32px', color: brightnessColor, margin: '0' }}>
                {Number(face_brightness).toFixed(1)}%
              </h2>
            </div>
            
            <div style={metricBoxStyle}>
              <p style={{ color: '#555', fontSize: '12px', marginBottom: '12px', fontWeight: '600' }}>FACE RATIO</p>
              <h2 style={{ fontSize: '32px', color: ratioColor, margin: '0' }}>
                {Number(face_ratio).toFixed(1)}%
              </h2>
            </div>
            
            <div style={{ ...metricBoxStyle, gridColumn: 'span 2' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '12px' }}>
                <p style={{ color: '#555', fontSize: '12px', fontWeight: '600', margin: '0' }}>MODEL CONFIDENCE</p>
                <span style={{ color: '#39FF14', fontSize: '20px', fontWeight: 'bold' }}>
                  {Number(face_conf).toFixed(1)}%
                </span>
              </div>
              <div style={{ width: '100%', height: '6px', backgroundColor: '#1A1A1A', borderRadius: '3px', overflow: 'hidden' }}>
                <div style={{ width: `${face_conf}%`, height: '100%', backgroundColor: '#39FF14', transition: 'width 1s' }} />
              </div>
              <p style={{ fontSize: '14px', color: '#888', marginTop: '20px', lineHeight: '1.6' }}>
                {data.message || "Deep Guard AI가 분석한 데이터입니다."}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalysisDetailPage;
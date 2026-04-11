import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

const AnalysisDetailPage = ({ sessionUser }) => {
  const { state: data } = useLocation();
  const navigate = useNavigate();

  if (!data) return <div style={{ color: 'white', padding: '50px' }}>기록을 찾을 수 없습니다.</div>;

  const context = data.context || {};
  const brightness = context.face_brightness || 0;
  const ratio = context.face_ratio || 0;
  const confidence = context.face_conf || context.conf || 0;

  return (
    <div style={{ backgroundColor: '#000', minHeight: '100vh', color: 'white', padding: '40px', fontFamily: 'sans-serif' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '30px', alignItems: 'center' }}>
        <button onClick={() => navigate(-1)} style={{ color: '#888', background: 'none', border: 'none', cursor: 'pointer', fontSize: '18px' }}>← 뒤로가기</button>
        {sessionUser && <span style={{ color: '#39FF14', fontWeight: 'bold' }}>분석 담당: {sessionUser.name}</span>}
      </header>

      <div style={{ display: 'flex', gap: '30px', maxWidth: '1200px', margin: '0 auto' }}>
        <div style={{ flex: 1.2, backgroundColor: '#050505', borderRadius: '24px', border: '1px solid #222', height: '600px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
          <img src={data.image_loc.startsWith('blob') ? data.image_loc : `http://localhost:8000${data.image_loc}`} alt="Analyzed" style={{ maxWidth: '95%', maxHeight: '95%', objectFit: 'contain' }} />
        </div>

        <div style={{ flex: 0.8, display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div style={{ padding: '30px', backgroundColor: '#0D0D0D', borderRadius: '24px', border: '1px solid #222', textAlign: 'center' }}>
            <p style={{ color: '#888' }}>Final Result</p>
            <h1 style={{ fontSize: '64px', color: data.label === 'FAKE' ? '#FF4B4B' : '#39FF14', margin: '10px 0' }}>{data.label}</h1>
            <p style={{ fontSize: '24px' }}>{(data.prob * 100).toFixed(1)}%</p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
            <div style={{ padding: '20px', backgroundColor: '#0D0D0D', borderRadius: '20px', border: '1px solid #222' }}>
              <p style={{ color: '#888', fontSize: '13px' }}>Brightness</p>
              <h2 style={{ color: brightness < 20 ? '#FF4B4B' : '#39FF14' }}>{brightness.toFixed(1)}%</h2>
            </div>
            <div style={{ padding: '20px', backgroundColor: '#0D0D0D', borderRadius: '20px', border: '1px solid #222' }}>
              <p style={{ color: '#888', fontSize: '13px' }}>Face Ratio</p>
              <h2 style={{ color: ratio < 3 ? '#FF4B4B' : '#39FF14' }}>{ratio.toFixed(1)}%</h2>
            </div>
            <div style={{ gridColumn: 'span 2', padding: '25px', backgroundColor: '#0D0D0D', borderRadius: '20px', border: '1px solid #222' }}>
              <p style={{ color: '#888', fontSize: '13px' }}>Model Confidence</p>
              <h2 style={{ color: '#39FF14' }}>{confidence.toFixed(1)}%</h2>
              <p style={{ fontSize: '14px', color: '#ccc', marginTop: '15px' }}>{data.message || "분석이 정상적으로 완료되었습니다."}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalysisDetailPage;
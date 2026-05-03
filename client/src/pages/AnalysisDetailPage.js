import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

const AnalysisDetailPage = ({ sessionUser }) => {
  const { state: data } = useLocation();
  const navigate = useNavigate();

  if (!data) return <div style={{ color: 'white', padding: '50px', textAlign: 'center' }}>데이터를 찾을 수 없습니다.</div>;

  /* 수치 추출 로직(백엔드 DB에서 사용하는 컬럼명과 API 응답 객체 이름을 모두 매핑하기) */
  const prob = data.prob ?? data.score ?? data.analysis?.prob ?? -1;
  
  // 백엔드 DB 컬럼명인 face_conf, face_ratio, face_brightness 최우선으로 체크
  const face_conf = data.face_conf ?? data.face_confidence ?? data.conf ?? data.analysis?.face_conf ?? 0;
  const face_ratio = data.face_ratio ?? data.ratio ?? data.analysis?.face_ratio ?? 0;
  const face_brightness = data.face_brightness ?? data.brightness ?? data.analysis?.face_brightness ?? 0;
  
  // 라벨 결정 로직
  let label = 'UNKNOWN';
  if (data.label && data.label !== 'UNKNOWN') {
    label = data.label.toUpperCase();
  } else if (prob !== -1) {
    label = prob > 0.5 ? 'FAKE' : 'REAL';
  }
  
  // 확률이 -1인 경우 (데이터 로드 실패) N/A 표시를 위해 isInvalid 설정
  const isInvalid = prob === -1;
  const displayProb = isInvalid ? 'N/A' : (Number(prob) * 100).toFixed(1) + '%';

  const brightnessColor = face_brightness < 20 ? '#FF4B4B' : '#39FF14';
  const ratioColor = face_ratio >= 3 ? '#39FF14' : '#FF4B4B';

  const mediaLoc = data.video_loc || data.image_loc || '';
  const isVideo = !!data.video_loc || (data.score !== undefined && !data.image_loc);

  return (
    <div style={{ backgroundColor: '#000', minHeight: '100vh', width: '100vw', color: 'white', padding: '40px 80px', boxSizing: 'border-box', fontFamily: 'sans-serif' }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '40px', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <button onClick={() => navigate(-1)} style={{ color: '#888', background: 'none', border: 'none', cursor: 'pointer', fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px', marginRight: '20px' }}>
            <span style={{ fontSize: '20px' }}>←</span> 뒤로가기
          </button>
          <span style={{ padding: '6px 14px', backgroundColor: '#111', borderRadius: '20px', fontSize: '12px', color: '#39FF14', border: '1px solid #333', fontWeight: 'bold' }}>
            {data.version_type?.toUpperCase() || 'V1'}
          </span>
          <span style={{ padding: '6px 14px', backgroundColor: '#111', borderRadius: '20px', fontSize: '12px', color: '#39FF14', border: '1px solid #333', fontWeight: 'bold', marginLeft: '8px' }}>
            {data.domain_type || '서양인'}
          </span>
        </div>
        {sessionUser && (
          <div style={{ backgroundColor: '#111', padding: '8px 16px', borderRadius: '12px', border: '1px solid #222' }}>
            <span style={{ color: '#39FF14', fontWeight: 'bold' }}>분석 담당: {sessionUser.name}</span>
          </div>
        )}
      </header>

      <div style={{ display: 'flex', gap: '40px', maxWidth: '1400px', margin: '0 auto', alignItems: 'stretch' }}>
        <div style={{ flex: 1.2, backgroundColor: '#050505', borderRadius: '28px', border: '1px solid #1A1A1A', height: '600px', display: 'flex', justifyContent: 'center', alignItems: 'center', overflow: 'hidden', boxShadow: '0 10px 30px rgba(0,0,0,0.5)' }}>
          {isVideo && mediaLoc ? (
             <video src={mediaLoc.startsWith('blob') ? mediaLoc : `http://localhost:8000${mediaLoc}`} controls style={{ maxWidth: '90%', maxHeight: '95%', borderRadius: '12px' }} />
          ) : (
            <img 
              src={mediaLoc.startsWith('blob') ? mediaLoc : `http://localhost:8000${mediaLoc}`} 
              alt="Analyzed media" 
              style={{ maxWidth: '90%', maxHeight: '95%', objectFit: 'contain', borderRadius: '12px' }} 
              onError={(e) => { e.target.src = 'https://via.placeholder.com/600x400?text=No+Image'; }}
            />
          )}
        </div>

        <div style={{ flex: 0.8, display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <div style={{ padding: '40px', backgroundColor: '#0D0D0D', borderRadius: '28px', border: '1px solid #1A1A1A', textAlign: 'center' }}>
            <p style={{ color: '#555', fontSize: '14px', letterSpacing: '2px' }}>FINAL ANALYSIS</p>
            <h1 style={{ fontSize: '72px', fontWeight: '900', color: label === 'FAKE' ? '#FF4B4B' : (isInvalid ? '#444' : '#39FF14'), margin: '10px 0' }}>{label}</h1>
            <p style={{ fontSize: '28px', fontWeight: 'bold' }}>{displayProb}</p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
            <div style={{ padding: '24px', backgroundColor: '#0D0D0D', borderRadius: '20px', border: '1px solid #1A1A1A' }}>
              <p style={{ color: '#555', fontSize: '12px', fontWeight: 'bold' }}>BRIGHTNESS</p>
              <h2 style={{ fontSize: '32px', color: brightnessColor, margin: '0' }}>{Number(face_brightness).toFixed(1)}%</h2>
              <div style={{ width: '40px', height: '3px', backgroundColor: brightnessColor, marginTop: '12px', borderRadius: '2px' }} />
            </div>
            <div style={{ padding: '24px', backgroundColor: '#0D0D0D', borderRadius: '20px', border: '1px solid #1A1A1A' }}>
              <p style={{ color: '#555', fontSize: '12px', fontWeight: 'bold' }}>FACE RATIO</p>
              <h2 style={{ fontSize: '32px', color: ratioColor, margin: '0' }}>{Number(face_ratio).toFixed(1)}%</h2>
              <div style={{ width: '40px', height: '3px', backgroundColor: ratioColor, marginTop: '12px', borderRadius: '2px' }} />
            </div>
            <div style={{ gridColumn: 'span 2', padding: '24px', backgroundColor: '#0D0D0D', borderRadius: '20px', border: '1px solid #1A1A1A' }}>
              <p style={{ color: '#555', fontSize: '12px', fontWeight: 'bold' }}>MODEL CONFIDENCE</p>
              <h2 style={{ fontSize: '32px', color: '#39FF14', margin: '0' }}>{Number(face_conf).toFixed(1)}%</h2>
              <div style={{ width: '100%', height: '6px', backgroundColor: '#1A1A1A', borderRadius: '3px', marginTop: '15px', overflow: 'hidden' }}>
                <div style={{ width: `${face_conf}%`, height: '100%', backgroundColor: '#39FF14', transition: 'width 1s' }} />
              </div>
              <p style={{ fontSize: '14px', color: '#888', marginTop: '20px' }}>{data.message || (isInvalid ? "분석 데이터를 불러오지 못했습니다." : "정상 분석 리포트입니다.")}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalysisDetailPage;
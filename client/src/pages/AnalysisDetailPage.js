import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
const apiUrl = process.env.REACT_APP_API_URL;

const AnalysisDetailPage = ({ sessionUser }) => {
  const { state: data } = useLocation();
  const navigate = useNavigate();

  if (!data) return <div style={{ color: 'white', padding: '50px', textAlign: 'center' }}>데이터를 찾을 수 없습니다.</div>;

  /**
   * [수치 추출 로직 보완]
   * 1. prob: 이미지 분석 확률 (0.0~1.0)
   * 2. score: 비디오 분석 확률 (0.0~1.0)
   * 3. data.analysis?.prob: 분석 직후 넘어온 객체 내부 값
   */
  const prob = data.prob ?? data.score ?? data.analysis?.prob ?? -1;
  
  // 얼굴 인식 신뢰도 및 기타 지표
  const face_conf = data.face_conf ?? data.analysis?.face_conf ?? data.analysis?.conf ?? data.conf ?? 0;
  const face_ratio = data.face_ratio ?? data.analysis?.face_ratio ?? 0;
  const face_brightness = data.face_brightness ?? data.analysis?.face_brightness ?? 0;
  
  const isUnknown = prob === -1;
  const label = isUnknown ? 'UNKNOWN' : (data.label || (prob > 0.5 ? 'FAKE' : 'REAL'));

  const brightnessColor = face_brightness < 20 ? '#FF4B4B' : '#39FF14';
  const ratioColor = face_ratio >= 3 ? '#39FF14' : '#FF4B4B';

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
        {/* 미디어 표시 영역 */}
        <div style={{ flex: 1.2, backgroundColor: '#050505', borderRadius: '28px', border: '1px solid #1A1A1A', height: '600px', display: 'flex', justifyContent: 'center', alignItems: 'center', overflow: 'hidden', boxShadow: '0 10px 30px rgba(0,0,0,0.5)' }}>
          {/* 비디오 결과이면서 video_loc가 있을 경우 비디오 태그, 그 외엔 이미지 태그 */}
          {data.score !== undefined && data.image_loc ? (
             <video src={data.image_loc.startsWith('blob') ? data.image_loc : `${apiUrl}${data.image_loc}`} controls style={{ maxWidth: '90%', maxHeight: '95%', borderRadius: '12px' }} />
          ) : (
            <img 
              src={data.image_loc?.startsWith('blob') ? data.image_loc : `${apiUrl}${data.image_loc}`} 
              alt="Analyzed media" 
              style={{ maxWidth: '90%', maxHeight: '95%', objectFit: 'contain', borderRadius: '12px' }} 
              onError={(e) => { e.target.src = 'https://via.placeholder.com/600x400?text=No+Image'; }}
            />
          )}
        </div>

        {/* 결과 수치 영역 */}
        <div style={{ flex: 0.8, display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <div style={{ padding: '40px', backgroundColor: '#0D0D0D', borderRadius: '28px', border: '1px solid #1A1A1A', textAlign: 'center' }}>
            <p style={{ color: '#555', fontSize: '14px', letterSpacing: '2px' }}>FINAL ANALYSIS</p>
            <h1 style={{ fontSize: '72px', fontWeight: '900', color: label === 'FAKE' ? '#FF4B4B' : (isUnknown ? '#444' : '#39FF14'), margin: '10px 0' }}>{label}</h1>
            <p style={{ fontSize: '28px', fontWeight: 'bold' }}>{isUnknown ? 'N/A' : (Number(prob) * 100).toFixed(1) + '%'}</p>
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
              <p style={{ fontSize: '14px', color: '#888', marginTop: '20px' }}>{data.message || (isUnknown ? "분석에 실패하였습니다." : "정상 분석 리포트입니다.")}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalysisDetailPage;
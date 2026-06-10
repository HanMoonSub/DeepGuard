import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';


const AnalysisDetailPage = ({ sessionUser }) => {
  const { state } = useLocation();
  const navigate = useNavigate();

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchDetail = async () => {
      const imageId = state?.image_id;
      const videoId = state?.video_id;

      if (!imageId && !videoId) { setLoading(false); return; }

      try {
        if (imageId) {
          const res = await axios.get(`/image/history/${imageId}`);
          setData(res.data.context);
        } else {
          const res = await axios.get(`/video/history/${videoId}`);
          setData(res.data.context);
        }
      } catch (e) {
        console.error("상세 조회 실패", e);
      } finally {
        setLoading(false);
      }
    };
    fetchDetail();
  }, [state]);

  if (loading) {
    return (
      <div style={{ backgroundColor: '#000', minHeight: '100vh', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', color: 'white' }}>
        <p style={{ fontSize: '18px', marginBottom: '20px' }}>불러오는 중...</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div style={{ backgroundColor: '#000', minHeight: '100vh', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', color: 'white' }}>
        <p style={{ fontSize: '18px', marginBottom: '20px' }}>분석 데이터를 찾을 수 없거나 비정상적인 접근입니다.</p>
        <button onClick={() => navigate(-1)} style={{ padding: '10px 20px', backgroundColor: '#1A2C50', color: '#39FF14', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }}>돌아가기</button>
      </div>
    );
  }

  const prob = data.prob ?? data.score ?? data.analysis?.prob ?? data.result_prob ?? -1;
  const face_conf = data.face_conf ?? data.face_confidence ?? data.conf ?? data.analysis?.face_conf ?? 0;
  const face_ratio = data.face_ratio ?? data.ratio ?? data.analysis?.face_ratio ?? 0;
  const face_brightness = data.face_brightness ?? data.brightness ?? data.analysis?.face_brightness ?? 0;
  
  let label = 'UNKNOWN';
  if (data.label && data.label !== 'UNKNOWN') {
    label = data.label.toUpperCase();
  } else if (prob !== -1) {
    label = prob > 0.5 ? 'FAKE' : 'REAL';
  }
  
  const isInvalid = prob === -1;
  const displayProb = isInvalid ? 'N/A' : (Number(prob) * 100).toFixed(1) + '%';

  const brightnessColor = face_brightness < 20 ? '#FF4B4B' : '#39FF14';
  const ratioColor = face_ratio >= 3 ? '#39FF14' : '#FF4B4B';

  const mediaLoc = data.video_loc || data.media_loc || data.image_loc || '';
  const mediaSrc = mediaLoc.startsWith('blob') ? mediaLoc : mediaLoc;
  const isVideo = !!data.video_loc || (data.score !== undefined && !data.image_loc) || window.location.pathname.includes('video');

  // [수정] WARNING 여부 판단
  const isWarning = data.status?.toUpperCase() === 'WARNING';

  return (
    <div style={{ backgroundColor: '#000', minHeight: '100vh', width: '100vw', color: 'white', padding: '40px 80px', boxSizing: 'border-box', fontFamily: 'sans-serif', overflowX: 'hidden' }}>
      
      {/* 상단 네비게이션 헤더 바 */}
      <header style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '40px', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <button onClick={() => navigate(-1)} style={{ color: '#888', background: 'none', border: 'none', cursor: 'pointer', fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px', marginRight: '20px' }}>
            <span style={{ fontSize: '20px' }}>←</span> 뒤로가기
          </button>
          <span style={{ padding: '6px 14px', backgroundColor: '#111', borderRadius: '20px', fontSize: '12px', color: '#39FF14', border: '1px solid #333', fontWeight: 'bold' }}>{data.version_type?.toUpperCase() || 'V1'}</span>
          <span style={{ padding: '6px 14px', backgroundColor: '#111', borderRadius: '20px', fontSize: '12px', color: '#39FF14', border: '1px solid #333', fontWeight: 'bold' }}>{data.domain_type || '서양인'}</span>
          <span style={{ padding: '6px 14px', backgroundColor: '#111', borderRadius: '20px', fontSize: '12px', color: '#39FF14', border: '1px solid #333', fontWeight: 'bold' }}>{data.model_type?.toUpperCase() || 'FAST'}</span>
        </div>
        {sessionUser && (
          <div style={{ backgroundColor: '#111', padding: '8px 16px', borderRadius: '12px', border: '1px solid #222', fontSize: '14px' }}>
            <span style={{ color: '#39FF14', fontWeight: 'bold' }}>분석 담당: {sessionUser.name}</span>
          </div>
        )}
      </header>

      {/* 2분할 메인 그리드 레이아웃 */}
      <div style={{ display: 'flex', gap: '40px', maxWidth: '1400px', margin: '0 auto', alignItems: 'stretch', minHeight: '580px' }}>
        
        {/* 왼쪽 섹션: 미디어 플레이어 전용 배치 존 */}
        <div style={{ flex: 1.2, backgroundColor: '#050505', borderRadius: '28px', border: '1px solid #1A1A1A', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', overflow: 'hidden', padding: '28px', boxSizing: 'border-box', boxShadow: '0 10px 30px rgba(0,0,0,0.5)', position: 'relative' }}>
          <div style={{ flex: 1, width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center', overflow: 'hidden' }}>
            {isVideo && mediaLoc ? (
              <video src={mediaSrc} controls autoPlay muted style={{ maxWidth: '95%', maxHeight: '95%', borderRadius: '12px', objectFit: 'contain' }} />
            ) : mediaLoc ? (
              <img src={mediaSrc} alt="Analyzed media" style={{ maxWidth: '95%', maxHeight: '95%', objectFit: 'contain', borderRadius: '12px' }} onError={(e) => { e.target.src = 'https://via.placeholder.com/600x400?text=No+Image'; }} />
            ) : (
              <div style={{ color: '#999', textAlign: 'center' }}>미디어가 존재하지 않습니다.</div>
            )}
          </div>

          {isVideo && mediaLoc && (
            <button 
              onClick={() => navigate('/video-timeline', { state: { ...data } })}
              style={{ marginTop: '20px', width: '95%', padding: '15px 0', backgroundColor: 'transparent', color: '#39FF14', border: '1px solid rgba(57, 255, 20, 0.4)', borderRadius: '6px', fontSize: '13px', fontWeight: '700', letterSpacing: '1.5px', textTransform: 'uppercase', cursor: 'pointer', transition: 'all 0.2s ease-in-out', boxShadow: '0 2px 8px rgba(57, 255, 20, 0.05)' }}
              onMouseEnter={(e) => { e.target.style.backgroundColor = 'rgba(57, 255, 20, 0.06)'; e.target.style.borderColor = '#39FF14'; e.target.style.boxShadow = '0 0 15px rgba(57, 255, 20, 0.15)'; }}
              onMouseLeave={(e) => { e.target.style.backgroundColor = 'transparent'; e.target.style.borderColor = 'rgba(57, 255, 20, 0.4)'; e.target.style.boxShadow = '0 2px 8px rgba(57, 255, 20, 0.05)'; }}
            >
              Forgery Localization: Heatmap + BBox
            </button>
          )}

          {!isVideo && mediaLoc && !isWarning && (
            <button
              onClick={() => navigate('/image-heatmap', {
                state: {
                  image_id: data.image_id,
                  image_loc: data.image_loc,
                  model_type: data.model_type || 'fast',
                  prob,
                  label,
                }
              })}
              style={{ marginTop: '20px', width: '95%', padding: '15px 0', backgroundColor: 'transparent', color: '#39FF14', border: '1px solid rgba(57, 255, 20, 0.4)', borderRadius: '6px', fontSize: '13px', fontWeight: '700', letterSpacing: '1.5px', textTransform: 'uppercase', cursor: 'pointer', transition: 'all 0.2s ease-in-out', boxShadow: '0 2px 8px rgba(57, 255, 20, 0.05)' }}
              onMouseEnter={(e) => { e.target.style.backgroundColor = 'rgba(57, 255, 20, 0.06)'; e.target.style.borderColor = '#39FF14'; e.target.style.boxShadow = '0 0 15px rgba(57, 255, 20, 0.15)'; }}
              onMouseLeave={(e) => { e.target.style.backgroundColor = 'transparent'; e.target.style.borderColor = 'rgba(57, 255, 20, 0.4)'; e.target.style.boxShadow = '0 2px 8px rgba(57, 255, 20, 0.05)'; }}
            >
              Forgery Localization: Heatmap + BBox
            </button>
          )}
        </div>

        {/* 오른쪽 섹션: 스코어 보드 or WARNING 메시지 */}
        <div style={{ flex: 0.8, display: 'flex', flexDirection: 'column', gap: '20px' }}>

          {isWarning ? (
            // WARNING일 때 — 경고 메시지만, 꽉 채움
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', padding: '40px', backgroundColor: '#0D0D0D', borderRadius: '28px', border: '1px solid #1A1A1A', textAlign: 'center', gap: '20px' }}>
              <p style={{ fontSize: '48px', margin: '0' }}>⚠</p>
              <p style={{ fontSize: '20px', fontWeight: 'bold', color: '#FFA500', margin: '0', letterSpacing: '2px' }}>UNDETECTED</p>
              <p style={{ fontSize: '14px', color: '#666', lineHeight: '1.8', margin: '0', maxWidth: '280px' }}>
                {data.result_msg || data.message || "얼굴을 탐지하지 못했습니다."}
              </p>
            </div>
          ) : (
            <>
              {/* 종합 리포트 판정 카드 — flex:1로 남은 공간 채움 */}
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', padding: '40px', backgroundColor: '#0D0D0D', borderRadius: '28px', border: '1px solid #1A1A1A', textAlign: 'center', gap: '8px' }}>
                <p style={{ color: '#aaa', fontSize: '13px', letterSpacing: '2px', margin: '0' }}>FINAL ANALYSIS</p>
                <h1 style={{ fontSize: '84px', fontWeight: '900', color: label === 'FAKE' ? '#FF4B4B' : (isInvalid ? '#999' : '#39FF14'), margin: '8px 0 0', letterSpacing: '1px', lineHeight: 1 }}>
                  {label}
                </h1>
                <div style={{ width: '40px', height: '2px', backgroundColor: label === 'FAKE' ? '#FF4B4B' : '#39FF14', borderRadius: '1px', margin: '12px 0' }} />
                <p style={{ fontSize: '12px', color: '#bbb', letterSpacing: '1.5px', margin: '0', fontWeight: 'bold' }}>FAKE 확률</p>
                <p style={{ fontSize: '48px', fontWeight: '900', margin: '0', color: label === 'FAKE' ? '#FF4B4B' : '#fff', letterSpacing: '1px', lineHeight: 1 }}>{displayProb}</p>
              </div>

              {/* 보조 메트릭 — 작게 고정 높이 */}
              <div style={{ padding: '20px 24px', backgroundColor: '#0D0D0D', borderRadius: '20px', border: '1px solid #1A1A1A' }}>
                <p style={{ color: '#999', fontSize: '11px', fontWeight: 'bold', letterSpacing: '1.5px', margin: '0 0 14px 0' }}>DETAIL METRICS</p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>

                  {/* BRIGHTNESS */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ color: '#999', fontSize: '12px', fontWeight: 'bold' }}>얼굴 밝기</span>
                    <span style={{ color: brightnessColor, fontSize: '14px', fontWeight: 'bold', fontFamily: 'monospace' }}>{Number(face_brightness).toFixed(1)}%</span>
                  </div>

                  {/* FACE RATIO */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ color: '#999', fontSize: '12px', fontWeight: 'bold' }}>얼굴 비율</span>
                    <span style={{ color: ratioColor, fontSize: '14px', fontWeight: 'bold', fontFamily: 'monospace' }}>{Number(face_ratio).toFixed(1)}%</span>
                  </div>

                  {/* MODEL CONFIDENCE */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                    <span style={{ color: '#999', fontSize: '12px', fontWeight: 'bold' }}>모델 신뢰도</span>
                    <span style={{ color: '#39FF14', fontSize: '14px', fontWeight: 'bold', fontFamily: 'monospace' }}>{Number(face_conf).toFixed(1)}%</span>
                  </div>
                  <div style={{ width: '100%', height: '4px', backgroundColor: '#1A1A1A', borderRadius: '2px', overflow: 'hidden' }}>
                    <div style={{ width: `${face_conf}%`, height: '100%', backgroundColor: '#39FF14', transition: 'width 1s cubic-bezier(0.1, 1, 0.1, 1)' }} />
                  </div>

                </div>
              </div>
            </>
          )}
        </div>

      </div>
    </div>
  );
};

export default AnalysisDetailPage;
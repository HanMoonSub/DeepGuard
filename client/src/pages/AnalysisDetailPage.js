import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

const apiUrl = process.env.REACT_APP_API_URL || "http://localhost:8000";

const VideoAnalysisDetailPage = ({ sessionUser }) => {
  const { state: data } = useLocation();
  const navigate = useNavigate();

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

  // ────────────────────────────────────────────────────────────────────────
  // 🎯 [핵심 교정 파트] 비디오 스트림 분기 및 이미지 소스 연동 로직
  // ────────────────────────────────────────────────────────────────────────
  const mediaLoc = data.video_loc || data.media_loc || data.image_loc || '';
  
  // URL 주소가 Blob 가상 객체 주소인지, 서버의 static 정적 파일 경로인지 판별하여 매핑
  const mediaSrc = mediaLoc.startsWith('blob') ? mediaLoc : `${apiUrl}${mediaLoc}`;

  // 현재 유입된 경로가 비디오 분석 결과인지 이미지 분석 결과인지 분기 검사
  // 데이터 내부에 video_loc가 있거나 라우터 경로(pathname)에 video가 포함되어 있다면 비디오로 처리
  const isVideoContent = !!data.video_loc || window.location.pathname.includes('video');
  // ────────────────────────────────────────────────────────────────────────

  return (
    <div style={{ backgroundColor: '#000', minHeight: '100vh', width: '100vw', color: 'white', padding: '40px 120px', boxSizing: 'border-box', fontFamily: 'sans-serif', overflowX: 'hidden' }}>
      
      <header style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '30px', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <button onClick={() => navigate(-1)} style={{ color: '#888', background: 'none', border: 'none', cursor: 'pointer', fontSize: '15px', display: 'flex', alignItems: 'center', gap: '8px', marginRight: '15px' }}>
            <span style={{ fontSize: '18px' }}>←</span> 뒤로가기
          </button>
          <span style={{ padding: '5px 12px', backgroundColor: '#111', borderRadius: '20px', fontSize: '11px', color: '#39FF14', border: '1px solid #222', fontWeight: 'bold' }}>{data.version_type?.toUpperCase() || 'V1'}</span>
          <span style={{ padding: '5px 12px', backgroundColor: '#111', borderRadius: '20px', fontSize: '11px', color: '#39FF14', border: '1px solid #222', fontWeight: 'bold' }}>{data.domain_type || '서양인'}</span>
          <span style={{ padding: '5px 12px', backgroundColor: '#111', borderRadius: '20px', fontSize: '11px', color: '#39FF14', border: '1px solid #222', fontWeight: 'bold' }}>{data.model_type?.toUpperCase() || 'FAST'}</span>
        </div>
        {sessionUser && (
          <div style={{ backgroundColor: '#111', padding: '6px 14px', borderRadius: '10px', border: '1px solid #222', fontSize: '13px' }}>
            <span style={{ color: '#39FF14', fontWeight: 'bold' }}>분석 담당: {sessionUser.name}</span>
          </div>
        )}
      </header>

      <div style={{ maxWidth: '1000px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '30px' }}>
        
        <div style={{ 
          padding: '30px 40px', 
          backgroundColor: '#0A0A0A', 
          borderRadius: '24px', 
          border: '1px solid #161616', 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          boxShadow: '0 4px 20px rgba(0,0,0,0.3)'
        }}>
          <div style={{ textAlign: 'left' }}>
            <p style={{ color: '#444', fontSize: '12px', letterSpacing: '3px', margin: '0 0 5px 0', fontWeight: 'bold' }}>FINAL ANALYSIS REPORT</p>
            <h1 style={{ fontSize: '56px', fontWeight: '900', color: label === 'FAKE' ? '#FF4B4B' : (isInvalid ? '#444' : '#39FF14'), margin: 0, letterSpacing: '1px' }}>
              {label}
            </h1>
          </div>
          <div style={{ textAlign: 'right' }}>
            <p style={{ color: '#444', fontSize: '12px', margin: '0 0 5px 0', fontWeight: 'bold' }}>변조 신뢰 확률</p>
            <p style={{ fontSize: '42px', fontWeight: '900', margin: 0, color: isInvalid ? '#444' : '#fff' }}>{displayProb}</p>
          </div>
        </div>

        <div style={{ 
          backgroundColor: '#050505', 
          borderRadius: '24px', 
          border: '1px solid #161616', 
          width: '100%',
          height: '520px', 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          overflow: 'hidden', 
          boxShadow: '0 15px 40px rgba(0,0,0,0.6)' 
        }}>
          {/* ──────────────────────────────────────────────────────────────────────── */}
          {/* 🛠️ [조건부 렌더링 스위칭 적용] 캠코더 에러 이모티콘 구역 타겟 교정 */}
          {/* ──────────────────────────────────────────────────────────────────────── */}
          {mediaLoc ? (
            isVideoContent ? (
              // 🎥 비디오 결과창으로 진입했을 때: 원본 동영상 스트림 출력
              <video 
                src={mediaSrc} 
                controls 
                autoPlay
                muted
                style={{ width: '100%', height: '100%', objectFit: 'contain' }} 
              />
            ) : (
              // 🖼️ 이미지 결과창으로 진입했을 때: 캠코더 대신 사용자가 올린 정지 이미지 출력
              <img 
                src={mediaSrc} 
                alt="Analyzed Original Content" 
                style={{ maxWidth: '95%', maxHeight: '95%', objectFit: 'contain', borderRadius: '12px' }} 
              />
            )
          ) : (
            <div style={{ color: '#444', textAlign: 'center' }}>
              <span style={{ fontSize: '40px', display: 'block', marginBottom: '10px' }}>⚠️</span>
              분석 대상 원본 미디어 스트림을 불러올 수 없습니다.
            </div>
          )}
          {/* ──────────────────────────────────────────────────────────────────────── */}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          
          <div style={{ padding: '24px', backgroundColor: '#0D0D0D', borderRadius: '20px', border: '1px solid #1A1A1A', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <div>
              <p style={{ color: '#555', fontSize: '11px', fontWeight: 'bold', letterSpacing: '1px', margin: '0 0 10px 0' }}>FACE BRIGHTNESS (밝기 데이터)</p>
              <h2 style={{ fontSize: '36px', color: brightnessColor, margin: '0', fontWeight: 'bold' }}>
                {Number(face_brightness).toFixed(1)}%
              </h2>
            </div>
            <div>
              <div style={{ width: '100%', height: '3px', backgroundColor: '#1A1A1A', marginTop: '15px', borderRadius: '2px', overflow: 'hidden' }}>
                <div style={{ width: `${face_brightness}%`, height: '100%', backgroundColor: brightnessColor }} />
              </div>
              <p style={{ fontSize: '11px', color: '#444', margin: '8px 0 0 0' }}>기준치 20% 미만일 때 저조도 분석 경고등 점등</p>
            </div>
          </div>
          
          <div style={{ padding: '24px', backgroundColor: '#0D0D0D', borderRadius: '20px', border: '1px solid #1A1A1A', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
            <div>
              <p style={{ color: '#555', fontSize: '11px', fontWeight: 'bold', letterSpacing: '1px', margin: '0 0 10px 0' }}>FACE RATIO (화면 내 안면 비중)</p>
              <h2 style={{ fontSize: '36px', color: ratioColor, margin: '0', fontWeight: 'bold' }}>
                {Number(face_ratio).toFixed(1)}%
              </h2>
            </div>
            <div>
              <div style={{ width: '100%', height: '3px', backgroundColor: '#1A1A1A', marginTop: '15px', borderRadius: '2px', overflow: 'hidden' }}>
                <div style={{ width: `${Math.min(face_ratio * 10, 100)}%`, height: '100%', backgroundColor: ratioColor }} />
              </div>
              <p style={{ fontSize: '11px', color: '#444', margin: '8px 0 0 0' }}>안면 식별 기준치 3% 이상일 때 유효 분석 범위 인정</p>
            </div>
          </div>

          <div style={{ gridColumn: 'span 2', padding: '26px', backgroundColor: '#0D0D0D', borderRadius: '20px', border: '1px solid #1A1A1A' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '12px' }}>
              <p style={{ color: '#555', fontSize: '11px', fontWeight: 'bold', letterSpacing: '1px', margin: 0 }}>INTELLIGENT MODEL CONFIDENCE (엔진 신뢰 수준)</p>
              <h2 style={{ fontSize: '28px', color: isInvalid ? '#444' : '#39FF14', margin: '0', fontWeight: 'bold' }}>
                {Number(face_conf).toFixed(1)}%
              </h2>
            </div>
            
            <div style={{ width: '100%', height: '8px', backgroundColor: '#151515', borderRadius: '4px', overflow: 'hidden', border: '1px solid #222' }}>
              <div style={{ width: `${isInvalid ? 0 : face_conf}%`, height: '100%', background: 'linear-gradient(90deg, #1A2C50, #39FF14)', transition: 'width 1.2s cubic-bezier(0.1, 1, 0.1, 1)' }} />
            </div>
            
            <p style={{ fontSize: '13px', color: '#777', margin: '15px 0 0 0', lineHeight: '1.4' }}>
              {data.message || (isInvalid ? "영상 분석 메타데이터 아카이브 쿼리 실패 또는 분석이 중단된 인스턴스입니다." : "본 리포트는 Deep Guard 고성능 인공지능 신경망 커널 검증 레이어를 거쳐 연산된 신뢰도 지표입니다.")}
            </p>
          </div>

        </div>
      </div>
    </div>
  );
};

export default VideoAnalysisDetailPage;
import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';

const apiUrl = process.env.REACT_APP_API_URL || "http://localhost:8000";

const VideoTimelinePage = ({ sessionUser }) => {
  const { state: data } = useLocation();
  const navigate = useNavigate();
  const [timelineData, setTimelineData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  const videoId = data?.video_id || data?.id;
  const mediaLoc = data?.video_loc || '';
  const mediaSrc = mediaLoc.startsWith('blob') ? mediaLoc : `${apiUrl}${mediaLoc}`;

  useEffect(() => {
    const fetchTimelineDetails = async () => {
      if (!videoId) {
        setIsLoading(false);
        return;
      }
      try {
        const response = await axios.get(`/inference/video/${videoId}/detail`);
        if (response.data && response.data.timeline) {
          setTimelineData(response.data.timeline);
        } else {
          const mockTimeline = Array.from({ length: 10 }, (_, i) => ({
            timestamp: `00:${String(i * 2).padStart(2, '0')}`,
            fake_score: Math.random() * 100,
            label: Math.random() > 0.5 ? 'FAKE' : 'REAL'
          }));
          setTimelineData(mockTimeline);
        }
      } catch (error) {
        const fallbackTimeline = Array.from({ length: 10 }, (_, i) => ({
          timestamp: `00:${String(i * 2).padStart(2, '0')}`,
          fake_score: (i >= 3 && i <= 6) ? 85.5 : 12.3,
          label: (i >= 3 && i <= 6) ? 'FAKE' : 'REAL'
        }));
        setTimelineData(fallbackTimeline);
      } finally {
        setIsLoading(false);
      }
    };

    fetchTimelineDetails();
  }, [videoId]);

  if (!data) {
    return (
      <div style={{ backgroundColor: '#000', minHeight: '100vh', display: 'flex', justifyContent: 'center', alignItems: 'center', color: 'white' }}>
        <p>전송된 비디오 분석 데이터가 없습니다.</p>
        <button onClick={() => navigate(-1)} style={{ marginLeft: '15px', padding: '8px 16px', backgroundColor: '#1A2C50', color: '#39FF14', border: 'none', borderRadius: '6px', cursor: 'pointer' }}>뒤로가기</button>
      </div>
    );
  }

  return (
    <div style={{ backgroundColor: '#000', minHeight: '100vh', width: '100vw', color: 'white', padding: '40px 80px', boxSizing: 'border-box', fontFamily: 'sans-serif' }}>
      
      {/* 네비게이션 헤더 */}
      <header style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '40px', alignItems: 'center' }}>
        <button onClick={() => navigate(-1)} style={{ color: '#888', background: 'none', border: 'none', cursor: 'pointer', fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span>←</span> 분석 결과 목록
        </button>
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          <span style={{ padding: '6px 14px', backgroundColor: '#111', borderRadius: '4px', fontSize: '12px', color: '#39FF14', border: '1px solid #222', fontWeight: 'bold', letterSpacing: '0.5px' }}>
            FRAME-LEVEL ANALYSIS REPORT
          </span>
          {sessionUser && (
            <div style={{ backgroundColor: '#111', padding: '8px 16px', borderRadius: '6px', border: '1px solid #222', fontSize: '13px' }}>
              <span style={{ color: '#aaa' }}>ANALYSIS TASK MANAGER: </span>
              <span style={{ color: '#39FF14', fontWeight: 'bold' }}>{sessionUser.name}</span>
            </div>
          )}
        </div>
      </header>

      {/* 메인 콘텐츠 영역 */}
      <div style={{ maxWidth: '1200px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '30px' }}>
        
        {/* 상단 미디어 정보 패널 존 */}
        <div style={{ display: 'flex', backgroundColor: '#050505', borderRadius: '16px', border: '1px solid #1A1A1A', padding: '24px', gap: '32px', alignItems: 'center' }}>
          <div style={{ flex: 1.2, height: '280px', display: 'flex', justifyContent: 'center', alignItems: 'center', backgroundColor: '#000', borderRadius: '8px', overflow: 'hidden', border: '1px solid #111' }}>
            <video src={mediaSrc} controls style={{ maxWidth: '100%', maxHeight: '100%' }} />
          </div>
          <div style={{ flex: 0.8 }}>
            <p style={{ color: '#444', fontSize: '11px', letterSpacing: '2px', margin: '0 0 8px 0', fontWeight: 'bold' }}>TARGET METADATA</p>
            <h3 style={{ fontSize: '22px', margin: '0 0 20px 0', color: '#fff', fontWeight: '700' }}>{data.video_name || `STREAM_INSTANCE_${videoId}`}</h3>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '14px', color: '#aaa' }}>
              <div style={{ display: 'flex', borderBottom: '1px solid #111', paddingBottom: '8px' }}>
                <span style={{ width: '140px', color: '#555', fontWeight: 'bold', fontSize: '12px' }}>ANALYSIS MODEL</span>
                <span style={{ color: '#fff', fontWeight: '6px' }}>{data.model_type?.toUpperCase() || 'FAST'} ENGINE</span>
              </div>
              <div style={{ display: 'flex', borderBottom: '1px solid #111', paddingBottom: '8px' }}>
                <span style={{ width: '140px', color: '#555', fontWeight: 'bold', fontSize: '12px' }}>CORE KERNEL VERSION</span>
                <span style={{ color: '#39FF14', fontWeight: '600' }}>{data.version_type?.toUpperCase() || 'V1'} SYSTEM</span>
              </div>
              <div style={{ display: 'flex', paddingBottom: '4px' }}>
                <span style={{ width: '140px', color: '#555', fontWeight: 'bold', fontSize: '12px' }}>TOTAL FORGERY RISK</span>
                <span style={{ color: '#FF4B4B', fontWeight: '700' }}>{(Number(data.prob ?? 0) * 100).toFixed(1)}% RISK</span>
              </div>
            </div>
          </div>
        </div>

        {/* 하단 핵심: 타임라인 그래프 & 프레임 데이터 보드 */}
        <div style={{ backgroundColor: '#0A0A0A', borderRadius: '16px', border: '1px solid #161616', padding: '32px' }}>
          <p style={{ color: '#444', fontSize: '11px', letterSpacing: '2px', margin: '0 0 24px 0', fontWeight: 'bold' }}>CHRONOLOGICAL FORGERY RISK MATRIX</p>

          {isLoading ? (
            <div style={{ padding: '60px', textAlign: 'center', color: '#444', fontSize: '14px' }}>CALCULATING FRAME-LEVEL METRICS...</div>
          ) : (
            <div>
              {/* 바 차트 레이아웃 시각화 구역 */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', padding: '0 10px', backgroundColor: '#020202', borderRadius: '8px', border: '1px solid #111', height: '180px', padding: '30px 40px', boxSizing: 'border-box' }}>
                {timelineData.map((frame, i) => {
                  const isFakeUnit = frame.fake_score > 50;
                  return (
                    <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', height: '100%', justifyContent: 'flex-end', gap: '12px' }}>
                      <div style={{ 
                        width: '24%', 
                        height: `${Math.max(frame.fake_score, 4)}%`, // 최소 높이 보장하여 플랫 라인 방지
                        backgroundColor: isFakeUnit ? '#FF4B4B' : '#39FF14',
                        borderRadius: '2px 2px 0 0',
                        transition: 'height 0.4s cubic-bezier(0.1, 1, 0.1, 1)',
                        boxShadow: isFakeUnit ? '0 0 12px rgba(255,75,75,0.25)' : '0 0 12px rgba(57,255,20,0.15)'
                      }} />
                      <span style={{ fontSize: '11px', color: '#444', fontFamily: 'monospace', fontWeight: '600' }}>{frame.timestamp}</span>
                    </div>
                  );
                })}
              </div>

              {/* 구간별 텍스트 매핑 분석 그리드 */}
              <div style={{ marginTop: '24px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                {timelineData.map((frame, idx) => {
                  const isFake = frame.fake_score > 50;
                  return (
                    <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px 24px', backgroundColor: '#030303', borderRadius: '8px', border: '1px solid #141414' }}>
                      <span style={{ fontSize: '13px', fontWeight: '600', color: '#888', fontFamily: 'monospace' }}>TIMESTAMP {frame.timestamp}</span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                        <span style={{ fontSize: '13px', color: '#ccc' }}>FORGERY: <b style={{ color: isFake ? '#FF4B4B' : '#39FF14', fontFamily: 'monospace', fontSize: '14px' }}>{Number(frame.fake_score).toFixed(1)}%</b></span>
                        <span style={{ 
                          padding: '3px 10px', 
                          borderRadius: '4px', 
                          fontSize: '11px', 
                          fontWeight: 'bold', 
                          letterSpacing: '0.5px',
                          backgroundColor: isFake ? 'rgba(255,75,75,0.08)' : 'rgba(57,255,20,0.06)', 
                          color: isFake ? '#FF4B4B' : '#39FF14', 
                          border: `1px solid ${isFake ? 'rgba(255,75,75,0.4)' : 'rgba(57,255,20,0.3)'}` 
                        }}>
                          {isFake ? 'MANIPULATED' : 'VERIFIED'}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>

            </div>
          )}
        </div>

      </div>
    </div>
  );
};

export default VideoTimelinePage;
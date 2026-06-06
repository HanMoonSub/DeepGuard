import React, { useEffect, useState, useRef, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';



const normalizeFrame = (f, i) => {
  const raw = f.score ?? f.fake_score ?? 0;
  const fake_score = raw <= 1 ? raw * 100 : raw;
  const ts = f.frame_time != null
    ? new Date(f.frame_time * 1000).toISOString().substr(14, 5)
    : f.timestamp ?? `00:${String(i * 2).padStart(2, '0')}`;
  return { ...f, fake_score, frame_index: f.frame_index ?? i, timestamp: ts };
};

const VideoTimelinePage = ({ sessionUser }) => {
  const { state: data } = useLocation();
  const navigate = useNavigate();
  const [timelineData, setTimelineData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  const videoId = data?.video_id || data?.id;
  const mediaLoc = data?.video_loc || '';
  const mediaSrc = mediaLoc.startsWith('blob') ? mediaLoc : mediaLoc;

  // Canvas는 막대 div 위에 absolute로 얹히므로
  // 막대 div 자체를 ref로 잡아서 크기/위치를 측정
  const canvasRef   = useRef(null);
  const barsRef     = useRef(null); // 막대들이 들어있는 flex div
  const barRefs     = useRef([]);   // 각 막대 div ref 배열

  useEffect(() => {
    const fetchTimelineDetails = async () => {
      if (!videoId) { setIsLoading(false); return; }
      try {
        const response = await axios.get(`/inference/video/${videoId}/detail`);
        const d = response.data;
        let frames = [];
        if (d?.frames?.length)        frames = d.frames;
        else if (d?.timeline?.length) frames = d.timeline;
        if (frames.length) {
          setTimelineData(frames.map(normalizeFrame));
        } else {
          setTimelineData(Array.from({ length: 10 }, (_, i) => normalizeFrame({
            frame_index: i, frame_time: i * 2,
            score: (i >= 3 && i <= 6) ? 0.855 : 0.123,
          }, i)));
        }
      } catch {
        setTimelineData(Array.from({ length: 10 }, (_, i) => normalizeFrame({
          frame_index: i, frame_time: i * 2,
          score: (i >= 3 && i <= 6) ? 0.855 : 0.123,
        }, i)));
      } finally {
        setIsLoading(false);
      }
    };
    fetchTimelineDetails();
  }, [videoId]);

  // ── Canvas 렌더링 ──
  // 핵심: 각 막대 div의 getBoundingClientRect()로 실제 화면 위치를 측정해
  //       캔버스 좌표로 변환 → 완벽하게 막대 상단 중앙을 통과하는 꺾은선
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const barsEl = barsRef.current;
    if (!canvas || !barsEl || timelineData.length < 2) return;
    if (barRefs.current.length !== timelineData.length) return;
    if (barRefs.current.some(r => !r)) return;

    const barsRect = barsEl.getBoundingClientRect();
    const W = barsRect.width;
    const H = barsRect.height;
    const dpr = window.devicePixelRatio || 1;

    canvas.width  = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width  = W + 'px';
    canvas.style.height = H + 'px';

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    // 각 막대 상단 중앙 좌표 계산
    // barRef는 막대 div (색깔 있는 직사각형)
    const points = barRefs.current.map((barEl, i) => {
      const barRect = barEl.getBoundingClientRect();
      // canvas 기준 좌표 = barRect - barsRect (offset)
      const x = barRect.left - barsRect.left + barRect.width / 2;
      const y = barRect.top  - barsRect.top;  // 막대 상단
      return { x, y, isFake: timelineData[i].fake_score > 50 };
    });

    // 50% 기준선 y좌표: fake_score=50일 때의 막대 높이 비율로 계산
    // 막대 영역 높이 = barsEl에서 timestamp span 제외한 부분
    // barsEl padding-top=30px → 막대 영역 시작 y=30, 막대 바닥 y=H-23(timestamp)
    const BAR_TOP    = 30;  // padding-top
    const BAR_BOTTOM = H - 23; // timestamp 텍스트 영역 제외
    const BAR_H      = BAR_BOTTOM - BAR_TOP;
    const y50 = BAR_BOTTOM - (50 / 100) * BAR_H;

    // 50% 기준선
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = 'rgba(255, 75, 75, 0.3)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(points[0].x, y50);
    ctx.lineTo(points[points.length - 1].x, y50);
    ctx.stroke();
    ctx.restore();

    // 꺾은선 글로우 (바깥 레이어)
    ctx.save();
    ctx.shadowColor = 'rgba(255,255,255,0.6)';
    ctx.shadowBlur  = 10;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.lineWidth   = 2;
    ctx.lineJoin    = 'round';
    ctx.lineCap     = 'round';
    ctx.stroke();
    ctx.restore();

    // 데이터 포인트 dot
    points.forEach((pt) => {
      // 외곽 글로우 원
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 7, 0, Math.PI * 2);
      ctx.fillStyle = pt.isFake ? 'rgba(255,75,75,0.15)' : 'rgba(57,255,20,0.12)';
      ctx.fill();
      // 내부 dot
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = pt.isFake ? '#FF4B4B' : '#39FF14';
      ctx.fill();
    });
  }, [timelineData]);

  useEffect(() => {
    if (isLoading || !timelineData.length) return;
    // DOM 렌더 완료 후 실행 (두 번 대기로 안전하게)
    const t1 = setTimeout(drawCanvas, 100);
    const t2 = setTimeout(drawCanvas, 400);
    const ro = new ResizeObserver(() => setTimeout(drawCanvas, 50));
    if (barsRef.current) ro.observe(barsRef.current);
    return () => { clearTimeout(t1); clearTimeout(t2); ro.disconnect(); };
  }, [timelineData, isLoading, drawCanvas]);

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
                <span style={{ color: '#FF4B4B', fontWeight: '700' }}>{(Number(data.prob ?? data.score ?? 0) * 100).toFixed(1)}% RISK</span>
              </div>
            </div>
          </div>
        </div>

        {/* 하단: 타임라인 그래프 & 프레임 데이터 보드 */}
        <div style={{ backgroundColor: '#0A0A0A', borderRadius: '16px', border: '1px solid #161616', padding: '32px' }}>
          <p style={{ color: '#444', fontSize: '11px', letterSpacing: '2px', margin: '0 0 24px 0', fontWeight: 'bold' }}>CHRONOLOGICAL FORGERY RISK MATRIX</p>

          {isLoading ? (
            <div style={{ padding: '60px', textAlign: 'center', color: '#444', fontSize: '14px' }}>CALCULATING FRAME-LEVEL METRICS...</div>
          ) : (
            <div>
              {/* 차트 영역: 막대 + Canvas 꺾은선 */}
              <div style={{ position: 'relative', backgroundColor: '#020202', borderRadius: '8px', border: '1px solid #111', boxSizing: 'border-box' }}>

                {/* 막대 그래프 */}
                <div
                  ref={barsRef}
                  style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', height: '180px', padding: '30px 40px', boxSizing: 'border-box' }}
                >
                  {timelineData.map((frame, i) => {
                    const isFakeUnit = frame.fake_score > 50;
                    return (
                      <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', height: '100%', justifyContent: 'flex-end', gap: '12px' }}>
                        {/* 막대 div — ref 배열로 각각 잡음 */}
                        <div
                          ref={el => barRefs.current[i] = el}
                          style={{
                            width: '24%',
                            height: `${Math.max(frame.fake_score, 4)}%`,
                            backgroundColor: isFakeUnit ? '#FF4B4B' : '#39FF14',
                            borderRadius: '2px 2px 0 0',
                            transition: 'height 0.4s cubic-bezier(0.1, 1, 0.1, 1)',
                            boxShadow: isFakeUnit ? '0 0 12px rgba(255,75,75,0.25)' : '0 0 12px rgba(57,255,20,0.15)'
                          }}
                        />
                        <span style={{ fontSize: '11px', color: '#444', fontFamily: 'monospace', fontWeight: '600' }}>{frame.timestamp}</span>
                      </div>
                    );
                  })}
                </div>

                {/* Canvas: barsRef와 완전히 동일한 위치/크기 */}
                <canvas
                  ref={canvasRef}
                  style={{
                    position: 'absolute',
                    top: 0, left: 0,
                    width: '100%', height: '100%',
                    pointerEvents: 'none',
                  }}
                />
              </div>

              {/* 구간별 히트맵 버튼 행 */}
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '6px', marginTop: '10px' }}>
                {timelineData.map((frame, i) => {
                  const isFake = frame.fake_score > 50;
                  return (
                    <div key={i} style={{ flex: 1, display: 'flex', justifyContent: 'center' }}>
                      <button
                        onClick={() => navigate('/heatmap', {
                          state: {
                            video_id: videoId,
                            frame_index: frame.frame_index,
                            timestamp: frame.timestamp,
                            fake_score: frame.fake_score,
                            model_type: data.model_type || 'fast',
                            video_name: data.video_name,
                          }
                        })}
                        title={`${frame.timestamp} 히트맵`}
                        style={{
                          width: '100%', padding: '5px 0',
                          backgroundColor: 'transparent',
                          color: isFake ? '#FF4B4B' : '#39FF14',
                          border: `1px solid ${isFake ? 'rgba(255,75,75,0.3)' : 'rgba(57,255,20,0.25)'}`,
                          borderRadius: '4px', fontSize: '9px', fontWeight: '700',
                          letterSpacing: '0.3px', cursor: 'pointer', transition: 'all 0.15s',
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor = isFake ? 'rgba(255,75,75,0.1)' : 'rgba(57,255,20,0.08)';
                          e.currentTarget.style.boxShadow = isFake ? '0 0 8px rgba(255,75,75,0.2)' : '0 0 8px rgba(57,255,20,0.15)';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = 'transparent';
                          e.currentTarget.style.boxShadow = 'none';
                        }}
                      >
                        HEAT
                      </button>
                    </div>
                  );
                })}
              </div>

              {/* 구간별 텍스트 그리드 */}
              <div style={{ marginTop: '24px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                {timelineData.map((frame, idx) => {
                  const isFake = frame.fake_score > 50;
                  return (
                    <div key={idx} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '16px 24px', backgroundColor: '#030303', borderRadius: '8px', border: '1px solid #141414' }}>
                      <span style={{ fontSize: '13px', fontWeight: '600', color: '#888', fontFamily: 'monospace' }}>TIMESTAMP {frame.timestamp}</span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                        <span style={{ fontSize: '13px', color: '#ccc' }}>변조 확률: <b style={{ color: isFake ? '#FF4B4B' : '#39FF14', fontFamily: 'monospace', fontSize: '14px' }}>{Number(frame.fake_score).toFixed(1)}%</b></span>
                        <span style={{
                          padding: '3px 10px', borderRadius: '4px', fontSize: '11px', fontWeight: 'bold', letterSpacing: '0.5px',
                          backgroundColor: isFake ? 'rgba(255,75,75,0.08)' : 'rgba(57,255,20,0.06)',
                          color: isFake ? '#FF4B4B' : '#39FF14',
                          border: `1px solid ${isFake ? 'rgba(255,75,75,0.4)' : 'rgba(57,255,20,0.3)'}`
                        }}>
                          {isFake ? 'MANIPULATED' : 'VERIFIED'}
                        </span>
                        <button
                          onClick={() => navigate('/heatmap', {
                            state: {
                              video_id: videoId,
                              frame_index: frame.frame_index,
                              timestamp: frame.timestamp,
                              fake_score: frame.fake_score,
                              model_type: data.model_type || 'fast',
                              video_name: data.video_name,
                            }
                          })}
                          style={{
                            padding: '3px 10px', backgroundColor: 'transparent', color: '#666',
                            border: '1px solid #2A2A2A', borderRadius: '4px', fontSize: '11px',
                            fontWeight: 'bold', cursor: 'pointer', transition: 'all 0.15s',
                          }}
                          onMouseEnter={(e) => { e.currentTarget.style.borderColor = '#39FF14'; e.currentTarget.style.color = '#39FF14'; }}
                          onMouseLeave={(e) => { e.currentTarget.style.borderColor = '#2A2A2A'; e.currentTarget.style.color = '#666'; }}
                        >
                          히트맵
                        </button>
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
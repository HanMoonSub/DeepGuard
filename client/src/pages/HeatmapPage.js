import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';

const apiUrl = process.env.REACT_APP_API_URL || "http://localhost:8000";
const POLL_INTERVAL = 2000;

const REQUEST_BODY = {
  branch_level: 'low',
  explainer_type: 'layercam',
  display_type: 'heatmap_bbox',
  overlay_ratio: 0.7,
  threshold: 0.9,
  aug_smooth: false,
  eigen_smooth: true,
};

const extractTaskId = (data) => {
  if (typeof data === 'string') return data;
  return data?.task_id || data?.id || null;
};

const toAbsoluteUrl = (path) => {
  if (!path) return null;
  if (path.startsWith('http') || path.startsWith('blob')) return path;
  return `${apiUrl}${path}`;
};

const HeatmapPage = ({ sessionUser }) => {
  const { state } = useLocation();
  const navigate = useNavigate();

  const { video_id, frame_index, timestamp, fake_score, model_type, video_name } = state || {};

  const [status, setStatus]           = useState('idle');
  const [heatmapSrc, setHeatmapSrc]   = useState(null);
  const [taskId, setTaskId]           = useState(null);
  const [errorMsg, setErrorMsg]       = useState('');
  const [errorDetail, setErrorDetail] = useState('');
  const [elapsed, setElapsed]         = useState(0);

  const pollingRef = useRef(null);
  const timerRef   = useRef(null);
  const isMounted  = useRef(true);

  useEffect(() => {
    isMounted.current = true;
    return () => { isMounted.current = false; stopPolling(); clearInterval(timerRef.current); };
  }, []);

  useEffect(() => {
    if (video_id != null) requestHeatmap();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const stopPolling = () => {
    if (pollingRef.current) { clearInterval(pollingRef.current); pollingRef.current = null; }
  };

  const requestHeatmap = async () => {
    if (video_id == null) { setErrorMsg('video_id가 없습니다.'); setStatus('error'); return; }
    stopPolling();
    clearInterval(timerRef.current);
    setElapsed(0);
    setStatus('submitting');
    setHeatmapSrc(null);
    setErrorMsg('');
    setErrorDetail('');
    setTaskId(null);

    const body = { model_type: model_type || 'fast', ...REQUEST_BODY };

    try {
      const res = await axios.post(`/explain/video/${video_id}/frame/${frame_index ?? 0}`, body);
      const tid = extractTaskId(res.data);
      if (!tid) throw new Error(`task_id 추출 실패. 응답: ${JSON.stringify(res.data)}`);
      setTaskId(tid);
      setStatus('polling');
      timerRef.current = setInterval(() => setElapsed(p => p + 1), 1000);
      startPolling(tid);
    } catch (e) {
      if (!isMounted.current) return;
      const detail = e.response?.data?.detail;
      let msg;
      if (Array.isArray(detail)) {
        msg = detail.map(d => `[${d.loc?.join('.')}] ${d.msg} (입력값: ${JSON.stringify(d.input)})`).join(' / ');
      } else if (typeof detail === 'string') {
        msg = detail;
      } else if (e.response?.data) {
        msg = JSON.stringify(e.response.data);
      } else {
        msg = e.message || '요청 실패';
      }
      setErrorMsg(`히트맵 생성 요청 실패 (${e.response?.status ?? ''})`);
      setErrorDetail(msg);
      setStatus('error');
    }
  };

  const startPolling = (tid) => {
    let networkErrorCount = 0;
    const MAX_ERRORS = 5;
    pollingRef.current = setInterval(async () => {
      if (!isMounted.current) return;
      try {
        const res = await axios.get(`/explain/frame/result/${tid}`);
        networkErrorCount = 0;
        const d = res.data;
        if (d === null || d === undefined) return;

        let loc = null;
        if (typeof d === 'string' && d.trim().length > 0) {
          const isUUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(d.trim());
          if (!isUUID) loc = d.trim();
        } else if (typeof d === 'object') {
          loc = d.cam_loc ?? d.result_loc ?? d.image_loc ?? d.heatmap_loc ?? d.file_loc
              ?? d.result_path ?? d.path ?? d.url ?? d.output_path ?? null;
          if (!loc) {
            const st = (d.status || '').toUpperCase();
            if (st === 'FAILED' || st === 'ERROR') {
              stopPolling(); clearInterval(timerRef.current);
              setErrorMsg('히트맵 생성 실패');
              setErrorDetail(d.result_msg || d.message || d.detail || '서버 오류');
              setStatus('error'); return;
            }
            return;
          }
        }
        if (loc) {
          stopPolling(); clearInterval(timerRef.current);
          setHeatmapSrc(toAbsoluteUrl(loc));
          setStatus('done');
        }
      } catch (e) {
        networkErrorCount++;
        if (networkErrorCount >= MAX_ERRORS) {
          stopPolling(); clearInterval(timerRef.current);
          setErrorMsg('서버 연결 실패');
          setErrorDetail(`${MAX_ERRORS}회 연속 연결 오류. (${e.message})`);
          setStatus('error');
        }
      }
    }, POLL_INTERVAL);
  };

  const isFake = (fake_score ?? 0) > 50;
  const isProcessing = status === 'submitting' || status === 'polling';

  if (!state) {
    return (
      <div style={{ backgroundColor: '#000', minHeight: '100vh', display: 'flex', justifyContent: 'center', alignItems: 'center', color: 'white' }}>
        <p>전송된 프레임 데이터가 없습니다.</p>
        <button onClick={() => navigate(-1)} style={{ marginLeft: '15px', padding: '8px 16px', backgroundColor: '#1A2C50', color: '#39FF14', border: 'none', borderRadius: '6px', cursor: 'pointer' }}>뒤로가기</button>
      </div>
    );
  }

  return (
    <div style={{ backgroundColor: '#000', minHeight: '100vh', width: '100vw', color: 'white', fontFamily: 'sans-serif', overflowX: 'hidden', position: 'relative' }}>

      {/* 배경 그리드 */}
      <div style={{ position: 'fixed', inset: 0, backgroundImage: 'linear-gradient(rgba(57,255,20,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(57,255,20,0.03) 1px, transparent 1px)', backgroundSize: '60px 60px', pointerEvents: 'none', zIndex: 0 }} />
      <div style={{ position: 'fixed', top: '-20%', right: '-10%', width: '600px', height: '600px', borderRadius: '50%', background: 'radial-gradient(circle, rgba(57,255,20,0.04) 0%, transparent 70%)', pointerEvents: 'none', zIndex: 0 }} />
      <div style={{ position: 'fixed', bottom: '-20%', left: '-10%', width: '500px', height: '500px', borderRadius: '50%', background: `radial-gradient(circle, ${isFake ? 'rgba(255,75,75,0.04)' : 'rgba(57,255,20,0.04)'} 0%, transparent 70%)`, pointerEvents: 'none', zIndex: 0 }} />

      <div style={{ position: 'relative', zIndex: 1, padding: '32px 60px', maxWidth: '1400px', margin: '0 auto' }}>

        {/* 헤더 */}
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '36px' }}>
          <button onClick={() => navigate(-1)}
            style={{ color: '#555', background: 'none', border: '1px solid #1A1A1A', cursor: 'pointer', fontSize: '13px', display: 'flex', alignItems: 'center', gap: '8px', padding: '8px 16px', borderRadius: '8px', transition: 'all 0.2s' }}
            onMouseEnter={e => { e.currentTarget.style.borderColor = '#39FF14'; e.currentTarget.style.color = '#39FF14'; }}
            onMouseLeave={e => { e.currentTarget.style.borderColor = '#1A1A1A'; e.currentTarget.style.color = '#555'; }}>
            ← 타임라인으로
          </button>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ padding: '6px 16px', borderRadius: '6px', fontSize: '12px', fontWeight: '900', letterSpacing: '2px', backgroundColor: isFake ? 'rgba(255,75,75,0.1)' : 'rgba(57,255,20,0.08)', color: isFake ? '#FF4B4B' : '#39FF14', border: `1px solid ${isFake ? 'rgba(255,75,75,0.4)' : 'rgba(57,255,20,0.3)'}` }}>
              {isFake ? 'FAKE' : 'REAL'}
            </div>
            <div style={{ padding: '6px 16px', backgroundColor: '#0A0A0A', borderRadius: '6px', fontSize: '12px', color: '#39FF14', border: '1px solid #1A1A1A', fontWeight: 'bold', letterSpacing: '1px' }}>
              FRAME FORGERY TRACE
            </div>
            {sessionUser && (
              <div style={{ backgroundColor: '#0A0A0A', padding: '6px 14px', borderRadius: '6px', border: '1px solid #1A1A1A', fontSize: '12px' }}>
                <span style={{ color: '#555' }}>담당: </span>
                <span style={{ color: '#39FF14', fontWeight: 'bold' }}>{sessionUser.name}</span>
              </div>
            )}
          </div>
        </header>

        {/* 스탯 바 */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '12px', marginBottom: '24px' }}>
          {[
            { label: 'VIDEO', value: video_name || `ID_${video_id}`, color: '#fff' },
            { label: 'TIMESTAMP', value: timestamp || `#${frame_index}`, color: '#39FF14', big: true },
            { label: 'FRAME INDEX', value: `#${frame_index ?? 0}`, color: '#fff' },
            { label: 'FORGERY RISK', value: `${Number(fake_score ?? 0).toFixed(1)}%`, color: isFake ? '#FF4B4B' : '#39FF14', big: true },
            { label: 'VERDICT', value: isFake ? 'FAKE' : 'REAL', color: isFake ? '#FF4B4B' : '#39FF14', big: true },
          ].map((item, i) => (
            <div key={i} style={{ backgroundColor: '#050505', border: '1px solid #1A1A1A', borderRadius: '12px', padding: '16px 20px', position: 'relative', overflow: 'hidden' }}>
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '2px', background: `linear-gradient(90deg, transparent, ${item.color}40, transparent)` }} />
              <p style={{ color: '#444', fontSize: '10px', fontWeight: 'bold', letterSpacing: '2px', margin: '0 0 6px 0' }}>{item.label}</p>
              <p style={{ color: item.color, fontWeight: '900', fontFamily: 'monospace', fontSize: item.big ? '20px' : '14px', margin: 0, letterSpacing: '1px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{item.value}</p>
            </div>
          ))}
        </div>

        {/* 히트맵 결과 패널 (전체 너비) */}
        <div style={{ backgroundColor: '#050505', borderRadius: '16px', border: `1px solid ${isProcessing ? 'rgba(57,255,20,0.2)' : status === 'done' ? 'rgba(57,255,20,0.15)' : '#1A1A1A'}`, display: 'flex', flexDirection: 'column', overflow: 'hidden', position: 'relative', transition: 'border-color 0.5s' }}>
          {isProcessing && (
            <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '2px', background: 'linear-gradient(90deg, transparent, #39FF14, transparent)', animation: 'slideRight 2s linear infinite', zIndex: 2 }} />
          )}

          {/* 패널 헤더 */}
          <div style={{ padding: '14px 22px', borderBottom: '1px solid #111', display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: '#030303' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: isProcessing ? '#39FF14' : status === 'done' ? '#39FF14' : '#333', boxShadow: isProcessing ? '0 0 8px #39FF14' : 'none', animation: isProcessing ? 'pulse 1.5s ease-in-out infinite' : 'none' }} />
              <span style={{ color: '#555', fontSize: '11px', fontWeight: 'bold', letterSpacing: '2px' }}>HEATMAP + BBOX OVERLAY</span>
            </div>
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
              {isProcessing && <span style={{ fontSize: '11px', color: '#39FF14', fontFamily: 'monospace' }}>{elapsed}s</span>}
              {status === 'done' && heatmapSrc && (
                <a href={heatmapSrc} download={`heatmap_f${frame_index}.png`}
                  style={{ fontSize: '11px', color: '#39FF14', textDecoration: 'none', border: '1px solid rgba(57,255,20,0.3)', padding: '4px 12px', borderRadius: '4px', fontWeight: 'bold' }}>
                  ↓ SAVE
                </a>
              )}
            </div>
          </div>

          {/* 패널 본문 */}
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '48px', minHeight: '480px', position: 'relative' }}>

            {isProcessing && (
              <div style={{ textAlign: 'center', padding: '40px 0' }}>
                <div style={{ width: '48px', height: '48px', border: '3px solid #1A1A1A', borderTop: '3px solid #39FF14', borderRadius: '50%', margin: '0 auto 24px', animation: 'spin 0.8s linear infinite' }} />
                <p style={{ fontSize: '13px', color: '#39FF14', letterSpacing: '2px', fontWeight: 'bold', margin: '0 0 8px 0' }}>GENERATING HEATMAP</p>
                <p style={{ fontSize: '11px', color: '#444', margin: '0 0 20px 0' }}>프레임 위조 흔적 시각화 처리 중...</p>
                <p style={{ fontSize: '12px', color: '#555', fontFamily: 'monospace', margin: 0 }}>{elapsed}s elapsed</p>
              </div>
            )}

            {status === 'done' && heatmapSrc && (
              <div style={{ width: '100%', position: 'relative', textAlign: 'center' }}>
                <div style={{ position: 'absolute', top: 0, right: 0, zIndex: 2, padding: '4px 10px', backgroundColor: 'rgba(57,255,20,0.1)', border: '1px solid rgba(57,255,20,0.3)', borderRadius: '6px', fontSize: '10px', color: '#39FF14', fontWeight: 'bold', letterSpacing: '1px' }}>
                  ✓ COMPLETE
                </div>
                <img src={heatmapSrc} alt="Heatmap Result"
                  style={{ maxWidth: '100%', maxHeight: '600px', borderRadius: '10px', objectFit: 'contain', display: 'block', margin: '0 auto', boxShadow: '0 0 40px rgba(57,255,20,0.08), 0 8px 32px rgba(0,0,0,0.6)' }}
                  onError={(e) => { e.target.style.display = 'none'; setErrorMsg('이미지 로드 실패'); setErrorDetail(`URL: ${heatmapSrc}`); setStatus('error'); }}
                />
              </div>
            )}

            {status === 'done' && !heatmapSrc && (
              <div style={{ textAlign: 'center' }}>
                <p style={{ color: '#555', marginBottom: '16px' }}>결과 이미지를 받지 못했습니다.</p>
                <button onClick={requestHeatmap} style={{ padding: '10px 24px', backgroundColor: 'transparent', color: '#39FF14', border: '1px solid rgba(57,255,20,0.4)', borderRadius: '8px', cursor: 'pointer', fontSize: '13px', fontWeight: 'bold' }}>재시도</button>
              </div>
            )}

            {status === 'error' && (
              <div style={{ textAlign: 'center', maxWidth: '400px' }}>
                <div style={{ width: '60px', height: '60px', borderRadius: '50%', border: '1px solid rgba(255,75,75,0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 20px', fontSize: '24px', backgroundColor: 'rgba(255,75,75,0.06)' }}>⚠</div>
                <p style={{ fontSize: '14px', fontWeight: 'bold', color: '#FF4B4B', marginBottom: '8px' }}>{errorMsg}</p>
                {errorDetail && (
                  <p style={{ fontSize: '10px', color: '#444', lineHeight: '1.8', marginBottom: '20px', fontFamily: 'monospace', wordBreak: 'break-all', backgroundColor: '#0A0A0A', padding: '12px', borderRadius: '6px', border: '1px solid #1A1A1A', textAlign: 'left' }}>
                    {errorDetail}
                  </p>
                )}
                <button onClick={requestHeatmap} style={{ padding: '10px 24px', backgroundColor: 'transparent', color: '#FF4B4B', border: '1px solid rgba(255,75,75,0.4)', borderRadius: '8px', cursor: 'pointer', fontSize: '13px', fontWeight: 'bold' }}>재시도</button>
              </div>
            )}
          </div>
        </div>

        {/* 하단 파라미터 바 */}
        <div style={{ marginTop: '16px', backgroundColor: '#0A0A0A', borderRadius: '12px', border: '1px solid #222', padding: '16px 24px', display: 'flex', gap: '0', alignItems: 'stretch', flexWrap: 'wrap', overflow: 'hidden', position: 'relative' }}>
          <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '1px', background: 'linear-gradient(90deg, transparent, rgba(57,255,20,0.4), transparent)' }} />
          <div style={{ display: 'flex', alignItems: 'center', paddingRight: '20px', marginRight: '20px', borderRight: '1px solid #1A1A1A' }}>
            <span style={{ color: '#39FF14', fontSize: '10px', fontWeight: 'bold', letterSpacing: '2px' }}>ANALYSIS PARAMS</span>
          </div>
          <div style={{ display: 'flex', gap: '0', flexWrap: 'wrap', flex: 1 }}>
            {Object.entries({ ...REQUEST_BODY, model_type: model_type || 'fast' }).map(([k, v], i, arr) => (
              <div key={k} style={{ display: 'flex', flexDirection: 'column', gap: '3px', paddingRight: '20px', marginRight: '20px', borderRight: i < arr.length - 1 ? '1px solid #1A1A1A' : 'none' }}>
                <span style={{ color: '#444', fontSize: '9px', fontFamily: 'monospace', letterSpacing: '1px' }}>{k}</span>
                <span style={{ color: String(v) === 'true' ? '#39FF14' : String(v) === 'false' ? '#FF4B4B' : '#fff', fontSize: '12px', fontFamily: 'monospace', fontWeight: 'bold' }}>{String(v)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes scanLine    { 0% { top:-2px; opacity:0; } 10% { opacity:1; } 90% { opacity:1; } 100% { top:100%; opacity:0; } }
        @keyframes loadingBar  { 0% { background-position:100% 0; } 100% { background-position:-100% 0; } }
        @keyframes slideRight  { 0% { transform:translateX(-100%); } 100% { transform:translateX(100%); } }
        @keyframes spin        { from { transform:rotate(0deg); } to { transform:rotate(360deg); } }
        @keyframes pulse       { 0%,100% { opacity:1; box-shadow:0 0 8px #39FF14; } 50% { opacity:0.4; box-shadow:0 0 3px #39FF14; } }
      `}</style>
    </div>
  );
};

export default HeatmapPage;
import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';

const apiUrl = process.env.REACT_APP_API_URL || "http://localhost:8000";
const POLL_INTERVAL = 2000;

const BRANCH_OPTIONS = {
  low:  { branch_level: 'low',  explainer_type: 'layercam',     display_type: 'heatmap_bbox', overlay_ratio: 0.7, threshold: 0.9 },
  high: { branch_level: 'high', explainer_type: 'eigengradcam', display_type: 'heatmap_bbox', overlay_ratio: 0.7, threshold: 0.9 },
};
const MODEL_OPTIONS = ['fast', 'pro'];

const extractTaskId = (data) => {
  if (typeof data === 'string') return data;
  return data?.task_id || data?.id || null;
};
const toAbsoluteUrl = (path) => {
  if (!path) return null;
  if (path.startsWith('http') || path.startsWith('blob')) return path;
  return `${apiUrl}${path}`;
};

const ImageHeatmapPage = ({ sessionUser }) => {
  const { state } = useLocation();
  const navigate  = useNavigate();
  const { image_id, image_loc, model_type, prob, label } = state || {};

  const [tempBranch, setTempBranch]         = useState('low');
  const [tempModel, setTempModel]           = useState(model_type || 'fast');
  const [selectedBranch, setSelectedBranch] = useState(null);
  const [selectedModel, setSelectedModel]   = useState(null);
  const [status, setStatus]                 = useState('idle');
  const [heatmapSrc, setHeatmapSrc]         = useState(null);
  const [taskId, setTaskId]                 = useState(null);
  const [errorMsg, setErrorMsg]             = useState('');
  const [errorDetail, setErrorDetail]       = useState('');
  const [elapsed, setElapsed]               = useState(0);

  const pollingRef = useRef(null);
  const timerRef   = useRef(null);
  const isMounted  = useRef(true);

  useEffect(() => {
    isMounted.current = true;
    return () => { isMounted.current = false; stopPolling(); clearInterval(timerRef.current); };
  }, []);

  const stopPolling = () => {
    if (pollingRef.current) { clearInterval(pollingRef.current); pollingRef.current = null; }
  };

  const handleBranchSelect = (branch, m) => {
    setSelectedBranch(branch);
    setSelectedModel(m);
    setTempBranch(branch);
    setTempModel(m);
    requestHeatmap(branch, m);
  };

  const requestHeatmap = async (branch, m) => {
    if (image_id == null) { setErrorMsg('image_id가 없습니다.'); setStatus('error'); return; }
    stopPolling();
    clearInterval(timerRef.current);
    setElapsed(0);
    setStatus('submitting');
    setHeatmapSrc(null);
    setErrorMsg('');
    setErrorDetail('');
    setTaskId(null);

    const body = { model_type: m, ...BRANCH_OPTIONS[branch] };

    try {
      const res = await axios.post(`/explain/image/${image_id}`, body);
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
      if (Array.isArray(detail)) msg = detail.map(d => `[${d.loc?.join('.')}] ${d.msg} (입력값: ${JSON.stringify(d.input)})`).join(' / ');
      else if (typeof detail === 'string') msg = detail;
      else if (e.response?.data) msg = JSON.stringify(e.response.data);
      else msg = e.message || '요청 실패';
      setErrorMsg(`히트맵 생성 요청 실패 (${e.response?.status ?? ''})`);
      setErrorDetail(msg);
      setStatus('error');
    }
  };

  const startPolling = (tid) => {
    let cnt = 0;
    pollingRef.current = setInterval(async () => {
      if (!isMounted.current) return;
      try {
        const res = await axios.get(`/explain/image/result/${tid}`);
        cnt = 0;
        const d = res.data;
        if (d === null || d === undefined) return;
        let loc = null;
        if (typeof d === 'string' && d.trim()) {
          const isUUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(d.trim());
          if (!isUUID) loc = d.trim();
        } else if (typeof d === 'object') {
          loc = d.cam_loc ?? d.result_loc ?? d.image_loc ?? d.heatmap_loc ?? d.file_loc ?? d.result_path ?? d.path ?? d.url ?? d.output_path ?? null;
          if (!loc) {
            const st = (d.status || '').toUpperCase();
            if (st === 'FAILED' || st === 'ERROR') { stopPolling(); clearInterval(timerRef.current); setErrorMsg('히트맵 생성 실패'); setErrorDetail(d.result_msg || d.message || '서버 오류'); setStatus('error'); return; }
            return;
          }
        }
        if (loc) { stopPolling(); clearInterval(timerRef.current); setHeatmapSrc(toAbsoluteUrl(loc)); setStatus('done'); }
      } catch (e) {
        cnt++;
        if (cnt >= 5) { stopPolling(); clearInterval(timerRef.current); setErrorMsg('서버 연결 실패'); setErrorDetail(e.message); setStatus('error'); }
      }
    }, POLL_INTERVAL);
  };

  const isFake = (prob ?? 0) > 0.5;
  const isProcessing = status === 'submitting' || status === 'polling';
  const imageSrc = image_loc ? toAbsoluteUrl(image_loc) : null;

  if (!state) return (
    <div style={{ backgroundColor: '#000', minHeight: '100vh', display: 'flex', justifyContent: 'center', alignItems: 'center', color: 'white' }}>
      <p>전송된 이미지 데이터가 없습니다.</p>
      <button onClick={() => navigate(-1)} style={{ marginLeft: '15px', padding: '8px 16px', backgroundColor: '#1A2C50', color: '#39FF14', border: 'none', borderRadius: '6px', cursor: 'pointer' }}>뒤로가기</button>
    </div>
  );

  return (
    <div style={{ backgroundColor: '#000', minHeight: '100vh', width: '100vw', color: 'white', fontFamily: 'sans-serif', overflowX: 'hidden', position: 'relative' }}>
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
            ← 분석 결과로
          </button>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ padding: '6px 16px', borderRadius: '6px', fontSize: '12px', fontWeight: '900', letterSpacing: '2px', backgroundColor: isFake ? 'rgba(255,75,75,0.1)' : 'rgba(57,255,20,0.08)', color: isFake ? '#FF4B4B' : '#39FF14', border: `1px solid ${isFake ? 'rgba(255,75,75,0.4)' : 'rgba(57,255,20,0.3)'}` }}>
              {label || (isFake ? 'FAKE' : 'REAL')}
            </div>
            <div style={{ padding: '6px 16px', backgroundColor: '#0A0A0A', borderRadius: '6px', fontSize: '12px', color: '#39FF14', border: '1px solid #1A1A1A', fontWeight: 'bold', letterSpacing: '1px' }}>IMAGE FORGERY TRACE</div>
            {sessionUser && (
              <div style={{ backgroundColor: '#0A0A0A', padding: '6px 14px', borderRadius: '6px', border: '1px solid #1A1A1A', fontSize: '12px' }}>
                <span style={{ color: '#555' }}>담당: </span>
                <span style={{ color: '#39FF14', fontWeight: 'bold' }}>{sessionUser.name}</span>
              </div>
            )}
          </div>
        </header>

        {/* 스탯 바 */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px', marginBottom: '24px' }}>
          {[
            { label: 'FAKE 확률', value: `${((prob ?? 0) * 100).toFixed(1)}%`, color: isFake ? '#FF4B4B' : '#39FF14', big: true },
            { label: 'VERDICT',   value: label || (isFake ? 'FAKE' : 'REAL'),  color: isFake ? '#FF4B4B' : '#39FF14', big: true },
            { label: 'MODEL',     value: (selectedModel || model_type || 'fast').toUpperCase(), color: '#fff' },
          ].map((item, i) => (
            <div key={i} style={{ backgroundColor: '#050505', border: '1px solid #1A1A1A', borderRadius: '12px', padding: '18px 22px', position: 'relative', overflow: 'hidden' }}>
              <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '2px', background: `linear-gradient(90deg, transparent, ${item.color}40, transparent)` }} />
              <p style={{ color: '#444', fontSize: '10px', fontWeight: 'bold', letterSpacing: '2px', margin: '0 0 6px 0' }}>{item.label}</p>
              <p style={{ color: item.color, fontWeight: '900', fontFamily: 'monospace', fontSize: item.big ? '22px' : '16px', margin: 0 }}>{item.value}</p>
            </div>
          ))}
        </div>

        {/* 메인 2열 */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>

          {/* 원본 이미지 */}
          {imageSrc && (
            <div style={{ backgroundColor: '#050505', borderRadius: '16px', border: '1px solid #1A1A1A', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
              <div style={{ padding: '14px 22px', borderBottom: '1px solid #111', display: 'flex', alignItems: 'center', gap: '10px', backgroundColor: '#030303' }}>
                <div style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: '#39FF14', boxShadow: '0 0 6px #39FF14' }} />
                <span style={{ color: '#555', fontSize: '11px', fontWeight: 'bold', letterSpacing: '2px' }}>ORIGINAL IMAGE</span>
              </div>
              <div style={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '28px', minHeight: '420px' }}>
                <img src={imageSrc} alt="Original"
                  style={{ maxWidth: '100%', maxHeight: '480px', borderRadius: '8px', objectFit: 'contain', boxShadow: '0 8px 32px rgba(0,0,0,0.6)' }}
                  onError={(e) => { e.target.style.display = 'none'; }}
                />
              </div>
            </div>
          )}

          {/* 히트맵 패널 */}
          <div style={{ backgroundColor: '#050505', borderRadius: '16px', border: `1px solid ${isProcessing ? 'rgba(57,255,20,0.2)' : status === 'done' ? 'rgba(57,255,20,0.15)' : '#1A1A1A'}`, display: 'flex', flexDirection: 'column', overflow: 'hidden', position: 'relative', transition: 'border-color 0.5s' }}>
            {isProcessing && <div style={{ position: 'absolute', top: 0, left: 0, right: 0, height: '2px', background: 'linear-gradient(90deg, transparent, #39FF14, transparent)', animation: 'slideRight 2s linear infinite', zIndex: 2 }} />}

            <div style={{ padding: '14px 22px', borderBottom: '1px solid #111', display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: '#030303' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <div style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: isProcessing ? '#39FF14' : status === 'done' ? '#39FF14' : '#333', boxShadow: isProcessing ? '0 0 8px #39FF14' : 'none', animation: isProcessing ? 'pulse 1.5s ease-in-out infinite' : 'none' }} />
                <span style={{ color: '#555', fontSize: '11px', fontWeight: 'bold', letterSpacing: '2px' }}>HEATMAP + BBOX OVERLAY</span>
                {selectedBranch && <span style={{ color: '#2A2A2A', fontSize: '10px', fontFamily: 'monospace' }}>{selectedBranch.toUpperCase()} / {(selectedModel || '').toUpperCase()}</span>}
              </div>
              <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
                {isProcessing && <span style={{ fontSize: '11px', color: '#39FF14', fontFamily: 'monospace' }}>{elapsed}s</span>}
                {status === 'done' && heatmapSrc && (
                  <a href={heatmapSrc} download={`heatmap_img${image_id}_${selectedBranch}_${selectedModel}.png`}
                    style={{ fontSize: '11px', color: '#39FF14', textDecoration: 'none', border: '1px solid rgba(57,255,20,0.3)', padding: '4px 12px', borderRadius: '4px', fontWeight: 'bold' }}>
                    ↓ SAVE
                  </a>
                )}
              </div>
            </div>

            <div style={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '28px', minHeight: '420px' }}>

              {/* 옵션 선택 */}
              {(status === 'idle' || status === 'error') && !isProcessing && (
                <div style={{ textAlign: 'center', width: '100%', maxWidth: '360px' }}>
                  {status === 'error' && (
                    <div style={{ marginBottom: '28px' }}>
                      <div style={{ width: '48px', height: '48px', borderRadius: '50%', border: '1px solid rgba(255,75,75,0.3)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 12px', fontSize: '20px', backgroundColor: 'rgba(255,75,75,0.06)' }}>⚠</div>
                      <p style={{ fontSize: '13px', fontWeight: 'bold', color: '#FF4B4B', marginBottom: '6px' }}>{errorMsg}</p>
                      {errorDetail && <p style={{ fontSize: '10px', color: '#444', lineHeight: '1.7', fontFamily: 'monospace', wordBreak: 'break-all', backgroundColor: '#0A0A0A', padding: '10px', borderRadius: '6px', border: '1px solid #1A1A1A', textAlign: 'left', marginBottom: '20px' }}>{errorDetail}</p>}
                    </div>
                  )}

                  {/* Branch 선택 */}
                  <p style={{ color: '#444', fontSize: '10px', letterSpacing: '2px', fontWeight: 'bold', margin: '0 0 10px 0' }}>BRANCH LEVEL</p>
                  <div style={{ display: 'flex', gap: '10px', justifyContent: 'center', marginBottom: '22px' }}>
                    {Object.keys(BRANCH_OPTIONS).map(branch => (
                      <button key={branch} onClick={() => setTempBranch(branch)}
                        style={{ padding: '12px 28px', backgroundColor: tempBranch === branch ? 'rgba(57,255,20,0.08)' : 'transparent', border: `1px solid ${tempBranch === branch ? '#39FF14' : '#333'}`, borderRadius: '8px', color: tempBranch === branch ? '#39FF14' : '#555', fontSize: '13px', fontWeight: 'bold', letterSpacing: '1px', cursor: 'pointer', transition: 'all 0.15s', textTransform: 'uppercase' }}>
                        {branch}
                        <span style={{ display: 'block', fontSize: '9px', color: tempBranch === branch ? 'rgba(57,255,20,0.5)' : '#2A2A2A', fontWeight: 'normal', marginTop: '3px' }}>
                          {branch === 'low' ? '국소 흔적' : '전역 구조'}
                        </span>
                      </button>
                    ))}
                  </div>

                  {/* Model 선택 */}
                  <p style={{ color: '#444', fontSize: '10px', letterSpacing: '2px', fontWeight: 'bold', margin: '0 0 10px 0' }}>MODEL TYPE</p>
                  <div style={{ display: 'flex', gap: '10px', justifyContent: 'center', marginBottom: '28px' }}>
                    {MODEL_OPTIONS.map(m => (
                      <button key={m} onClick={() => setTempModel(m)}
                        style={{ padding: '12px 28px', backgroundColor: tempModel === m ? 'rgba(57,255,20,0.08)' : 'transparent', border: `1px solid ${tempModel === m ? '#39FF14' : '#333'}`, borderRadius: '8px', color: tempModel === m ? '#39FF14' : '#555', fontSize: '13px', fontWeight: 'bold', letterSpacing: '1px', cursor: 'pointer', transition: 'all 0.15s', textTransform: 'uppercase' }}>
                        {m}
                      </button>
                    ))}
                  </div>

                  {/* 실행 버튼 */}
                  <button
                    onClick={() => handleBranchSelect(tempBranch, tempModel)}
                    style={{ padding: '14px 52px', backgroundColor: '#1A2C50', border: '1px solid #2A4C80', borderRadius: '10px', color: '#fff', fontSize: '14px', fontWeight: 'bold', letterSpacing: '2px', cursor: 'pointer', transition: 'all 0.2s' }}
                    onMouseEnter={e => { e.currentTarget.style.backgroundColor = '#243870'; }}
                    onMouseLeave={e => { e.currentTarget.style.backgroundColor = '#1A2C50'; }}>
                    GENERATE
                  </button>
                </div>
              )}

              {/* 로딩 */}
              {isProcessing && (
                <div style={{ textAlign: 'center', padding: '40px 0' }}>
                  <div style={{ width: '48px', height: '48px', border: '3px solid #1A1A1A', borderTop: '3px solid #39FF14', borderRadius: '50%', margin: '0 auto 24px', animation: 'spin 0.8s linear infinite' }} />
                  <p style={{ fontSize: '13px', color: '#39FF14', letterSpacing: '2px', fontWeight: 'bold', margin: '0 0 8px 0' }}>GENERATING HEATMAP</p>
                  <p style={{ fontSize: '11px', color: '#444', margin: '0 0 6px 0' }}>위조 흔적 시각화 처리 중...</p>
                  <p style={{ fontSize: '11px', color: '#333', fontFamily: 'monospace', margin: '0 0 4px 0' }}>{(selectedBranch || tempBranch || '').toUpperCase()} BRANCH · {(selectedModel || tempModel || '').toUpperCase()} MODEL</p>
                  <p style={{ fontSize: '12px', color: '#555', fontFamily: 'monospace', margin: 0 }}>{elapsed}s elapsed</p>
                </div>
              )}

              {/* 결과 */}
              {status === 'done' && heatmapSrc && (
                <div style={{ width: '100%' }}>
                  <div style={{ position: 'relative' }}>
                    <div style={{ position: 'absolute', top: 0, right: 0, zIndex: 2, padding: '4px 10px', backgroundColor: 'rgba(57,255,20,0.1)', border: '1px solid rgba(57,255,20,0.3)', borderRadius: '6px', fontSize: '10px', color: '#39FF14', fontWeight: 'bold' }}>
                      ✓ {(selectedBranch || '').toUpperCase()} / {(selectedModel || '').toUpperCase()}
                    </div>
                    <img src={heatmapSrc} alt="Heatmap"
                      style={{ maxWidth: '100%', maxHeight: '480px', borderRadius: '10px', objectFit: 'contain', display: 'block', margin: '0 auto', boxShadow: '0 0 40px rgba(57,255,20,0.08), 0 8px 32px rgba(0,0,0,0.6)' }}
                      onError={(e) => { e.target.style.display = 'none'; setErrorMsg('이미지 로드 실패'); setErrorDetail(`URL: ${heatmapSrc}`); setStatus('error'); }}
                    />
                  </div>
                  {/* 재선택 버튼 */}
                  <div style={{ display: 'flex', gap: '8px', justifyContent: 'center', marginTop: '16px', flexWrap: 'wrap' }}>
                    {Object.keys(BRANCH_OPTIONS).flatMap(branch => MODEL_OPTIONS.map(m => ({ branch, m }))).map(({ branch, m }) => (
                      <button key={branch + m} onClick={() => handleBranchSelect(branch, m)}
                        style={{ padding: '6px 14px', backgroundColor: (selectedBranch === branch && selectedModel === m) ? 'rgba(57,255,20,0.1)' : 'transparent', border: `1px solid ${(selectedBranch === branch && selectedModel === m) ? '#39FF14' : '#2A2A2A'}`, borderRadius: '6px', color: (selectedBranch === branch && selectedModel === m) ? '#39FF14' : '#444', fontSize: '10px', fontWeight: 'bold', cursor: 'pointer', letterSpacing: '1px', transition: 'all 0.15s' }}>
                        {branch.toUpperCase()} / {m.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {status === 'done' && !heatmapSrc && (
                <div style={{ textAlign: 'center' }}>
                  <p style={{ color: '#555', marginBottom: '16px' }}>결과 이미지를 받지 못했습니다.</p>
                  <button onClick={() => requestHeatmap(selectedBranch, selectedModel)} style={{ padding: '10px 24px', backgroundColor: 'transparent', color: '#39FF14', border: '1px solid rgba(57,255,20,0.4)', borderRadius: '8px', cursor: 'pointer', fontSize: '13px', fontWeight: 'bold' }}>재시도</button>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes scanLine   { 0% { top:-2px; opacity:0; } 10% { opacity:1; } 90% { opacity:1; } 100% { top:100%; opacity:0; } }
        @keyframes slideRight { 0% { transform:translateX(-100%); } 100% { transform:translateX(100%); } }
        @keyframes spin       { from { transform:rotate(0deg); } to { transform:rotate(360deg); } }
        @keyframes pulse      { 0%,100% { opacity:1; box-shadow:0 0 8px #39FF14; } 50% { opacity:0.4; box-shadow:0 0 3px #39FF14; } }
      `}</style>
    </div>
  );
};

export default ImageHeatmapPage;
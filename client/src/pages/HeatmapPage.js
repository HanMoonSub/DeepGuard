import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';

const apiUrl = '';
const POLL_INTERVAL = 2000;

const BRANCH_CONFIG = {
  low: {
    branch_level: 'low',
    explainer_type: 'layercam',      // low branch: LayerCAM
    display_type: 'heatmap_bbox',
    overlay_ratio: 0.7,
    threshold: 0.9,
    aug_smooth: false,
    eigen_smooth: true,
  },
  high: {
    branch_level: 'high',
    explainer_type: 'xgradcam',      // high branch: XGradCAM (허용값: gradcamelementwise, layercam, xgradcam)
    display_type: 'heatmap_bbox',
    overlay_ratio: 0.7,
    threshold: 0.9,
    aug_smooth: false,
    eigen_smooth: true,
  },
};

// API 응답에서 이미지 경로 추출
// POST /explain/video/{id}/frame/{idx} → 202, body = "task_id_string"
// GET  /explain/frame/result/{task_id} → 200, body = "image_path_string" or { result_loc, ... }
const extractTaskId = (data) => {
  if (typeof data === 'string') return data;
  return data?.task_id || data?.id || null;
};

const extractImagePath = (data) => {
  if (typeof data === 'string') return data;
  return data?.result_loc || data?.image_loc || data?.heatmap_loc || data?.file_loc || null;
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

  const [activeBranch, setActiveBranch] = useState('high');
  // status: 'idle' | 'submitting' | 'polling' | 'done' | 'error'
  const [status, setStatus] = useState('idle');
  const [heatmapSrc, setHeatmapSrc] = useState(null);
  const [taskId, setTaskId] = useState(null);
  const [errorMsg, setErrorMsg] = useState('');
  const [errorDetail, setErrorDetail] = useState(''); // 개발용 상세 에러

  const pollingRef = useRef(null);
  const isMounted = useRef(true);

  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
      stopPolling();
    };
  }, []);

  const stopPolling = () => {
    if (pollingRef.current) { clearInterval(pollingRef.current); pollingRef.current = null; }
  };

  const handleBranchChange = (branch) => {
    if (status === 'submitting' || status === 'polling') return;
    stopPolling();
    setActiveBranch(branch);
    setStatus('idle');
    setHeatmapSrc(null);
    setTaskId(null);
    setErrorMsg('');
    setErrorDetail('');
  };

  // ── STEP 1: POST 접수 → task_id 수신 ──
  const requestHeatmap = async () => {
    if (video_id == null) {
      setErrorMsg('video_id가 없습니다.');
      setErrorDetail('state에 video_id가 전달되지 않았습니다.');
      setStatus('error');
      return;
    }

    stopPolling();
    setStatus('submitting');
    setHeatmapSrc(null);
    setErrorMsg('');
    setErrorDetail('');
    setTaskId(null);

    const body = {
      model_type: model_type || 'fast',
      ...BRANCH_CONFIG[activeBranch],
    };

    try {
      // POST /explain/video/{video_id}/frame/{frame_index}
      // Response 202: "task_id_string"
      const res = await axios.post(
        `/explain/video/${video_id}/frame/${frame_index ?? 0}`,
        body
      );

      const tid = extractTaskId(res.data);
      if (!tid) {
        throw new Error(`task_id 추출 실패. 응답: ${JSON.stringify(res.data)}`);
      }

      setTaskId(tid);
      setStatus('polling');
      startPolling(tid);
    } catch (e) {
      if (!isMounted.current) return;
      const msg = e.response?.data?.detail || e.response?.data || e.message || '요청 실패';
      setErrorMsg('히트맵 생성 요청 실패');
      setErrorDetail(typeof msg === 'string' ? msg : JSON.stringify(msg));
      setStatus('error');
    }
  };

  // ── STEP 2: GET 폴링 → 결과 이미지 수신 ──
  const startPolling = (tid) => {
    let networkErrorCount = 0;
    const MAX_ERRORS = 5;

    pollingRef.current = setInterval(async () => {
      if (!isMounted.current) return;
      try {
        const res = await axios.get(`/explain/frame/result/${tid}`);
        networkErrorCount = 0;
        const d = res.data;

        // 디버그: 실제 응답 구조 확인 (확인 후 제거 가능)
        console.log('[heatmap polling] response:', JSON.stringify(d));

        if (d === null || d === undefined) return;

        let loc = null;

        if (typeof d === 'string' && d.length > 0) {
          // 응답이 문자열: 이미지 경로인지 확인
          if (d.includes('/') || d.match(/\.(png|jpg|jpeg|webp)/i)) {
            loc = d;
          }
        } else if (typeof d === 'object') {
          // 응답이 객체: 가능한 모든 경로 필드 시도
          loc = d.result_loc ?? d.image_loc ?? d.heatmap_loc ?? d.file_loc
              ?? d.result_path ?? d.path ?? d.url ?? d.image_url ?? d.output_path
              ?? d.result ?? null;

          if (!loc) {
            const st = (d.status || '').toUpperCase();
            if (st === 'FAILED' || st === 'ERROR') {
              stopPolling();
              setErrorMsg('히트맵 생성 실패');
              setErrorDetail(d.result_msg || d.message || d.detail || '서버 오류');
              setStatus('error');
              return;
            }
            // PENDING / STARTED → 계속 폴링
            return;
          }
        }

        if (loc) {
          stopPolling();
          setHeatmapSrc(toAbsoluteUrl(loc));
          setStatus('done');
        }
      } catch (e) {
        networkErrorCount++;
        console.warn(`[heatmap polling] 에러 ${networkErrorCount}/${MAX_ERRORS}:`, e.message);
        if (networkErrorCount >= MAX_ERRORS) {
          stopPolling();
          setErrorMsg('서버 연결 실패');
          setErrorDetail(`${MAX_ERRORS}회 연속 연결 오류. 백엔드 서버 상태를 확인해주세요. (${e.message})`);
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
    <div style={{ backgroundColor: '#000', minHeight: '100vh', width: '100vw', color: 'white', padding: '40px 80px', boxSizing: 'border-box', fontFamily: 'sans-serif', overflowX: 'hidden' }}>

      {/* 헤더 */}
      <header style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '40px', alignItems: 'center' }}>
        <button onClick={() => navigate(-1)} style={{ color: '#888', background: 'none', border: 'none', cursor: 'pointer', fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '20px' }}>←</span> 타임라인으로
        </button>
        <div style={{ display: 'flex', gap: '15px', alignItems: 'center' }}>
          <span style={{ padding: '6px 14px', backgroundColor: '#111', borderRadius: '4px', fontSize: '12px', color: '#39FF14', border: '1px solid #222', fontWeight: 'bold', letterSpacing: '0.5px' }}>
            FORGERY TRACE VISUALIZATION
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

        {/* 프레임 메타 패널 */}
        <div style={{ display: 'flex', backgroundColor: '#050505', borderRadius: '16px', border: '1px solid #1A1A1A', padding: '24px', gap: '32px', alignItems: 'center' }}>
          <div style={{ flex: 1 }}>
            <p style={{ color: '#444', fontSize: '11px', letterSpacing: '2px', margin: '0 0 8px 0', fontWeight: 'bold' }}>TARGET METADATA</p>
            <h3 style={{ fontSize: '22px', margin: '0 0 20px 0', color: '#fff', fontWeight: '700' }}>{video_name || `STREAM_INSTANCE_${video_id}`}</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '14px' }}>
              <div style={{ display: 'flex', borderBottom: '1px solid #111', paddingBottom: '8px' }}>
                <span style={{ width: '160px', color: '#555', fontWeight: 'bold', fontSize: '12px' }}>FRAME INDEX</span>
                <span style={{ color: '#fff', fontFamily: 'monospace' }}>#{frame_index ?? 0}</span>
              </div>
              <div style={{ display: 'flex', borderBottom: '1px solid #111', paddingBottom: '8px' }}>
                <span style={{ width: '160px', color: '#555', fontWeight: 'bold', fontSize: '12px' }}>TIMESTAMP</span>
                <span style={{ color: '#39FF14', fontWeight: '600', fontFamily: 'monospace' }}>{timestamp || '—'}</span>
              </div>
              <div style={{ display: 'flex', borderBottom: '1px solid #111', paddingBottom: '8px' }}>
                <span style={{ width: '160px', color: '#555', fontWeight: 'bold', fontSize: '12px' }}>FORGERY RISK</span>
                <span style={{ color: isFake ? '#FF4B4B' : '#39FF14', fontWeight: '700', fontFamily: 'monospace' }}>
                  {Number(fake_score ?? 0).toFixed(1)}%
                </span>
              </div>
              <div style={{ display: 'flex', paddingBottom: '4px' }}>
                <span style={{ width: '160px', color: '#555', fontWeight: 'bold', fontSize: '12px' }}>VERDICT</span>
                <span style={{ color: isFake ? '#FF4B4B' : '#39FF14', fontWeight: '700', fontFamily: 'monospace' }}>
                  {isFake ? 'FAKE' : 'REAL'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* 메인 2열 */}
        <div style={{ display: 'flex', gap: '30px', alignItems: 'flex-start' }}>

          {/* 좌: 히트맵 결과 뷰어 */}
          <div style={{ flex: 1.4, backgroundColor: '#050505', borderRadius: '16px', border: '1px solid #1A1A1A', display: 'flex', flexDirection: 'column', minHeight: '520px' }}>

            {/* 뷰어 헤더 */}
            <div style={{ padding: '16px 24px', borderBottom: '1px solid #111', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ color: '#444', fontSize: '11px', fontWeight: 'bold', letterSpacing: '1.5px' }}>HEATMAP + BBOX OVERLAY</span>
              {status === 'done' && heatmapSrc && (
                <a
                  href={heatmapSrc}
                  download={`heatmap_f${frame_index}_${activeBranch}.png`}
                  style={{ fontSize: '11px', color: '#39FF14', textDecoration: 'none', border: '1px solid rgba(57,255,20,0.3)', padding: '4px 12px', borderRadius: '4px', fontWeight: 'bold' }}
                >
                  ↓ SAVE
                </a>
              )}
            </div>

            {/* 뷰어 본문 */}
            <div style={{ flex: 1, display: 'flex', justifyContent: 'center', alignItems: 'center', padding: '30px', minHeight: '460px' }}>

              {/* idle */}
              {status === 'idle' && (
                <div style={{ textAlign: 'center', color: '#333' }}>
                  <p style={{ fontSize: '14px', letterSpacing: '2px', fontWeight: 'bold' }}>BRANCH LEVEL을 선택하고</p>
                  <p style={{ fontSize: '14px', letterSpacing: '2px', fontWeight: 'bold', marginTop: '6px' }}>GENERATE를 실행하세요</p>
                </div>
              )}

              {/* 처리중 */}
              {isProcessing && (
                <div style={{ textAlign: 'center', width: '100%' }}>
                  <div style={{ position: 'relative', width: '180px', height: '180px', margin: '0 auto 30px', backgroundColor: '#0A0A0A', borderRadius: '12px', border: '1px solid #222', overflow: 'hidden' }}>
                    <div style={{ position: 'absolute', width: '100%', height: '2px', background: 'linear-gradient(90deg, transparent, #39FF14, transparent)', boxShadow: '0 0 12px #39FF14', animation: 'scanLine 2s ease-in-out infinite' }} />
                    <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <span style={{ fontSize: '9px', color: '#39FF14', letterSpacing: '2px', fontWeight: 'bold' }}>
                        {status === 'submitting' ? '서버 접수 중...' : '히트맵 생성 중...'}
                      </span>
                    </div>
                  </div>
                  <p style={{ fontSize: '11px', color: '#39FF14', letterSpacing: '3px', fontWeight: 'bold', marginBottom: '6px' }}>
                    {status === 'submitting' ? 'SUBMITTING...' : 'GENERATING HEATMAP...'}
                  </p>
                  <p style={{ fontSize: '11px', color: '#444' }}>
                    {activeBranch.toUpperCase()} BRANCH · LAYERCAM · heatmap_bbox
                  </p>
                  {taskId && (
                    <p style={{ fontSize: '10px', color: '#333', marginTop: '8px', fontFamily: 'monospace' }}>TASK: {taskId}</p>
                  )}
                  <div style={{ width: '180px', height: '2px', backgroundColor: '#111', borderRadius: '1px', margin: '20px auto 0', overflow: 'hidden' }}>
                    <div style={{ width: '100%', height: '100%', background: 'linear-gradient(90deg, #1A2C50, #39FF14, #1A2C50)', backgroundSize: '200%', animation: 'loadingBar 1.5s linear infinite' }} />
                  </div>
                </div>
              )}

              {/* 결과 이미지 */}
              {status === 'done' && heatmapSrc && (
                <div style={{ width: '100%', textAlign: 'center' }}>
                  <img
                    src={heatmapSrc}
                    alt="HeatMap + BBox"
                    style={{ maxWidth: '100%', maxHeight: '480px', borderRadius: '10px', objectFit: 'contain', display: 'block', margin: '0 auto' }}
                    onError={(e) => {
                      e.target.style.display = 'none';
                      setErrorMsg('이미지 로드 실패');
                      setErrorDetail(`URL: ${heatmapSrc}`);
                      setStatus('error');
                    }}
                  />
                </div>
              )}

              {/* 결과 없음 */}
              {status === 'done' && !heatmapSrc && (
                <div style={{ textAlign: 'center', color: '#555' }}>
                  <p style={{ fontSize: '14px' }}>결과 이미지를 받지 못했습니다.</p>
                  <button onClick={requestHeatmap} style={{ marginTop: '16px', padding: '8px 20px', backgroundColor: 'transparent', color: '#39FF14', border: '1px solid #39FF14', borderRadius: '6px', cursor: 'pointer', fontSize: '12px', fontWeight: 'bold' }}>재시도</button>
                </div>
              )}

              {/* 오류 */}
              {status === 'error' && (
                <div style={{ textAlign: 'center' }}>
                  <p style={{ fontSize: '36px', marginBottom: '16px' }}>⚠</p>
                  <p style={{ fontSize: '14px', fontWeight: 'bold', color: '#FF4B4B', marginBottom: '8px' }}>{errorMsg || '분석 실패'}</p>
                  {errorDetail && (
                    <p style={{ fontSize: '11px', color: '#555', maxWidth: '340px', lineHeight: '1.8', margin: '0 auto 16px', fontFamily: 'monospace', wordBreak: 'break-all' }}>
                      {errorDetail}
                    </p>
                  )}
                  <button
                    onClick={requestHeatmap}
                    style={{ padding: '8px 20px', backgroundColor: 'transparent', color: '#FF4B4B', border: '1px solid #FF4B4B', borderRadius: '6px', cursor: 'pointer', fontSize: '12px', fontWeight: 'bold' }}
                  >
                    재시도
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* 우: 컨트롤 패널 */}
          <div style={{ flex: 0.7, display: 'flex', flexDirection: 'column', gap: '20px' }}>

            {/* 브랜치 선택 */}
            <div style={{ backgroundColor: '#0D0D0D', borderRadius: '16px', border: '1px solid #1A1A1A', padding: '28px' }}>
              <p style={{ color: '#555', fontSize: '12px', fontWeight: 'bold', letterSpacing: '1.5px', margin: '0 0 16px 0' }}>BRANCH LEVEL</p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                {[
                  { key: 'low',  desc: '국소 위조 흔적 포착 (Subtle Artifacts)' },
                  { key: 'high', desc: '전역 의미 구조 포착 (Global Artifacts)' },
                ].map(({ key, desc }) => (
                  <button
                    key={key}
                    onClick={() => handleBranchChange(key)}
                    disabled={isProcessing}
                    style={{
                      padding: '16px 20px',
                      backgroundColor: activeBranch === key ? 'rgba(57,255,20,0.06)' : 'transparent',
                      border: `1px solid ${activeBranch === key ? '#39FF14' : '#222'}`,
                      borderRadius: '10px',
                      cursor: isProcessing ? 'not-allowed' : 'pointer',
                      color: activeBranch === key ? '#39FF14' : '#555',
                      fontWeight: 'bold', fontSize: '13px', letterSpacing: '1px',
                      textAlign: 'left', transition: 'all 0.15s',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                      <span>{key.toUpperCase()} BRANCH</span>
                      {activeBranch === key && <span style={{ fontSize: '10px', color: '#39FF14' }}>● ACTIVE</span>}
                    </div>
                    <div style={{ fontSize: '11px', color: '#444', fontWeight: 'normal', lineHeight: '1.6' }}>{desc}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* 파라미터 요약 */}
            <div style={{ backgroundColor: '#0D0D0D', borderRadius: '16px', border: '1px solid #1A1A1A', padding: '28px' }}>
              <p style={{ color: '#555', fontSize: '12px', fontWeight: 'bold', letterSpacing: '1.5px', margin: '0 0 14px 0' }}>REQUEST PARAMETERS</p>
              {[
                ['model_type', model_type || 'fast'],
                ...Object.entries(BRANCH_CONFIG[activeBranch]),
              ].map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid #111', fontSize: '12px' }}>
                  <span style={{ color: '#555', fontFamily: 'monospace' }}>{k}</span>
                  <span style={{ color: '#39FF14', fontFamily: 'monospace', fontWeight: 'bold' }}>{String(v)}</span>
                </div>
              ))}
            </div>

            {/* 분석 시작 버튼 */}
            <button
              onClick={requestHeatmap}
              disabled={isProcessing}
              style={{
                padding: '18px',
                backgroundColor: isProcessing ? '#111' : '#1A2C50',
                color: isProcessing ? '#444' : 'white',
                border: 'none', borderRadius: '12px',
                fontWeight: 'bold', fontSize: '16px', letterSpacing: '1px',
                cursor: isProcessing ? 'not-allowed' : 'pointer',
                transition: 'background 0.2s',
              }}
            >
              {isProcessing ? '분석 중...' : 'GENERATE HEATMAP'}
            </button>

            {/* task_id 표시 (디버그용) */}
            {taskId && (
              <div style={{ backgroundColor: '#050505', borderRadius: '10px', border: '1px solid #111', padding: '12px 16px' }}>
                <p style={{ color: '#333', fontSize: '10px', letterSpacing: '1px', margin: '0 0 4px 0', fontWeight: 'bold' }}>TASK ID</p>
                <p style={{ color: '#444', fontSize: '10px', fontFamily: 'monospace', margin: 0, wordBreak: 'break-all' }}>{taskId}</p>
              </div>
            )}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes scanLine {
          0%   { top: -2px; opacity: 0; }
          10%  { opacity: 1; }
          90%  { opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }
        @keyframes loadingBar {
          0%   { background-position: 100% 0; }
          100% { background-position: -100% 0; }
        }
      `}</style>
    </div>
  );
};

export default HeatmapPage;
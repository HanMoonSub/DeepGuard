import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';

import Navbar from '../components/Navbar';
import useTheme from '../hooks/useTheme';
import AmbientBackground from '../components/AmbientBackground';
import Footer from '../components/Footer';
import Button from '../components/Button';
import Card from '../components/Card';
import HeatmapVisualSettings, { DEFAULT_DISPLAY } from '../components/HeatmapVisualSettings';
import './HeatmapPage.css';
import './ImageHeatmapPage.css';

const POLL_INTERVAL = 2000;

// branch별 XAI 기법 (low: 국소 흔적 / high: 전역 구조 / both: 둘을 결합)
const EXPLAINER_BY_BRANCH = { low: 'layercam', high: 'eigengradcam', both: 'fusion' };
// 버전별 표시할 branch와 기본값 (v1: MS-EffGCViT / v2: DeepClip beta)
const BRANCHES_BY_VERSION = { v1: ['low', 'high', 'both'], v2: ['low', 'high', 'both'] };
const DEFAULT_BRANCH = { v1: 'both', v2: 'both' };
const extractTaskId = (data) => {
  if (typeof data === 'string') return data;
  return data?.task_id || data?.id || null;
};
const toAbsoluteUrl = (path) => {
  if (!path) return null;
  if (path.startsWith('http') || path.startsWith('blob')) return path;
  return path;
};
// 공통 레이아웃: Navbar(light) + 본문 + Footer
const HeatmapLayout = ({ sessionUser, children }) => {
  const [theme, setTheme] = useTheme();
  return (
    <div className="dg-heatmap">
      <AmbientBackground />
      <Navbar variant="light" sessionUser={sessionUser} theme={theme} onThemeChange={setTheme} />
      <main className="dg-heatmap__main">
        <div className="dg-heatmap__wrap">{children}</div>
      </main>
      <Footer />
    </div>
  );
};

const ImageHeatmapPage = ({ sessionUser }) => {
  const { state } = useLocation();
  const navigate  = useNavigate();
  const { image_id, model_type, version_type, prob, label } = state || {};
  const modelType = model_type || 'fast'; // 분석에 사용한 모델 그대로 사용
  const versionType = BRANCHES_BY_VERSION[version_type] ? version_type : 'v1';
  const branches = BRANCHES_BY_VERSION[versionType];
  const defaultBranch = DEFAULT_BRANCH[versionType];

  const [branch, setBranch]           = useState(defaultBranch);
  const [draft, setDraft]             = useState(DEFAULT_DISPLAY);
  const [applied, setApplied]         = useState({ branch: defaultBranch, ...DEFAULT_DISPLAY });
  const [status, setStatus]           = useState('idle');
  const [heatmapSrc, setHeatmapSrc]   = useState(null);
  const [, setTaskId]                 = useState(null);
  const [errorMsg, setErrorMsg]       = useState('');
  const [errorDetail, setErrorDetail] = useState('');

  const pollingRef = useRef(null);
  const isMounted  = useRef(true);
  const startedRef = useRef(false); // StrictMode 중복 요청 방지
  const reqIdRef   = useRef(0);     // 최신 요청만 반영

  useEffect(() => {
    isMounted.current = true;
    return () => { isMounted.current = false; stopPolling(); };
  }, []);

  // 진입 즉시 생성 (버전별 기본 branch + 기본 Display 설정)
  useEffect(() => {
    if (!state || startedRef.current) return;
    startedRef.current = true;
    requestHeatmap(defaultBranch, DEFAULT_DISPLAY);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const stopPolling = () => {
    if (pollingRef.current) { clearInterval(pollingRef.current); pollingRef.current = null; }
  };

  const requestHeatmap = async (nextBranch, display) => {
    if (image_id == null) { setErrorMsg('image_id가 없습니다.'); setStatus('error'); return; }
    const myId = ++reqIdRef.current;
    stopPolling();
    setStatus('submitting');
    setHeatmapSrc(null);
    setErrorMsg('');
    setErrorDetail('');
    setTaskId(null);
    setApplied({ branch: nextBranch, display_type: display.display_type, overlay_ratio: display.overlay_ratio, threshold: display.threshold });

    const body = {
      model_type: modelType,
      branch_level: nextBranch,
      explainer_type: EXPLAINER_BY_BRANCH[nextBranch],
      display_type: display.display_type,
      overlay_ratio: display.overlay_ratio,
      threshold: display.threshold,
    };

    try {
      const res = await axios.post(`/explain/image/${image_id}`, body);
      if (myId !== reqIdRef.current) return;
      const tid = extractTaskId(res.data);
      if (!tid) throw new Error(`task_id 추출 실패. 응답: ${JSON.stringify(res.data)}`);
      setTaskId(tid);
      setStatus('polling');
      startPolling(tid, myId);
    } catch (e) {
      if (!isMounted.current || myId !== reqIdRef.current) return;
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

  const startPolling = (tid, myId) => {
    if (!isMounted.current) return;
    let cnt = 0;
    pollingRef.current = setInterval(async () => {
      if (!isMounted.current) return;
      try {
        const res = await axios.get(`/explain/image/result/${tid}`);
        if (myId !== reqIdRef.current) return;
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
            if (st === 'FAILED' || st === 'ERROR') { stopPolling(); setErrorMsg('히트맵 생성 실패'); setErrorDetail(d.result_msg || d.message || '서버 오류'); setStatus('error'); return; }
            return;
          }
        }
        if (loc) { stopPolling(); setHeatmapSrc(toAbsoluteUrl(loc)); setStatus('done'); }
      } catch (e) {
        if (myId !== reqIdRef.current) return;
        cnt++;
        if (cnt >= 5) { stopPolling(); setErrorMsg('서버 연결 실패'); setErrorDetail(e.message); setStatus('error'); }
      }
    }, POLL_INTERVAL);
  };

  const handleReset = () => { setBranch(defaultBranch); setDraft(DEFAULT_DISPLAY); };
  const handleApply = () => requestHeatmap(branch, draft);
  const handleRetry = () => requestHeatmap(applied.branch, applied);

  const isFake = (prob ?? 0) > 0.5;
  const isProcessing = status === 'submitting' || status === 'polling';
  const tone = isFake ? 'fake' : 'real';
  const verdict = label || (isFake ? 'FAKE' : 'REAL');
  const dirtyCount = [
    branch !== applied.branch,
    draft.display_type !== applied.display_type,
    draft.overlay_ratio !== applied.overlay_ratio,
    draft.threshold !== applied.threshold,
  ].filter(Boolean).length;
  const canReset = branch !== defaultBranch
    || draft.display_type !== DEFAULT_DISPLAY.display_type
    || draft.overlay_ratio !== DEFAULT_DISPLAY.overlay_ratio
    || draft.threshold !== DEFAULT_DISPLAY.threshold;
  const isDone = status === 'done' && !!heatmapSrc;

  if (!state) return (
    <HeatmapLayout sessionUser={sessionUser}>
      <Card className="dg-card--static dg-heatmap__error">
        <div className="dg-heatmap__error-icon" aria-hidden="true">!</div>
        <p className="dg-heatmap__error-title">전송된 이미지 데이터가 없습니다</p>
        <p className="dg-heatmap__error-desc">분석 결과 화면의 ‘이미지 위조 흔적 분석하기’로 들어와 주세요.</p>
        <div className="dg-heatmap__error-actions">
          <Button variant="primary" onClick={() => navigate('/analysis')}>이미지 분석으로</Button>
          <Button variant="outline" onClick={() => navigate(-1)}>뒤로가기</Button>
        </div>
      </Card>
    </HeatmapLayout>
  );

  const stats = [
    { label: '딥페이크 확률', value: `${((prob ?? 0) * 100).toFixed(1)}%`, tone },
    { label: '판정 결과',     value: verdict,                               tone },
    { label: '모델',          value: `${versionType.toUpperCase()} · ${modelType.toUpperCase()}` },
  ];

  return (
    <HeatmapLayout sessionUser={sessionUser}>
      {/* 상단: 뒤로가기 + 제목 */}
      <header className="dg-heatmap__head">
        <button type="button" className="dg-heatmap__back" onClick={() => navigate(-1)}>
          <span aria-hidden="true">←</span> 분석 결과로
        </button>
        <h2 className="dg-heatmap__title">이미지 위조 흔적 분석</h2>
      </header>

      {/* 왼쪽 히트맵 | 오른쪽 시각화 설정 + 분석 요약 */}
      <div className="dg-ihm">
        <section className={`dg-ihm__stage${isProcessing ? ' is-active' : ''}`} aria-label="히트맵 결과">
          <div className="dg-ihm__canvas">
            {isDone && (
              <img src={heatmapSrc} alt="위조 흔적 히트맵"
                onError={(e) => { e.target.style.display = 'none'; setErrorMsg('이미지 로드 실패'); setErrorDetail(`URL: ${heatmapSrc}`); setStatus('error'); }}
              />
            )}

            {/* 로딩 */}
            {(isProcessing || status === 'idle') && (
              <div className="dg-ihm__state">
                <div className="dg-heatmap__spinner" />
                <p className="dg-ihm__state-title">위조 흔적을 그리는 중이에요</p>
              </div>
            )}

            {/* 에러 */}
            {status === 'error' && (
              <div className="dg-ihm__state dg-heatmap__options">
                <div className="dg-heatmap__alert">
                  <div className="dg-heatmap__alert-icon" aria-hidden="true">!</div>
                  <p className="dg-heatmap__alert-msg">{errorMsg}</p>
                  {errorDetail && <p className="dg-heatmap__alert-detail">{errorDetail}</p>}
                </div>
                <Button variant="primary" onClick={handleRetry}>다시 시도</Button>
              </div>
            )}

            {status === 'done' && !heatmapSrc && (
              <div className="dg-ihm__state">
                <p className="dg-ihm__state-title">결과 이미지를 받지 못했어요</p>
                <Button variant="outline" onClick={handleRetry}>재시도</Button>
              </div>
            )}

            {/* 하단 바: 범례 · 현재 표시 · 저장 */}
            {isDone && (
              <div className="dg-ihm__bar">
                {applied.display_type !== 'bbox' && (
                  <div className="dg-ihm__legend" aria-label="히트맵 색 범례: 파랑 낮음, 빨강 높음">
                    <span>낮음</span><i aria-hidden="true" /><span>높음</span>
                  </div>
                )}
                <Button variant="outline" className="dg-ihm__save" href={heatmapSrc}
                  download={`heatmap_img${image_id}_${applied.branch}_${modelType}_${applied.display_type}.png`}>
                  이미지 저장
                </Button>
              </div>
            )}
          </div>
        </section>

        <aside className="dg-ihm__side">
          <HeatmapVisualSettings branches={branches} branch={branch} setBranch={setBranch}
            draft={draft} setDraft={setDraft} dirtyCount={dirtyCount} isProcessing={isProcessing}
            onApply={handleApply} onReset={handleReset} canReset={canReset} />

          <dl className="dg-ihm__summary">
            {stats.map(item => (
              <div key={item.label} className="dg-ihm__summary-row">
                <dt>{item.label}</dt>
                <dd className={item.tone ? `dg-heatmap__tone--${item.tone}` : ''}>{item.value}</dd>
              </div>
            ))}
          </dl>
        </aside>
      </div>
    </HeatmapLayout>
  );
};

export default ImageHeatmapPage;

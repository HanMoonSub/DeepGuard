import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';

import Navbar from '../components/Navbar';
import useTheme from '../hooks/useTheme';
import Footer from '../components/Footer';
import Button from '../components/Button';
import Card from '../components/Card';
import HeatmapVisualSettings, { DEFAULT_DISPLAY } from '../components/HeatmapVisualSettings';
import AmbientBackground from '../components/AmbientBackground';
import HeatmapZoomImage from '../components/HeatmapZoomImage';
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

// 구간 시각 → 'N초' (60초 이상은 'M분 N초'). frame_time 우선, 없으면 'mm:ss' 타임스탬프 해석
const secondsOf = (f) => {
  if (!f) return null;
  if (f.frame_time != null && Number.isFinite(Number(f.frame_time))) return Math.round(Number(f.frame_time));
  const m = /^(\d+):(\d{1,2})$/.exec(String(f.timestamp || '').trim());
  return m ? Number(m[1]) * 60 + Number(m[2]) : null;
};
const secLabel = (f) => {
  const s = secondsOf(f);
  if (s == null) return f?.timestamp || '';
  return s >= 60 ? `${Math.floor(s / 60)}분 ${s % 60}초` : `${s}초`;
};

const HeatmapPage = ({ sessionUser }) => {
  const { state } = useLocation();
  const navigate  = useNavigate();
  const { video_id, frame_index, timestamp, frame_time, fake_score, model_type, version_type } = state || {};
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
  const requestedFrameRef = useRef(-1); // 마지막으로 요청한 frame_index (StrictMode 중복 요청 방지)
  const appliedRef = useRef(null);      // 구간 이동 시 그대로 쓸 현재 적용 설정
  const reqIdRef   = useRef(0);         // 최신 요청만 반영

  // 타임라인에서 넘어온 구간 목록 (없으면 현재 구간 하나)
  const frames = Array.isArray(state?.frames) && state.frames.length
    ? state.frames
    : (state ? [{ frame_index, timestamp, frame_time, fake_score }] : []);
  const curPos = frames.findIndex(f => f.frame_index === frame_index);
  const prevFrame = curPos > 0 ? frames[curPos - 1] : null;
  const nextFrame = curPos >= 0 && curPos < frames.length - 1 ? frames[curPos + 1] : null;

  useEffect(() => {
    isMounted.current = true;
    return () => { isMounted.current = false; stopPolling(); };
  }, []);

  // 진입 시 기본 설정으로 생성, 이후 구간을 옮기면 지금 적용된 설정 그대로 다시 생성
  useEffect(() => {
    if (!state) return;
    const key = frame_index ?? 0;
    if (requestedFrameRef.current === key) return;
    const isFirst = requestedFrameRef.current === -1;
    requestedFrameRef.current = key;
    if (isFirst || !appliedRef.current) requestHeatmap(defaultBranch, DEFAULT_DISPLAY);
    else requestHeatmap(appliedRef.current.branch, appliedRef.current);
  }, [frame_index]); // eslint-disable-line react-hooks/exhaustive-deps

  // 이전/다음 구간으로 이동 — 히스토리를 쌓지 않아 '타임라인으로'는 그대로 한 번에 돌아감
  const goFrame = (f) => {
    if (!f) return;
    navigate('/heatmap', {
      replace: true,
      state: { ...state, frame_index: f.frame_index, timestamp: f.timestamp, frame_time: f.frame_time, fake_score: f.fake_score },
    });
  };

  const stopPolling = () => {
    if (pollingRef.current) { clearInterval(pollingRef.current); pollingRef.current = null; }
  };

  const requestHeatmap = async (nextBranch, display) => {
    if (video_id == null) { setErrorMsg('video_id가 없습니다.'); setStatus('error'); return; }
    const myId = ++reqIdRef.current;
    stopPolling();
    setStatus('submitting');
    setHeatmapSrc(null);
    setErrorMsg('');
    setErrorDetail('');
    setTaskId(null);
    const nextApplied = { branch: nextBranch, display_type: display.display_type, overlay_ratio: display.overlay_ratio, threshold: display.threshold };
    appliedRef.current = nextApplied;
    setApplied(nextApplied);

    const body = {
      model_type: modelType,
      branch_level: nextBranch,
      explainer_type: EXPLAINER_BY_BRANCH[nextBranch],
      display_type: display.display_type,
      overlay_ratio: display.overlay_ratio,
      threshold: display.threshold,
    };

    try {
      const res = await axios.post(`/explain/video/${video_id}/frame/${frame_index ?? 0}`, body);
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
        const res = await axios.get(`/explain/frame/result/${tid}`);
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

  const isFake = (fake_score ?? 0) > 50;
  const isProcessing = status === 'submitting' || status === 'polling';
  const tone = isFake ? 'fake' : 'real';
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
        <p className="dg-heatmap__error-title">전송된 프레임 데이터가 없습니다</p>
        <p className="dg-heatmap__error-desc">타임라인 화면에서 확인할 프레임의 ‘히트맵’ 버튼으로 들어와 주세요.</p>
        <div className="dg-heatmap__error-actions">
          <Button variant="primary" onClick={() => navigate('/video-analysis')}>동영상 분석으로</Button>
          <Button variant="outline" onClick={() => navigate(-1)}>뒤로가기</Button>
        </div>
      </Card>
    </HeatmapLayout>
  );

  // 시각화 설정 위 요약 (이미지 위조 흔적과 같은 구성)
  const stats = [
    { label: '딥페이크 확률', value: `${Number(fake_score ?? 0).toFixed(1)}%`, tone },
    { label: '판정 결과',     value: isFake ? 'FAKE' : 'REAL',                 tone },
    { label: '모델',          value: `${versionType.toUpperCase()} · ${modelType.toUpperCase()}` },
  ];
  const curFrame = curPos >= 0 ? frames[curPos] : { timestamp, frame_time };

  return (
    <HeatmapLayout sessionUser={sessionUser}>
      {/* 상단: 뒤로가기 + 제목 */}
      <header className="dg-heatmap__head">
        <button type="button" className="dg-heatmap__back" onClick={() => navigate(-1)}>
          <span aria-hidden="true">←</span> 타임라인으로
        </button>
        <h2 className="dg-heatmap__title">프레임 위조 흔적 분석</h2>
      </header>

      {/* 왼쪽 히트맵 | 오른쪽 시각화 설정 (이미지 위조 흔적과 같은 구성) */}
      <div className="dg-ihm">
        <section className={`dg-ihm__stage dg-ihm__stage--nav${isProcessing ? ' is-active' : ''}`} aria-label="히트맵 결과">
          {/* 이전/다음 구간: 히트맵 왼쪽·오른쪽 가운데 화살표, 아래에 'N초' */}
          {frames.length > 1 && (
            <>
              <button type="button" className="dg-ihm__nav dg-ihm__nav--prev" onClick={() => goFrame(prevFrame)}
                disabled={!prevFrame} aria-label={prevFrame ? `이전 구간 ${secLabel(prevFrame)}` : '이전 구간 없음'}>
                <span className="dg-ihm__nav-arrow" aria-hidden="true">
                  <svg viewBox="0 0 24 24"><path d="M15 5l-7 7 7 7" /></svg>
                </span>
                <span className="dg-ihm__nav-time">{prevFrame ? secLabel(prevFrame) : '처음'}</span>
              </button>
              <button type="button" className="dg-ihm__nav dg-ihm__nav--next" onClick={() => goFrame(nextFrame)}
                disabled={!nextFrame} aria-label={nextFrame ? `다음 구간 ${secLabel(nextFrame)}` : '다음 구간 없음'}>
                <span className="dg-ihm__nav-arrow" aria-hidden="true">
                  <svg viewBox="0 0 24 24"><path d="M9 5l7 7-7 7" /></svg>
                </span>
                <span className="dg-ihm__nav-time">{nextFrame ? secLabel(nextFrame) : '마지막'}</span>
              </button>
            </>
          )}
          <div className="dg-ihm__canvas">
            {/* 왼쪽 위: 현재 구간 시각 */}
            <span className="dg-ihm__frame-chip">
              {secLabel(curFrame) || `#${frame_index}`}
              {frames.length > 1 && curPos >= 0 && <em>{curPos + 1} / {frames.length}</em>}
            </span>
            {isDone && (
              <HeatmapZoomImage src={heatmapSrc} alt="위조 흔적 히트맵"
                zoomable={applied.display_type !== 'heatmap'}
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

            {/* 왼쪽 아래 작은 범례 · 오른쪽 위 저장 아이콘 (그림을 가리지 않게) */}
            {isDone && applied.display_type !== 'bbox' && (
              <div className="dg-ihm__legend" aria-label="히트맵 색 범례: 파랑 낮음, 빨강 높음">
                <span>낮음</span><i aria-hidden="true" /><span>높음</span>
              </div>
            )}
            {isDone && (
              <a className="dg-ihm__download" href={heatmapSrc}
                download={`heatmap_f${frame_index}_${applied.branch}_${modelType}_${applied.display_type}.png`} aria-label="이미지 저장" title="이미지 저장">
                <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 4v11M7 10l5 5 5-5M5 20h14" /></svg>
              </a>
            )}
          </div>
        </section>

        <aside className="dg-ihm__side">
          <dl className="dg-ihm__summary">
            {stats.map(item => (
              <div key={item.label} className="dg-ihm__summary-row">
                <dt>{item.label}</dt>
                <dd className={item.tone ? `dg-heatmap__tone--${item.tone}` : ''}>{item.value}</dd>
              </div>
            ))}
          </dl>
          <HeatmapVisualSettings branches={branches} branch={branch} setBranch={setBranch}
            draft={draft} setDraft={setDraft} dirtyCount={dirtyCount} isProcessing={isProcessing}
            onApply={handleApply} onReset={handleReset} canReset={canReset} />
        </aside>
      </div>
    </HeatmapLayout>
  );
};

export default HeatmapPage;

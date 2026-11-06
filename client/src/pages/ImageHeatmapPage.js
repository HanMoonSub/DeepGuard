import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';

import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import Button from '../components/Button';
import Card from '../components/Card';
import './HeatmapPage.css';

const POLL_INTERVAL = 2000;

// branch별 XAI 기법 (low: 국소 흔적 / high: 전역 구조)
const EXPLAINER_BY_BRANCH = { low: 'layercam', high: 'eigengradcam' };
const BRANCH_HINT = { low: '국소 흔적', high: '전역 구조' };
const BRANCH_LABEL = { low: '국소 부위 표시', high: '전역 부위 표시' };
const DISPLAY_OPTIONS = [
  { value: 'heatmap',      label: 'Heatmap' },
  { value: 'bbox',         label: 'BBox' },
  { value: 'heatmap_bbox', label: 'Heatmap + BBox' },
];
const DISPLAY_EYEBROW = { heatmap: 'HEATMAP', bbox: 'BBOX', heatmap_bbox: 'HEATMAP + BBOX' };
const DEFAULT_DISPLAY = { display_type: 'heatmap_bbox', overlay_ratio: 0.7, threshold: 0.8 };

const extractTaskId = (data) => {
  if (typeof data === 'string') return data;
  return data?.task_id || data?.id || null;
};
const toAbsoluteUrl = (path) => {
  if (!path) return null;
  if (path.startsWith('http') || path.startsWith('blob')) return path;
  return path;
};
// 0.05 단위 슬라이더 부동소수 오차 정리
const round2 = (v) => Math.round(Number(v) * 100) / 100;

// 공통 레이아웃: Navbar(light) + 본문 + Footer
const HeatmapLayout = ({ sessionUser, children }) => (
  <div className="dg-heatmap">
    <Navbar variant="light" sessionUser={sessionUser} />
    <main className="dg-heatmap__main">
      <div className="dg-heatmap__wrap">{children}</div>
    </main>
    <Footer />
  </div>
);

// Display 설정 패널: 값은 draft에만 반영하고 '적용' 시 재요청
const rangeFill = (v, min, max) => ({ '--fill': `${((v - min) / (max - min)) * 100}%` });

const DisplaySettings = ({ draft, setDraft, isDirty, isProcessing, onApply }) => {
  const isBBox = draft.display_type === 'bbox';
  const update = (key, value) => setDraft(prev => ({ ...prev, [key]: value }));
  const reset = (key) => update(key, DEFAULT_DISPLAY[key]);
  const isDefault = (key) => draft[key] === DEFAULT_DISPLAY[key];

  return (
    <Card className="dg-card--static dg-heatmap__settings">
      <details className="dg-heatmap__settings-fold">
        <summary className="dg-heatmap__settings-summary">
          <span className="dg-heatmap__eyebrow">위조 흔적 표시 설정</span>
          <span className="dg-heatmap__settings-meta">
            <span>{DISPLAY_OPTIONS.find(opt => opt.value === draft.display_type)?.label}</span>
            {!isBBox && <span>투명도 {draft.overlay_ratio.toFixed(2)}</span>}
            <span>임계값 {draft.threshold.toFixed(2)}</span>
            {isDirty && <span className="is-dirty">미적용 변경</span>}
          </span>
          <i className="dg-heatmap__settings-chevron" aria-hidden="true">▾</i>
        </summary>

        <div className="dg-heatmap__settings-body">
          {/* ① Display Mode → display_type */}
          <section className="dg-heatmap__settings-box dg-heatmap__settings-box--mode">
            <div className="dg-heatmap__mode">
              <span className="dg-heatmap__setting-label" id="hm-display-label">위조 흔적 표시</span>
              <div className="dg-heatmap__radios" role="radiogroup" aria-labelledby="hm-display-label">
                {DISPLAY_OPTIONS.map(opt => (
                  <label key={opt.value} className={`dg-heatmap__radio${draft.display_type === opt.value ? ' is-active' : ''}`}>
                    <input type="radio" name="hm-display" value={opt.value}
                      checked={draft.display_type === opt.value}
                      onChange={() => update('display_type', opt.value)} />
                    <i className="dg-heatmap__radio-dot" aria-hidden="true" />
                    {opt.label}
                  </label>
                ))}
              </div>
            </div>
            <Button variant="primary" className="dg-heatmap__apply" onClick={onApply} disabled={isProcessing || !isDirty}>
              {isProcessing ? '적용 중...' : '적용'}
            </Button>
          </section>

          {/* ② Image Weight | Threshold */}
          <section className="dg-heatmap__settings-box dg-heatmap__settings-box--sliders">
            {/* Image Weight → overlay_ratio */}
            <div className={`dg-heatmap__slider${isBBox ? ' is-disabled' : ''}`}>
              <div className="dg-heatmap__slider-head">
                <div className="dg-heatmap__slider-text">
                  <div className="dg-heatmap__slider-title">
                    <label className="dg-heatmap__setting-label" htmlFor="hm-overlay">원본 이미지 투명도</label>
                    {isBBox && <span className="dg-heatmap__setting-badge">BBox 모드에서는 적용되지 않음</span>}
                  </div>
                  <p className="dg-heatmap__setting-desc">값이 작을수록 원본이 옅어지고 위조 흔적이 진해집니다</p>
                </div>
                <div className="dg-heatmap__value-box">
                  <span>{draft.overlay_ratio.toFixed(2)}</span>
                  <button type="button" className="dg-heatmap__reset" title="기본값으로 초기화"
                    aria-label="원본 이미지 투명도 초기화" disabled={isBBox || isDefault('overlay_ratio')}
                    onClick={() => reset('overlay_ratio')}>↺</button>
                </div>
              </div>
              <div className="dg-heatmap__range-wrap">
                <span className="dg-heatmap__range-tick" aria-hidden="true">0.5</span>
                <input id="hm-overlay" type="range" className="dg-heatmap__range"
                  min="0.5" max="0.9" step="0.05" value={draft.overlay_ratio} disabled={isBBox}
                  style={rangeFill(draft.overlay_ratio, 0.5, 0.9)}
                  onChange={(e) => update('overlay_ratio', round2(e.target.value))} />
                <span className="dg-heatmap__range-tick" aria-hidden="true">0.9</span>
              </div>
            </div>

            {/* Threshold */}
            <div className="dg-heatmap__slider">
              <div className="dg-heatmap__slider-head">
                <div className="dg-heatmap__slider-text">
                  <div className="dg-heatmap__slider-title">
                    <label className="dg-heatmap__setting-label" htmlFor="hm-threshold">위조 흔적 임계값</label>
                  </div>
                  <p className="dg-heatmap__setting-desc">
                    상위 몇 %까지 위조 의심 영역으로 표시할지 정합니다 — 현재 상위 {Math.round((1 - draft.threshold) * 100)}%
                  </p>
                </div>
                <div className="dg-heatmap__value-box">
                  <span>{draft.threshold.toFixed(2)}</span>
                  <button type="button" className="dg-heatmap__reset" title="기본값으로 초기화"
                    aria-label="위조 흔적 임계값 초기화" disabled={isDefault('threshold')}
                    onClick={() => reset('threshold')}>↺</button>
                </div>
              </div>
              <div className="dg-heatmap__range-wrap">
                <span className="dg-heatmap__range-tick" aria-hidden="true">0.7</span>
                <input id="hm-threshold" type="range" className="dg-heatmap__range"
                  min="0.7" max="0.95" step="0.05" value={draft.threshold}
                  style={rangeFill(draft.threshold, 0.7, 0.95)}
                  onChange={(e) => update('threshold', round2(e.target.value))} />
                <span className="dg-heatmap__range-tick" aria-hidden="true">0.95</span>
              </div>
            </div>
          </section>
        </div>
      </details>
    </Card>
  );
};

const ImageHeatmapPage = ({ sessionUser }) => {
  const { state } = useLocation();
  const navigate  = useNavigate();
  const { image_id, image_loc, model_type, prob, label } = state || {};
  const modelType = model_type || 'fast'; // 분석에 사용한 모델 그대로 사용

  const [branch, setBranch]           = useState('high');
  const [draft, setDraft]             = useState(DEFAULT_DISPLAY);
  const [applied, setApplied]         = useState({ branch: 'high', ...DEFAULT_DISPLAY });
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

  // 진입 즉시 생성 (high branch + 기본 Display 설정)
  useEffect(() => {
    if (!state || startedRef.current) return;
    startedRef.current = true;
    requestHeatmap('high', DEFAULT_DISPLAY);
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

  const handleBranchSelect = (b) => {
    setBranch(b);
    requestHeatmap(b, draft);
  };
  const handleApply = () => requestHeatmap(branch, draft);
  const handleRetry = () => requestHeatmap(applied.branch, applied);

  const isFake = (prob ?? 0) > 0.5;
  const isProcessing = status === 'submitting' || status === 'polling';
  const imageSrc = image_loc ? toAbsoluteUrl(image_loc) : null;
  const tone = isFake ? 'fake' : 'real';
  const verdict = label || (isFake ? 'FAKE' : 'REAL');
  const isDirty = branch !== applied.branch
    || draft.display_type !== applied.display_type
    || draft.overlay_ratio !== applied.overlay_ratio
    || draft.threshold !== applied.threshold;

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
    { label: '딥페이크 확률', value: `${((prob ?? 0) * 100).toFixed(1)}%`, tone, big: true },
    { label: '판정 결과',     value: verdict,                               tone, big: true },
    { label: '모델',          value: modelType.toUpperCase() },
  ];

  return (
    <HeatmapLayout sessionUser={sessionUser}>
      {/* 상단: 뒤로가기 + 제목 + 판정/모델 태그 */}
      <header className="dg-heatmap__head">
        <button type="button" className="dg-heatmap__back" onClick={() => navigate(-1)}>
          <span aria-hidden="true">←</span> 분석 결과로
        </button>
        <h2 className="dg-heatmap__title">이미지 위조 흔적 분석</h2>
        <div className="dg-heatmap__chips">
          <span className={`dg-heatmap__chip--${tone}`}>{verdict}</span>
          <span>{modelType.toUpperCase()}</span>
        </div>
      </header>

      {/* 스탯 */}
      <Card className="dg-card--static dg-heatmap__stats dg-heatmap__stats--3">
        {stats.map(item => (
          <dl key={item.label} className={`dg-heatmap__stat${item.big ? ' dg-heatmap__stat--big' : ''}`}>
            <dt>{item.label}</dt>
            <dd className={item.tone ? `dg-heatmap__tone--${item.tone}` : ''}>{item.value}</dd>
          </dl>
        ))}
      </Card>

      {/* Display 설정 */}
      <DisplaySettings draft={draft} setDraft={setDraft} isDirty={isDirty}
        isProcessing={isProcessing} onApply={handleApply} />

      {/* 메인 2열: 원본 | 히트맵 */}
      <div className={`dg-heatmap__grid${imageSrc ? '' : ' dg-heatmap__grid--single'}`}>

        {/* 원본 이미지 */}
        {imageSrc && (
          <Card className="dg-card--static dg-heatmap__panel">
            <div className="dg-heatmap__panel-head">
              <div className="dg-heatmap__panel-title">
                <i className="dg-heatmap__dot is-done" />
                <span className="dg-heatmap__eyebrow">원본 이미지</span>
              </div>
            </div>
            <div className="dg-heatmap__media">
              <img src={imageSrc} alt="Original"
                onError={(e) => { e.target.style.display = 'none'; }}
              />
            </div>
          </Card>
        )}

        {/* 히트맵 패널 */}
        <Card className={`dg-card--static dg-heatmap__panel${isProcessing ? ' is-active' : ''}`}>
          <div className="dg-heatmap__panel-head">
            <div className="dg-heatmap__panel-title">
              <i className={`dg-heatmap__dot${isProcessing ? ' is-active' : status === 'done' ? ' is-done' : ''}`} />
              <span className="dg-heatmap__eyebrow">{DISPLAY_EYEBROW[applied.display_type]}</span>
            </div>
            <div className="dg-heatmap__panel-tools">
              {status === 'done' && heatmapSrc && (
                <Button variant="outline" className="dg-heatmap__save" href={heatmapSrc}
                  download={`heatmap_img${image_id}_${applied.branch}_${modelType}_${applied.display_type}.png`}>
                  ↓ 저장
                </Button>
              )}
            </div>
          </div>

          {/* 결과 */}
          {status === 'done' && heatmapSrc ? (
            <div className="dg-heatmap__result">
              <div className="dg-heatmap__media">
                <img src={heatmapSrc} alt="Heatmap"
                  onError={(e) => { e.target.style.display = 'none'; setErrorMsg('이미지 로드 실패'); setErrorDetail(`URL: ${heatmapSrc}`); setStatus('error'); }}
                />
              </div>
            </div>
          ) : (
            <div className="dg-heatmap__media dg-heatmap__media--stage">
              {/* 에러 */}
              {status === 'error' && (
                <div className="dg-heatmap__options">
                  <div className="dg-heatmap__alert">
                    <div className="dg-heatmap__alert-icon" aria-hidden="true">!</div>
                    <p className="dg-heatmap__alert-msg">{errorMsg}</p>
                    {errorDetail && <p className="dg-heatmap__alert-detail">{errorDetail}</p>}
                  </div>
                  <Button variant="primary" onClick={handleRetry}>다시 시도</Button>
                </div>
              )}

              {/* 로딩 */}
              {(isProcessing || status === 'idle') && (
                <div className="dg-heatmap__loading">
                  <div className="dg-heatmap__spinner" />
                </div>
              )}

              {status === 'done' && !heatmapSrc && (
                <div className="dg-heatmap__empty">
                  <p>결과 이미지를 받지 못했습니다.</p>
                  <Button variant="outline" onClick={handleRetry}>재시도</Button>
                </div>
              )}
            </div>
          )}

          {/* Branch 전환 */}
          <div className="dg-heatmap__combos">
            {Object.keys(EXPLAINER_BY_BRANCH).map(b => (
              <button key={b} type="button" title={BRANCH_HINT[b]} disabled={isProcessing}
                onClick={() => handleBranchSelect(b)}
                className={`dg-heatmap__combo${branch === b ? ' is-active' : ''}`}>
                {BRANCH_LABEL[b]}
              </button>
            ))}
          </div>
        </Card>
      </div>

    </HeatmapLayout>
  );
};

export default ImageHeatmapPage;

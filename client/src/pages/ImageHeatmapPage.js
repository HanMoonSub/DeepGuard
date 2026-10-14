import React, { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';

import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import Button from '../components/Button';
import Card from '../components/Card';
import './HeatmapPage.css';

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
  return path;
};

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
  const [, setTaskId]                 = useState(null);
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
  const tone = isFake ? 'fake' : 'real';
  const verdict = label || (isFake ? 'FAKE' : 'REAL');

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
    { label: 'FAKE 확률', value: `${((prob ?? 0) * 100).toFixed(1)}%`, tone, big: true },
    { label: 'VERDICT',   value: verdict,                               tone, big: true },
    { label: 'MODEL',     value: (selectedModel || model_type || 'fast').toUpperCase() },
  ];

  return (
    <HeatmapLayout sessionUser={sessionUser}>
      {/* 상단: 뒤로가기 + 판정/모델 태그 */}
      <header className="dg-heatmap__head">
        <button type="button" className="dg-heatmap__back" onClick={() => navigate(-1)}>
          <span aria-hidden="true">←</span> 분석 결과로
        </button>
        <div className="dg-heatmap__chips">
          <span className={`dg-heatmap__chip--${tone}`}>{verdict}</span>
          {model_type && <span>{model_type.toUpperCase()}</span>}
        </div>
      </header>

      <h2 className="dg-heatmap__title">이미지 위조 흔적 분석</h2>
      <p className="dg-heatmap__sub">원본과 히트맵을 나란히 비교하며 모델이 주목한 위조 의심 영역을 확인하세요.</p>

      {/* 스탯 */}
      <Card className="dg-card--static dg-heatmap__stats dg-heatmap__stats--3">
        {stats.map(item => (
          <dl key={item.label} className={`dg-heatmap__stat${item.big ? ' dg-heatmap__stat--big' : ''}`}>
            <dt>{item.label}</dt>
            <dd className={item.tone ? `dg-heatmap__tone--${item.tone}` : ''}>{item.value}</dd>
          </dl>
        ))}
      </Card>

      {/* 메인 2열: 원본 | 히트맵 */}
      <div className={`dg-heatmap__grid${imageSrc ? '' : ' dg-heatmap__grid--single'}`}>

        {/* 원본 이미지 */}
        {imageSrc && (
          <Card className="dg-card--static dg-heatmap__panel">
            <div className="dg-heatmap__panel-head">
              <div className="dg-heatmap__panel-title">
                <i className="dg-heatmap__dot is-done" />
                <span className="dg-heatmap__eyebrow">ORIGINAL IMAGE</span>
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
          {isProcessing && <div className="dg-heatmap__progress" />}

          <div className="dg-heatmap__panel-head">
            <div className="dg-heatmap__panel-title">
              <i className={`dg-heatmap__dot${isProcessing ? ' is-active' : status === 'done' ? ' is-done' : ''}`} />
              <span className="dg-heatmap__eyebrow">HEATMAP + BBOX OVERLAY</span>
              {selectedBranch && <span className="dg-heatmap__panel-tag">{selectedBranch.toUpperCase()} / {(selectedModel || '').toUpperCase()}</span>}
            </div>
            <div className="dg-heatmap__panel-tools">
              {isProcessing && <span className="dg-heatmap__elapsed">{elapsed}s</span>}
              {status === 'done' && heatmapSrc && (
                <Button variant="outline" className="dg-heatmap__save" href={heatmapSrc}
                  download={`heatmap_img${image_id}_${selectedBranch}_${selectedModel}.png`}>
                  ↓ 저장
                </Button>
              )}
            </div>
          </div>

          {/* 결과 */}
          {status === 'done' && heatmapSrc ? (
            <div className="dg-heatmap__result">
              <div className="dg-heatmap__media">
                <span className="dg-heatmap__badge">✓ {(selectedBranch || '').toUpperCase()} / {(selectedModel || '').toUpperCase()}</span>
                <img src={heatmapSrc} alt="Heatmap"
                  onError={(e) => { e.target.style.display = 'none'; setErrorMsg('이미지 로드 실패'); setErrorDetail(`URL: ${heatmapSrc}`); setStatus('error'); }}
                />
              </div>
              {/* 재선택 */}
              <div className="dg-heatmap__combos">
                {Object.keys(BRANCH_OPTIONS).flatMap(branch => MODEL_OPTIONS.map(m => ({ branch, m }))).map(({ branch, m }) => (
                  <button key={branch + m} type="button" onClick={() => handleBranchSelect(branch, m)}
                    className={`dg-heatmap__combo${(selectedBranch === branch && selectedModel === m) ? ' is-active' : ''}`}>
                    {branch.toUpperCase()} / {m.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="dg-heatmap__media dg-heatmap__media--stage">

              {/* 옵션 선택 */}
              {(status === 'idle' || status === 'error') && !isProcessing && (
                <div className="dg-heatmap__options">
                  {status === 'error' && (
                    <div className="dg-heatmap__alert">
                      <div className="dg-heatmap__alert-icon" aria-hidden="true">!</div>
                      <p className="dg-heatmap__alert-msg">{errorMsg}</p>
                      {errorDetail && <p className="dg-heatmap__alert-detail">{errorDetail}</p>}
                    </div>
                  )}

                  {/* Branch 선택 */}
                  <p className="dg-heatmap__eyebrow">BRANCH LEVEL</p>
                  <div className="dg-heatmap__choices">
                    {Object.keys(BRANCH_OPTIONS).map(branch => (
                      <button key={branch} type="button" onClick={() => setTempBranch(branch)}
                        className={`dg-heatmap__choice${tempBranch === branch ? ' is-active' : ''}`}>
                        {branch}
                        <small>{branch === 'low' ? '국소 흔적' : '전역 구조'}</small>
                      </button>
                    ))}
                  </div>

                  {/* Model 선택 */}
                  <p className="dg-heatmap__eyebrow">MODEL TYPE</p>
                  <div className="dg-heatmap__choices">
                    {MODEL_OPTIONS.map(m => (
                      <button key={m} type="button" onClick={() => setTempModel(m)}
                        className={`dg-heatmap__choice${tempModel === m ? ' is-active' : ''}`}>
                        {m}
                      </button>
                    ))}
                  </div>

                  {/* 실행 버튼 */}
                  <Button variant="primary" className="dg-btn--block dg-heatmap__generate"
                    onClick={() => handleBranchSelect(tempBranch, tempModel)}>
                    히트맵 생성하기
                  </Button>
                </div>
              )}

              {/* 로딩 */}
              {isProcessing && (
                <div className="dg-heatmap__loading">
                  <div className="dg-heatmap__spinner" />
                  <p className="dg-heatmap__loading-title">GENERATING HEATMAP</p>
                  <p className="dg-heatmap__loading-desc">위조 흔적 시각화 처리 중...</p>
                  <p className="dg-heatmap__loading-meta">{(selectedBranch || tempBranch || '').toUpperCase()} BRANCH · {(selectedModel || tempModel || '').toUpperCase()} MODEL</p>
                  <p className="dg-heatmap__loading-meta">{elapsed}s elapsed</p>
                </div>
              )}

              {status === 'done' && !heatmapSrc && (
                <div className="dg-heatmap__empty">
                  <p>결과 이미지를 받지 못했습니다.</p>
                  <Button variant="outline" onClick={() => requestHeatmap(selectedBranch, selectedModel)}>재시도</Button>
                </div>
              )}
            </div>
          )}
        </Card>
      </div>
    </HeatmapLayout>
  );
};

export default ImageHeatmapPage;

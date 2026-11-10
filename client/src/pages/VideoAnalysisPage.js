import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

import Navbar from '../components/Navbar';
import useTheme from '../hooks/useTheme';
import Footer from '../components/Footer';
import Button from '../components/Button';
// 레이아웃(사이드바 | 중앙 | 설정)은 이미지 분석 페이지와 동일 → .dg-analysis 스타일 재사용
import './analysispage.css';
import WebcamCapture, { MediaSourceBar } from '../components/WebcamCapture';
import AnalysisWarning from '../components/AnalysisWarning';
import './VideoAnalysisPage.css';

// 예시 비디오 (public/samples/videos/ 하위에 실제 파일 배치 필요 / 파일명 다르면 이 배열만 수정)
const SAMPLE_VIDEOS = [
  { name: 'western_fake.mp4', src: '/samples/videos/western_fake.mp4', domain: 'western', label: '서양인 Fake' },
  { name: 'western_real.mp4', src: '/samples/videos/western_real.mp4', domain: 'western', label: '서양인 Real' },
];

// FAKE/REAL/UNKNOWN → 배지 modifier
const labelTone = (label) => (label === 'FAKE' ? 'fake' : label === 'REAL' ? 'real' : 'unknown');

// 초 → mm:ss
const formatElapsed = (sec) => `${String(Math.floor(sec / 60)).padStart(2, '0')}:${String(sec % 60).padStart(2, '0')}`;

const SidebarIcon = ({ open }) => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <rect x="3" y="4" width="18" height="16" rx="2" /><line x1="9" y1="4" x2="9" y2="20" />
    {open ? <polyline points="12 9 15 12 12 15" /> : <polyline points="15 9 12 12 15 15" />}
  </svg>
);

// 개발 서버 전용 미리보기 — 백엔드 없이 이펙트 확인용
// ?preview=analyzing (분석 중) | fake | real (결과 + 퍼지는 링). 빌드(production)에서는 무시됨
const DEV_PREVIEW = process.env.NODE_ENV === 'development'
  ? new URLSearchParams(window.location.search).get('preview')
  : null;
const PREVIEW_RESULT = ['fake', 'real'].includes(DEV_PREVIEW)
  ? { status: 'SUCCESS', label: DEV_PREVIEW.toUpperCase(), video_id: 'preview' }
  : null;

const VideoAnalysisPage = ({ sessionUser, onLogout, setSessionUser }) => {
  const navigate = useNavigate();
  const [theme, setTheme] = useTheme();

  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(DEV_PREVIEW === 'analyzing');
  const [history, setHistory] = useState([]);

  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [selectedSample, setSelectedSample] = useState(null);

  const [modelType, setModelType] = useState('fast');
  const [domainType, setDomainType] = useState('western');

  const [result, setResult] = useState(PREVIEW_RESULT);
  const [statusMessage, setStatusMessage] = useState('');

  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  // 진행 표시 전용: 분석 경과 시간(초). 폴링/API와 무관
  const [elapsed, setElapsed] = useState(0);

  // 방금 분석이 끝난 video_id — 완료 이펙트(결과 팝 + 새 기록 하이라이트)용
  const [freshId, setFreshId] = useState(PREVIEW_RESULT ? 'preview' : null);

  const pollingTimer = useRef(null);
  const freshTimer = useRef(null);
  const isMounted = useRef(true);

  useEffect(() => () => clearTimeout(freshTimer.current), []);

  const markFresh = useCallback((videoId) => {
    clearTimeout(freshTimer.current);
    setFreshId(String(videoId));
    freshTimer.current = setTimeout(() => setFreshId(null), 3000);
  }, []);

  const fetchHistory = useCallback(async () => {
    if (!sessionUser) return;
    try {
      const response = await axios.get('/video/history');
      if (response.data.status === "success") {
        setHistory(response.data.context || []);
      }
    } catch (e) { console.log("비디오 히스토리 로드 실패"); }
  }, [sessionUser]);

  useEffect(() => {
    isMounted.current = true;
    const syncSession = async () => {
      if (!sessionUser) {
        try {
          const response = await axios.get('/auth/check');
          if (response.data && response.data.user) setSessionUser(response.data.user);
        } catch (error) { console.log("세션 확인 실패"); }
      }
    };
    syncSession();
    fetchHistory();

    return () => {
      isMounted.current = false;
      if (pollingTimer.current) {
        clearInterval(pollingTimer.current);
        pollingTimer.current = null;
      }
    };
  }, [sessionUser, setSessionUser, fetchHistory]);

  // 분석 중일 때만 경과 시간 카운트 (표시용)
  useEffect(() => {
    if (!isAnalyzing) return;
    setElapsed(0);
    const timer = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(timer);
  }, [isAnalyzing]);

  const startPolling = useCallback((videoId) => {
    if (!videoId) return;
    if (pollingTimer.current) clearInterval(pollingTimer.current);

    pollingTimer.current = setInterval(async () => {
      if (!isMounted.current) return;
      try {
        const response = await axios.get(`/inference/video/${videoId}`);
        const data = response.data;
        if (data.status === 'SUCCESS' || data.prob !== undefined || data.analysis !== undefined) {
          clearInterval(pollingTimer.current);
          pollingTimer.current = null;
          setResult({ ...data, video_id: videoId });
          setIsAnalyzing(false);
          setStatusMessage('');
          markFresh(videoId);
          fetchHistory();
        } else if (data.status === 'FAILED' || data.status === 'ERROR') {
          clearInterval(pollingTimer.current);
          pollingTimer.current = null;
          setIsAnalyzing(false);
          setStatusMessage('');
          alert(data.result_msg || "비디오 분석 실패");
        } else if (data.status === 'WARNING') {
          clearInterval(pollingTimer.current);
          pollingTimer.current = null;
          setResult({ ...data, status: 'WARNING' });
          setIsAnalyzing(false);
          setStatusMessage('');
          fetchHistory();
        } else {
          setStatusMessage(data.message || "비디오 분석 진행 중...");
        }
      } catch (err) { setStatusMessage("결과 확인 중..."); }
    }, 2000);
  }, [fetchHistory, markFresh]);

  const handlePredict = async () => {
    if (modelType === 'pro' && !sessionUser) {
      alert("PRO 모델 분석은 로그인이 필요합니다.");
      navigate('/login');
      return;
    }
    if (!file) return alert("동영상을 먼저 업로드해주세요.");

    const formData = new FormData();
    formData.append('videofile', file);
    formData.append('model_type', modelType);
    formData.append('domain_type', domainType === 'western' ? '서양인' : '동양인');
    formData.append('version_type', 'v1'); // 판정은 v1(MS-EffGCViT)만 사용

    setIsAnalyzing(true);
    setStatusMessage('서버 접수 중...');
    setResult(null);

    try {
      const response = await axios.post('/inference/video', formData);
      const videoId = response.data?.video_id || response.data;
      if (videoId) startPolling(videoId);
      else setIsAnalyzing(false);
    } catch (err) {
      setIsAnalyzing(false);
      setStatusMessage('');
    }
  };

  const handleFileSelect = (selectedFile) => {
    if (!selectedFile) return;
    setFile(selectedFile);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setResult(null);
    setStatusMessage('');
    setIsCameraOpen(false);
    setSelectedSample(null); // 샘플 외 경로로 들어오면 선택 해제
  };

  // 예시 비디오를 File로 변환 후 기존 업로드 흐름에 주입 + 도메인 자동 세팅
  const handleSelectSample = async (sample) => {
    try {
      const res = await fetch(sample.src);
      const blob = await res.blob();
      const file = new File([blob], sample.name, { type: blob.type });
      setDomainType(sample.domain);
      handleFileSelect(file);
      setSelectedSample(sample.name); // handleFileSelect 리셋 이후 선택 표시
    } catch (e) {
      alert("예시 파일을 불러오지 못했어요.");
    }
  };

  // 기록 한 건 삭제 — 호버 시 노출되는 휴지통 버튼
  const handleDeleteOne = async (e, id) => {
    e.stopPropagation();
    if (!window.confirm("이 기록을 삭제하시겠습니까?")) return;
    try {
      await axios.delete(`/video/history/${id}`);
      fetchHistory();
    } catch (error) {
      console.error(error);
      alert("서버에서 기록을 삭제하는 중 오류가 발생했습니다.");
    }
  };

  const handleNewAnalysis = () => {
    setFile(null); setPreviewUrl(null); setResult(null); setStatusMessage(''); setIsCameraOpen(false); setSelectedSample(null);
    if (pollingTimer.current) clearInterval(pollingTimer.current);
  };

  // 진행 단계: 기존 state에서만 계산 (접수 → 프레임 분석 → 판정)
  const isReceiving = statusMessage === '서버 접수 중...';
  const steps = [
    { label: '업로드·접수', state: isReceiving ? 'active' : 'done' },
    { label: '프레임 분석', state: isReceiving ? 'wait' : 'active' },
    { label: '판정', state: 'wait' },
  ];

  const renderResult = () => {
    if (isAnalyzing) {
      return (
        <div className="dg-analysis__scanning">
          {/* 탐지 펄스: 중앙 코어에서 원이 계속 퍼져나감 */}
          <div className="dg-analysis__pulse" aria-hidden="true">
            <span /><span /><span />
            <div className="dg-analysis__pulse-core">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3l7 3v5c0 4.5-3 8.3-7 10-4-1.7-7-5.5-7-10V6l7-3z" /><path d="M9 12l2 2 4-4" /></svg>
            </div>
          </div>
          <h3 className="dg-analysis__scan-title">AI가 영상을 분석하고 있어요</h3>

          <ol className="dg-video__steps">
            {steps.map((step, i) => (
              <li key={step.label} className={`dg-video__step is-${step.state}`}>
                <span className="dg-video__step-dot">{step.state === 'done' ? '✓' : i + 1}</span>
                <span className="dg-video__step-label">{step.label}</span>
              </li>
            ))}
          </ol>

          <div className="dg-analysis__progress">
            <div className="dg-analysis__progress-head">
              <span>{statusMessage || "분석 준비 중..."}</span>
              <span className="dg-analysis__live">LIVE</span>
            </div>
            <div className="dg-analysis__bar"><div /></div>
          </div>

          <p className="dg-video__elapsed">
            경과 시간 <strong>{formatElapsed(elapsed)}</strong>
          </p>
          <p className="dg-video__hint">
            영상 길이에 따라 수십 초~수 분이 걸릴 수 있어요.
            {sessionUser && <><br />페이지를 떠나도 분석은 계속되고, 결과는 ‘내 영상 기록’에서 확인할 수 있어요.</>}
          </p>
        </div>
      );
    }

    if (!result) {
      return (
        <div className="dg-analysis__placeholder">
          <svg className="dg-analysis__placeholder-ring" width="132" height="132" viewBox="0 0 132 132" aria-hidden="true">
            <circle className="dg-analysis__ring-track" cx="66" cy="66" r="54" fill="none" strokeWidth="12" />
            <circle className="dg-analysis__ring-arc" cx="66" cy="66" r="54" fill="none" strokeWidth="12" strokeLinecap="round" strokeDasharray="90 340" transform="rotate(-90 66 66)" />
          </svg>
          <div className="dg-analysis__placeholder-text">
            <p className="dg-analysis__placeholder-title">분석 결과가 여기에 표시됩니다</p>
          </div>
          <div className="dg-analysis__placeholder-lines" aria-hidden="true">
            <span />
            <span />
          </div>
        </div>
      );
    }

    if (result.status === 'WARNING') {
      return (
        <AnalysisWarning mode="video" onRetry={handleNewAnalysis} />
      );
    }

    const label = result.label || (result.score > 0.5 ? 'FAKE' : 'REAL');
    return (
      <div className={`dg-analysis__result${freshId === String(result.video_id) ? ' is-fresh' : ''}`}>
        <p className="dg-analysis__eyebrow">분석 결과</p>
        <p className={`dg-analysis__verdict dg-analysis__verdict--${labelTone(label)}`}>{label}</p>
        <Button
          variant="outline"
          className="dg-analysis__detail-btn"
          onClick={() => navigate('/video-detail', { state: { video_id: result.video_id } })}
        >
          상세 결과 보기 →
        </Button>
      </div>
    );
  };

  return (
    <div className="dg-analysis dg-video">
      <Navbar variant="light" sessionUser={sessionUser} onLogout={onLogout} logoutRedirect="/video-analysis" theme={theme} onThemeChange={setTheme} />

      <div className="dg-analysis__workspace">
        <aside className={`dg-analysis__side${isSidebarOpen ? '' : ' is-collapsed'}`}>
          <div className="dg-analysis__side-inner">
            <Button variant="primary" className="dg-btn--block" onClick={handleNewAnalysis}>+ 새 영상 분석</Button>

            <div className="dg-analysis__side-head">
              <h3>내 영상 기록</h3>
              <div className="dg-analysis__side-tools">
                <button type="button" className="dg-analysis__icon-btn" onClick={() => setIsSidebarOpen(false)} aria-label="사이드바 접기">
                  <SidebarIcon open={false} />
                </button>
              </div>
            </div>

            <div className="dg-analysis__history">
              {sessionUser ? (
                history.map((item, index) => {
                  const currentId = item.video_id || item.id;
                  const dType = item.domain_type || '서양인';
                  const mType = item.model_type ? item.model_type.toUpperCase() : 'FAST';

                  return (
                    <div key={currentId || index} className="dg-analysis__history-row">
                      <div
                        onClick={() => navigate('/video-detail', {
                          state: { video_id: currentId }
                        })}
                        className={`dg-analysis__history-item${labelTone(item.label) === 'real' ? ' is-real' : ''}${labelTone(item.label) === 'fake' ? ' is-fake' : ''}${freshId === String(currentId) ? ' is-new' : ''}`}
                      >
                        <div className="dg-analysis__thumb">
                          {item.video_loc ? (
                            <video src={`${item.video_loc}#t=0.1`} muted preload="metadata" />
                          ) : '📹'}
                        </div>
                        <div className="dg-analysis__history-meta">
                          <div className="dg-analysis__history-type">
                            <span className="dg-analysis__tag">{dType}</span>
                            <span className="dg-analysis__tag">{mType}</span>
                          </div>
                          <div className="dg-analysis__history-date">
                            {item.created_at?.slice(0, 10)}
                            <span>{item.created_at?.slice(11, 16)}</span>
                          </div>
                        </div>
                        <span className={`dg-analysis__badge dg-analysis__badge--${labelTone(item.label)}`}>{labelTone(item.label) === 'unknown' ? '판정 불가' : item.label}</span>
                      </div>
                      <button type="button" className="dg-analysis__trash" onClick={(e) => handleDeleteOne(e, currentId)} aria-label="기록 삭제" title="삭제">
                        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6M10 11v6M14 11v6" /></svg>
                      </button>
                    </div>
                  );
                })
              ) : (
                <div className="dg-analysis__locked">
                  <span className="dg-analysis__locked-icon" aria-hidden="true">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="4" y="10" width="16" height="10" rx="2" /><path d="M8 10V7a4 4 0 0 1 8 0v3" /></svg>
                  </span>
                  <p>로그인하면 분석 기록이<br />안전하게 저장돼요</p>
                  <button type="button" className="dg-analysis__text-btn" onClick={() => navigate('/login')}>로그인하기</button>
                </div>
              )}
            </div>

          </div>
        </aside>

        <main className="dg-analysis__center">
          <div className="dg-analysis__head">
            {!isSidebarOpen && (
              <button type="button" className="dg-analysis__icon-btn dg-analysis__icon-btn--boxed" onClick={() => setIsSidebarOpen(true)} aria-label="사이드바 열기">
                <SidebarIcon open />
              </button>
            )}
            <div>
              <h2>동영상 분석</h2>
              <p className="dg-analysis__sub">얼굴이 나오는 영상을 올리면 프레임 단위로 딥페이크 여부를 판정합니다.</p>
            </div>
          </div>

          {/* 예시 비디오 — 로컬 샘플 없이 바로 테스트 */}
          <div className="dg-analysis__samples">
            <span className="dg-analysis__samples-label">예시로 체험하기</span>
            <div className="dg-analysis__samples-list">
              {SAMPLE_VIDEOS.map((s) => {
                const isFake = s.label.includes('Fake');
                const isSelected = selectedSample === s.name;
                return (
                  <div
                    key={s.name}
                    className={`dg-analysis__sample${isSelected ? ' is-selected' : ''}`}
                    onClick={() => handleSelectSample(s)}
                    title={s.label}
                  >
                    <video src={`${s.src}#t=0.1`} muted preload="metadata" />
                    {/* 우상단 FAKE/REAL 뱃지 */}
                    <span className={`dg-analysis__sample-tag dg-analysis__sample-tag--${isFake ? 'fake' : 'real'}`}>
                      {isFake ? 'FAKE' : 'REAL'}
                    </span>
                    {/* 하단 그라데이션 + 도메인 */}
                    <div className="dg-analysis__sample-caption">{s.label.split(' ')[0]}</div>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="dg-analysis__panes">
            <div
              className={`dg-analysis__pane dg-analysis__drop${previewUrl ? ' has-preview' : ''}${isCameraOpen ? ' is-camera' : ''}`}
              onClick={() => { if (!previewUrl && !isCameraOpen) document.getElementById('vIn').click(); }}
            >
              {isCameraOpen ? (
                <WebcamCapture mode="video" onCapture={handleFileSelect} onClose={() => setIsCameraOpen(false)} />
              ) : previewUrl ? (
                <>
                  <video className="dg-video__preview" src={previewUrl} controls />
                  {isAnalyzing && <span className="dg-video__preview-badge">분석 중</span>}
                </>
              ) : (
                <div className="dg-analysis__drop-body">
                  {/* 이클립스 오빗 — 로그인 화면 이클립스 링의 축소판 */}
                  <div className="dg-analysis__orbit">
                    <span className="dg-analysis__orbit-halo" aria-hidden="true" />
                    <span className="dg-analysis__orbit-ring" aria-hidden="true" />
                    <div className="dg-analysis__drop-icon"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M12 16V5M7 10l5-5 5 5M5 19h14" /></svg></div>
                  </div>
                  <p className="dg-analysis__drop-title">동영상 업로드</p>
                  <p className="dg-analysis__drop-desc">클릭해서 동영상을 선택하거나 웹캠으로 녹화하세요</p>
                </div>
              )}
              {!isCameraOpen && (
                <MediaSourceBar mode="video" disabled={isAnalyzing}
                  onUpload={() => document.getElementById('vIn').click()}
                  onWebcam={() => setIsCameraOpen(true)} />
              )}
              <input id="vIn" type="file" accept="video/*" hidden
                onChange={(e) => { handleFileSelect(e.target.files[0]); e.target.value = ''; }} />
            </div>

            <div className={`dg-analysis__pane dg-analysis__output${isAnalyzing ? ' is-analyzing' : ''}`}>
              {renderResult()}
            </div>
          </div>
        </main>

        <aside className="dg-analysis__settings">
          <h3 className="dg-analysis__settings-title">분석 설정</h3>

          <div className="dg-analysis__field">
            <p className="dg-analysis__field-label">대상 도메인</p>
            <div className="dg-analysis__segment">
              <button type="button" className={domainType === 'western' ? 'is-on' : ''} onClick={() => setDomainType('western')}>서양인</button>
              <button type="button" className={domainType === 'asian' ? 'is-on' : ''} onClick={() => setDomainType('asian')}>동양인</button>
            </div>
          </div>

          <div className="dg-analysis__field">
            <p className="dg-analysis__field-label">모델 선택</p>
            <div className="dg-analysis__models">
              {/* FAST */}
              <div className={`dg-analysis__model${modelType === 'fast' ? ' is-on' : ''}`} onClick={() => setModelType('fast')}>
                <label className="dg-analysis__model-name">
                  <input type="radio" checked={modelType === 'fast'} onChange={() => setModelType('fast')} />
                  FAST
                </label>
                <span className="dg-analysis__model-desc">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M13 3 4 14h7l-1 7 9-11h-7l1-7z" /></svg>
                  빠른 속도
                  <span className="dg-analysis__model-note">- 추천</span>
                </span>
              </div>
              {/* PRO */}
              <div className={`dg-analysis__model${modelType === 'pro' ? ' is-on' : ''}`} onClick={() => setModelType('pro')}>
                <label className="dg-analysis__model-name">
                  <input type="radio" checked={modelType === 'pro'} onChange={() => setModelType('pro')} />
                  PRO
                </label>
                <span className="dg-analysis__model-desc">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="9" /><circle cx="12" cy="12" r="4" /><circle cx="12" cy="12" r="0.5" fill="currentColor" /></svg>
                  높은 정확도
                  <span className="dg-analysis__model-note">- 회원 전용</span>
                </span>
              </div>
            </div>
          </div>

          <Button variant="primary" className="dg-btn--block dg-analysis__submit" onClick={handlePredict} disabled={isAnalyzing}>
            {isAnalyzing ? "분석 중..." : "분석 시작"}
          </Button>
        </aside>
      </div>

      <Footer />
    </div>
  );
};

export default VideoAnalysisPage;

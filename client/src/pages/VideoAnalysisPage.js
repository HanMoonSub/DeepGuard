import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

// 예시 비디오 (public/samples/videos/ 하위에 실제 파일 배치 필요 / 파일명 다르면 이 배열만 수정)
const SAMPLE_VIDEOS = [
  { name: 'western_fake.mp4', src: '/samples/videos/western_fake.mp4', domain: 'western', label: '서양인 Fake' },
  { name: 'western_real.mp4', src: '/samples/videos/western_real.mp4', domain: 'western', label: '서양인 Real' },
];

const VideoAnalysisPage = ({ sessionUser, onLogout, setSessionUser }) => {
  const navigate = useNavigate();

  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [history, setHistory] = useState([]);

  const [showOptions, setShowOptions] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState([]);
  const [selectedSample, setSelectedSample] = useState(null);

  const [versionType, setVersionType] = useState('v2');
  const [modelType, setModelType] = useState('fast');
  const [domainType, setDomainType] = useState('western');

  const [result, setResult] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');

  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  const pollingTimer = useRef(null);
  const isMounted = useRef(true);

  const sideBarStyle = {
    width: isSidebarOpen ? '280px' : '0',
    backgroundColor: '#050505',
    borderRight: isSidebarOpen ? '1px solid #222' : 'none',
    display: 'flex',
    flexDirection: 'column',
    padding: isSidebarOpen ? '25px' : '0',
    overflow: 'hidden',
    flexShrink: 0,
    transition: 'width 0.3s ease, padding 0.3s ease',
  };
  const centerZoneStyle = { flex: 1, padding: '40px', display: 'flex', flexDirection: 'column', gap: '20px', overflowY: 'auto', position: 'relative' };
  const rightPanelStyle = { width: '340px', backgroundColor: '#0D0D0D', borderLeft: '1px solid #222', padding: '30px', display: 'flex', flexDirection: 'column' };
  const innerBoxStyle = { flex: 1, border: '2px dashed #333', borderRadius: '20px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', backgroundColor: '#0a0a0a', cursor: 'pointer', position: 'relative', transition: 'all 0.3s' };
  const plusBtnStyle = { width: '60px', height: '60px', borderRadius: '50%', backgroundColor: '#1A2C50', display: 'flex', justifyContent: 'center', alignItems: 'center', fontSize: '30px', color: '#39FF14', marginBottom: '20px', boxShadow: '0 0 15px rgba(57, 255, 20, 0.2)' };

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
  }, [fetchHistory]);

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
    formData.append('version_type', versionType);

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
    setShowOptions(false);
    setSelectedSample(null); // 샘플 외 경로로 들어오면 선택 해제
  };

  // 예시 비디오를 File로 변환 후 기존 업로드 흐름에 주입 + 도메인 자동 세팅
  const handleSelectSample = async (sample) => {
    try {
      const res = await fetch(sample.src);
      const blob = await res.blob();
      const file = new File([blob], sample.name, { type: blob.type });
      if (sample.domain === 'asian') setVersionType('v2'); // 동양인은 v2 전용
      setDomainType(sample.domain);
      handleFileSelect(file);
      setSelectedSample(sample.name); // handleFileSelect 리셋 이후 선택 표시
    } catch (e) {
      alert("예시 파일을 불러오지 못했어요.");
    }
  };

  const toggleSelect = (id) => {
    setSelectedIds(prev => prev.includes(id) ? prev.filter(item => item !== id) : [...prev, id]);
  };

  const handleDeleteSelected = async () => {
    if (selectedIds.length === 0) return;
    if (!window.confirm(`${selectedIds.length}개의 기록을 완전히 삭제하시겠습니까?`)) return;

    try {
      const deletePromises = selectedIds.map(id =>
        axios.delete(`/video/history/${id}`)
      );
      await Promise.all(deletePromises);
      alert("선택한 비디오 기록이 삭제되었습니다.");
      fetchHistory();
      setSelectedIds([]);
      setIsEditMode(false);
    } catch (error) {
      console.error(error);
      alert("서버에서 기록을 삭제하는 중 오류가 발생했습니다.");
    }
  };

  const handleLogoutClick = async () => {
    if (!window.confirm("로그아웃 하시겠습니까?")) return;
    try {
      await axios.get('/auth/logout');
      onLogout();
      navigate('/video-analysis');
    } catch (error) { alert("로그아웃 실패"); }
  };

  return (
    <div style={{ display: 'flex', backgroundColor: '#000', height: '100vh', width: '100vw', color: 'white', fontFamily: 'sans-serif', overflow: 'hidden' }}>

      <aside style={sideBarStyle}>
        <div style={{ width: '230px', display: 'flex', flexDirection: 'column', height: '100%', flexShrink: 0 }}>
          <button onClick={() => { setFile(null); setPreviewUrl(null); setResult(null); setStatusMessage(''); setShowOptions(false); setSelectedSample(null); if (pollingTimer.current) clearInterval(pollingTimer.current); }} style={{ backgroundColor: '#1A2C50', color: 'white', padding: '14px', borderRadius: '10px', border: 'none', marginBottom: '35px', cursor: 'pointer', fontWeight: 'bold' }}>+ 새 영상 분석</button>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
            <h3 style={{ fontSize: '16px', margin: 0 }}>내 비디오 기록</h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              {sessionUser && history.length > 0 && (
                <button onClick={() => { setIsEditMode(!isEditMode); setSelectedIds([]); }} style={{ background: 'none', border: 'none', color: '#39FF14', cursor: 'pointer', fontSize: '12px' }}>
                  {isEditMode ? '취소' : '편집'}
                </button>
              )}
              <button onClick={() => setIsSidebarOpen(false)} style={{ background: 'none', border: 'none', color: '#666', cursor: 'pointer', display: 'flex', padding: 0 }} aria-label="사이드바 접기">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="4" width="18" height="16" rx="2" /><line x1="9" y1="4" x2="9" y2="20" /><polyline points="15 9 12 12 15 15" /></svg>
              </button>
            </div>
          </div>

          <div style={{ flex: 1, overflowY: 'auto' }}>
            {sessionUser ? (
              history.map((item, index) => {
                const currentId = item.video_id || item.id;
                const isSelected = selectedIds.includes(currentId);
                const vType = item.version_type ? item.version_type.toUpperCase() : 'V2';
                const dType = item.domain_type || '서양인';
                const mType = item.model_type ? item.model_type.toUpperCase() : 'FAST';

                return (
                  <div key={currentId || index} style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '15px' }}>
                    {isEditMode && <input type="checkbox" checked={isSelected} onChange={() => toggleSelect(currentId)} style={{ accentColor: '#39FF14', width: '18px', height: '18px' }} />}
                    <div
                      onClick={() => !isEditMode && navigate('/video-detail', {
                        state: { video_id: currentId }
                      })}
                      style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '15px', padding: '12px', backgroundColor: '#111', borderRadius: '12px', border: isSelected ? '1px solid #39FF14' : '1px solid #222', cursor: isEditMode ? 'default' : 'pointer' }}
                    >
                      <div style={{ width: '45px', height: '45px', backgroundColor: '#222', borderRadius: '8px', overflow: 'hidden' }}>
                        {item.video_loc ? (
                          <video
                            src={`${item.video_loc}#t=0.1`}
                            muted
                            preload="metadata"
                            style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                          />
                        ) : '📹'}
                      </div>
                      <div>
                        <div style={{ fontSize: '12px', color: '#aaa', fontWeight: 'bold', marginBottom: '2px' }}>{vType} | {dType} | {mType}</div>
                        <div style={{ fontSize: '11px', color: '#888' }}>
                          {item.created_at?.slice(0, 10)}
                          <span style={{ color: '#555', marginLeft: '8px' }}>{item.created_at?.slice(11, 16)}</span>
                        </div>
                        <div style={{
                          fontSize: '12px',
                          fontWeight: 'bold',
                          color: item.label === 'FAKE' ? '#FF4B4B' : item.label === 'REAL' ? '#39FF14' : '#888'
                        }}>{item.label}</div>
                      </div>
                    </div>
                  </div>
                );
              })
            ) : <p style={{ color: '#444', textAlign: 'center' }}>로그인 필요</p>}
          </div>

          {isEditMode && (
            <button
              onClick={handleDeleteSelected}
              disabled={selectedIds.length === 0}
              style={{ width: '100%', padding: '12px', backgroundColor: selectedIds.length > 0 ? '#FF4B4B' : '#222', color: '#fff', border: 'none', borderRadius: '8px', cursor: selectedIds.length > 0 ? 'pointer' : 'not-allowed', marginTop: '10px', fontWeight: 'bold' }}
            >
              선택 삭제 ({selectedIds.length})
            </button>
          )}

          <div style={{ borderTop: '1px solid #222', paddingTop: '20px', marginTop: '20px' }}>
            <button onClick={() => navigate('/main')} style={{ width: '100%', padding: '12px', backgroundColor: 'transparent', color: '#fff', border: '1px solid #444', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold', marginBottom: '15px' }}>메인 화면</button>
            {sessionUser ? (
              <div style={{ textAlign: 'center' }}>
                <p style={{ fontSize: '13px', color: '#666', marginBottom: '10px' }}>{sessionUser.name}님 접속 중</p>
                <button onClick={handleLogoutClick} style={{ width: '100%', padding: '12px', backgroundColor: 'transparent', color: '#FF4B4B', border: '1px solid #FF4B4B', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }}>로그아웃</button>
              </div>
            ) : <button onClick={() => navigate('/login')} style={{ width: '100%', padding: '12px', backgroundColor: '#1A2C50', color: 'white', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }}>로그인</button>}
          </div>
        </div>
      </aside>

      <main style={centerZoneStyle}>
        {!isSidebarOpen && (
          <button
            onClick={() => setIsSidebarOpen(true)}
            style={{ position: 'absolute', top: '20px', left: '20px', zIndex: 10, background: '#111', border: '1px solid #333', borderRadius: '8px', color: '#39FF14', cursor: 'pointer', padding: '8px', display: 'flex' }}
            aria-label="사이드바 열기"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="4" width="18" height="16" rx="2" /><line x1="9" y1="4" x2="9" y2="20" /><polyline points="12 9 15 12 12 15" /></svg>
          </button>
        )}
        <h2 style={{ fontSize: '28px', fontWeight: 'bold' }}>Deep Guard AI</h2>

        {/* 예시 비디오 — 로컬 샘플 없이 바로 테스트 */}
        <div style={{ display: 'flex', gap: '12px', overflowX: 'auto', paddingBottom: '6px' }}>
          {SAMPLE_VIDEOS.map((s) => {
            const isFake = s.label.includes('Fake');
            const tagColor = isFake ? '#FF4B4B' : '#39FF14';
            const isSelected = selectedSample === s.name;
            return (
              <div key={s.name} className="sample-thumb" onClick={() => handleSelectSample(s)} title={s.label}
                style={{
                  position: 'relative', flexShrink: 0, width: '100px', height: '74px',
                  borderRadius: '12px', overflow: 'hidden', cursor: 'pointer',
                  border: isSelected ? '2px solid #39FF14' : '1px solid #2a2a2a',
                  boxShadow: isSelected ? '0 0 0 1px #39FF14, 0 6px 18px rgba(57,255,20,0.4)' : 'none'
                }}>
                <video src={`${s.src}#t=0.1`} muted preload="metadata"
                  style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block', pointerEvents: 'none' }} />
                {/* 우상단 FAKE/REAL 뱃지 */}
                <span style={{ position: 'absolute', top: '6px', right: '6px', fontSize: '8px', fontWeight: 'bold', color: tagColor, background: 'rgba(0,0,0,0.55)', border: `1px solid ${tagColor}`, borderRadius: '4px', padding: '1px 5px', letterSpacing: '0.5px', backdropFilter: 'blur(2px)' }}>
                  {isFake ? 'FAKE' : 'REAL'}
                </span>
                {/* 하단 그라데이션 + 도메인 */}
                <div style={{ position: 'absolute', left: 0, right: 0, bottom: 0, padding: '14px 8px 5px', background: 'linear-gradient(transparent, rgba(0,0,0,0.88))' }}>
                  <span style={{ fontSize: '10px', color: '#e5e5e5', fontWeight: 'bold' }}>{s.label.split(' ')[0]}</span>
                </div>
              </div>
            );
          })}
        </div>

        <div style={{ display: 'flex', flex: 1, gap: '20px' }}>
          <div style={{...innerBoxStyle, border: showOptions ? '2px solid #39FF14' : '2px dashed #333'}} onClick={() => { if(!previewUrl) setShowOptions(!showOptions); }}>
            {previewUrl ? <video src={previewUrl} controls style={{ maxWidth: '95%', maxHeight: '95%', objectFit: 'contain' }} /> : (
               <div style={{textAlign:'center', display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
                 <div style={plusBtnStyle}>+</div>
                 <p style={{fontSize:'20px', fontWeight: 'bold', marginBottom: '8px'}}>Video Upload</p>
                 <p style={{color:'#555', fontSize:'14px'}}>동영상을 업로드하세요</p>
                 {showOptions && (
                   <div style={{ marginTop: '25px', display: 'flex', gap: '12px' }}>
                     <button onClick={(e) => { e.stopPropagation(); document.getElementById('vIn').click(); }} style={{ padding: '10px 18px', backgroundColor: '#222', color: '#39FF14', border: '1px solid #39FF14', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }}>내 PC 파일</button>
                     {/* <button onClick={(e) => { e.stopPropagation(); alert("준비 중입니다."); setShowOptions(false); }} style={{ padding: '10px 18px', backgroundColor: '#222', color: '#fff', border: '1px solid #444', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }}>Cloud Drive</button> */}
                   </div>
                 )}
               </div>
            )}
            <input id="vIn" type="file" accept="video/*" hidden onChange={(e) => handleFileSelect(e.target.files[0])} />
          </div>

          <div style={{...innerBoxStyle, border: isAnalyzing ? 'none' : '2px dashed #333', cursor: 'default', position: 'relative', overflow: 'hidden'}}>
            {isAnalyzing ? (
              <div style={{ textAlign: 'center', width: '100%', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', zIndex: 2, backgroundColor: '#050505' }}>
                <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, borderRadius: '20px', padding: '2px', background: 'linear-gradient(45deg, #ff0000, #ff7300, #fffb00, #48ff00, #00ffd5, #002bff, #7a00ff, #ff00c8, #ff0000)', backgroundSize: '400%', zIndex: -1, animation: 'rainbow 15s linear infinite' }} />
                <div style={{ position: 'absolute', top: '2px', left: '2px', right: '2px', bottom: '2px', backgroundColor: '#0a0a0a', borderRadius: '18px', zIndex: -1, overflow: 'hidden' }}>
                  <div style={{ position: 'absolute', width: '100%', height: '2px', background: 'linear-gradient(90deg, transparent, #39FF14, transparent)', boxShadow: '0 0 15px #39FF14', top: '-10%', animation: 'scanLine 3s ease-in-out infinite' }} />
                </div>
                <div style={{ marginBottom: '40px' }}>
                  <p style={{ fontSize: '11px', letterSpacing: '4px', color: '#39FF14', opacity: 0.8, marginBottom: '5px', fontWeight: 'bold' }}>VIDEO CORE ENGINE</p>
                  <h3 style={{ fontSize: '22px', fontWeight: '900', margin: 0, color: '#fff', letterSpacing: '1px' }}>SCANNING FRAMES...</h3>
                </div>
                <div style={{ width: '65%' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                    <span style={{ fontSize: '11px', color: '#555', fontWeight: 'bold' }}>{statusMessage || "PROCESSING..."}</span>
                    <span style={{ fontSize: '11px', color: '#39FF14', fontWeight: 'bold', animation: 'blink 1s step-end infinite' }}>LIVE</span>
                  </div>
                  <div style={{ width: '100%', height: '3px', backgroundColor: '#111', borderRadius: '2px', overflow: 'hidden' }}>
                    <div style={{ width: '100%', height: '100%', background: 'linear-gradient(90deg, #1A2C50, #39FF14, #1A2C50)', backgroundSize: '200% 100%', animation: 'loadingBar 1.5s linear infinite' }} />
                  </div>
                </div>
              </div>
            ) : result ? (
              result.status === 'WARNING' ? (
                <div style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '20px' }}>
                  <div style={{ width: '64px', height: '64px', borderRadius: '50%', background: 'rgba(255,165,0,0.1)', border: '1px solid rgba(255,165,0,0.35)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '20px' }}>
                    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#FFA500" strokeWidth="2">
                      <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" /><line x1="11" y1="8" x2="11" y2="11" /><line x1="11" y1="14" x2="11.01" y2="14" />
                    </svg>
                  </div>
                  <p style={{ fontSize: '24px', fontWeight: 'bold', color: '#FFA500', margin: '0 0 12px' }}>분석을 완료하지 못했어요</p>
                  <p style={{ fontSize: '14px', color: '#aaa', lineHeight: '1.7', maxWidth: '300px', margin: 0 }}>
                    {result.result_msg}
                  </p>
                </div>
              ) : (() => {
                const label = result.label || (result.score > 0.5 ? 'FAKE' : 'REAL');
                const color = label === 'FAKE' ? '#FF4B4B' : '#39FF14';
                return (
                  <div style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                    <p style={{ fontSize: '56px', fontWeight: 900, color, margin: 0, lineHeight: 1 }}>{label}</p>
                    <button
                      onClick={() => navigate('/video-detail', { state: { video_id: result.video_id } })}
                      style={{
                        marginTop: '80px',
                        color,
                        background: 'none',
                        border: 'none',
                        padding: 0,
                        fontWeight: 'bold',
                        fontSize: '14px',
                        textDecoration: 'underline',
                        textUnderlineOffset: '3px',
                        cursor: 'pointer'
                      }}
                    >
                      상세 결과 보기
                    </button>
                  </div>
                );
              })()
            ) : <p style={{ color: '#222' }}>READY TO ANALYZE</p>}
          </div>
        </div>
      </main>

      <aside style={rightPanelStyle}>
        <h3 style={{ fontSize: '22px', marginBottom: '30px', borderBottom: '1px solid #222', paddingBottom: '15px' }}>분석 설정</h3>
        <div style={{ marginBottom: '25px' }}>
          <p style={{ color: '#39FF14', fontSize: '14px', marginBottom: '10px', fontWeight: 'bold' }}>버전 선택</p>
          <div style={{ display: 'flex', backgroundColor: '#000', borderRadius: '12px', padding: '5px', border: '1px solid #333' }}>
            <button onClick={() => setVersionType('v1')} style={{ flex: 1, padding: '10px', borderRadius: '8px', border: 'none', backgroundColor: versionType === 'v1' ? '#222' : 'transparent', color: versionType === 'v1' ? '#39FF14' : '#666', cursor: 'pointer' }}>V1</button>
            <button
              onClick={() => setVersionType('v2')}
              style={{ position: 'relative', flex: 1, padding: '10px', borderRadius: '8px', border: 'none', backgroundColor: versionType === 'v2' ? '#222' : 'transparent', color: versionType === 'v2' ? '#39FF14' : '#666', cursor: 'pointer' }}
            >
              V2
              <span style={{ position: 'absolute', top: '-9px', right: '-4px', transform: 'rotate(18deg)', lineHeight: 0 }}>
                <svg width="22" height="22" viewBox="0 0 24 24" fill="#FFD700">
                  <path d="M5 16L3 5l5.5 4L12 4l3.5 5L21 5l-2 11H5zm0 2.5h14v1.5H5z" />
                </svg>
              </span>
            </button>
          </div>
        </div>
        <div style={{ marginBottom: '25px' }}>
          <p style={{ color: '#39FF14', fontSize: '14px', marginBottom: '10px', fontWeight: 'bold' }}>대상 도메인</p>
          <div style={{ display: 'flex', gap: '10px' }}>
            <button onClick={() => setDomainType('western')} style={{ flex: 1, padding: '12px', borderRadius: '8px', border: 'none', backgroundColor: domainType === 'western' ? '#222' : '#000', color: domainType === 'western' ? '#39FF14' : '#666', cursor: 'pointer' }}>서양인</button>
            <button onClick={() => setDomainType('asian')} disabled={versionType === 'v1'} style={{ flex: 1, padding: '12px', borderRadius: '8px', border: 'none', backgroundColor: domainType === 'asian' ? '#222' : '#000', color: domainType === 'asian' ? '#39FF14' : '#666', cursor: versionType === 'v1' ? 'not-allowed' : 'pointer' }}>동양인</button>
          </div>
        </div>
        <p style={{ color: '#39FF14', fontSize: '14px', marginBottom: '14px', fontWeight: 'bold' }}>모델 선택</p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {/* FAST */}
          <div
            onClick={() => setModelType('fast')}
            style={{ position: 'relative', padding: '14px', background: '#111', border: `1px solid ${modelType === 'fast' ? '#39FF14' : '#333'}`, borderRadius: '10px', cursor: 'pointer' }}
          >
            <span style={{ position: 'absolute', top: '12px', right: '12px', fontSize: '10px', fontWeight: 'bold', color: '#1a1400', background: 'linear-gradient(135deg, #FFD700, #FFA500)', padding: '3px 10px', borderRadius: '20px' }}>
              RECOMMENDED
            </span>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
              <input type="radio" checked={modelType === 'fast'} onChange={() => setModelType('fast')} style={{ accentColor: '#39FF14' }} />
              <span style={{ color: '#fff', fontWeight: 'bold', fontSize: '14px' }}>FAST</span>
            </div>
            <div style={{ paddingLeft: '26px' }}>
              <span style={{ fontSize: '11px', color: '#888', display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#39FF14" strokeWidth="2"><path d="M13 3 4 14h7l-1 7 9-11h-7l1-7z" /></svg>
                빠른 속도
              </span>
            </div>
          </div>
          {/* PRO */}
          <div
            onClick={() => setModelType('pro')}
            style={{ position: 'relative', padding: '14px', background: '#111', border: `1px solid ${modelType === 'pro' ? '#39FF14' : '#333'}`, borderRadius: '10px', cursor: 'pointer' }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px' }}>
              <input type="radio" checked={modelType === 'pro'} onChange={() => setModelType('pro')} style={{ accentColor: '#39FF14' }} />
              <span style={{ color: '#fff', fontWeight: 'bold', fontSize: '14px' }}>PRO</span>
              <span style={{ fontSize: '10px', color: '#FFD700', background: 'rgba(255,215,0,0.1)', border: '1px solid rgba(255,215,0,0.3)', padding: '1px 7px', borderRadius: '5px', fontWeight: 'bold' }}>회원 전용</span>
            </div>
            <div style={{ paddingLeft: '26px' }}>
              <span style={{ fontSize: '11px', color: '#888', display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#39FF14" strokeWidth="2"><circle cx="12" cy="12" r="9" /><circle cx="12" cy="12" r="4" /><circle cx="12" cy="12" r="0.5" fill="#39FF14" /></svg>
                높은 정확도
              </span>
            </div>
          </div>
        </div>
        <button onClick={handlePredict} disabled={isAnalyzing} style={{ marginTop: 'auto', padding: '18px', backgroundColor: isAnalyzing ? '#222' : '#1A2C50', color: 'white', border: 'none', borderRadius: '12px', fontWeight: 'bold', fontSize: '16px', cursor: isAnalyzing ? 'not-allowed' : 'pointer' }}>
          {isAnalyzing ? "분석 중..." : "분석 시작"}
        </button>
      </aside>

      <style>{`
        @keyframes rainbow { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
        @keyframes loadingBar { 0% { background-position: 100% 0%; } 100% { background-position: -100% 0%; } }
        @keyframes scanLine { 0% { top: 0%; opacity: 0; } 10% { opacity: 1; } 90% { opacity: 1; } 100% { top: 100%; opacity: 0; } }
        @keyframes blink { 50% { opacity: 0; } }

      `}</style>
    </div>
  );
};

export default VideoAnalysisPage;
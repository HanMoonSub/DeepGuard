import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const apiUrl = process.env.REACT_APP_API_URL || "http://localhost:8000";


const AnalysisPage = ({ sessionUser, onLogout, setSessionUser }) => {
  const navigate = useNavigate();
  
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [history, setHistory] = useState([]);
  
  const [showOptions, setShowOptions] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState([]);

  const [versionType, setVersionType] = useState('v2'); 
  const [modelType, setModelType] = useState('fast');   
  const [domainType, setDomainType] = useState('western'); 
  
  const [result, setResult] = useState(null); 
  const [statusMessage, setStatusMessage] = useState('');

  const pollingTimer = useRef(null);
  const isMounted = useRef(true);

  const sideBarStyle = { width: '280px', backgroundColor: '#050505', borderRight: '1px solid #222', display: 'flex', flexDirection: 'column', padding: '25px' };
  const centerZoneStyle = { flex: 1, padding: '40px', display: 'flex', flexDirection: 'column', gap: '20px', overflowY: 'auto' };
  const rightPanelStyle = { width: '340px', backgroundColor: '#0D0D0D', borderLeft: '1px solid #222', padding: '30px', display: 'flex', flexDirection: 'column' };
  const innerBoxStyle = { flex: 1, border: '2px dashed #333', borderRadius: '20px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', backgroundColor: '#0a0a0a', cursor: 'pointer', position: 'relative', transition: 'all 0.3s' };
  const plusBtnStyle = { width: '60px', height: '60px', borderRadius: '50%', backgroundColor: '#1A2C50', display: 'flex', justifyContent: 'center', alignItems: 'center', fontSize: '30px', color: '#39FF14', marginBottom: '20px', boxShadow: '0 0 15px rgba(57, 255, 20, 0.2)' };

  const fetchHistory = useCallback(async () => {
    if (!sessionUser) return;
    try {
      const response = await axios.get('/image/history');
      if (response.data.status === "success") {
        setHistory(response.data.context || []); 
      }
    } catch (e) { console.log("히스토리 로드 실패"); }
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

  const startPolling = useCallback((imageId) => {
    if (!imageId) return;
    if (pollingTimer.current) clearInterval(pollingTimer.current);

    pollingTimer.current = setInterval(async () => {
      if (!isMounted.current) return;
      try {
        const response = await axios.get(`/inference/image/${imageId}`);
        const data = response.data;
        if (data.status === 'SUCCESS' || data.prob !== undefined || data.analysis !== undefined) {
          clearInterval(pollingTimer.current);
          pollingTimer.current = null;
          // [수정] image_id 포함
          setResult({ ...data, image_id: imageId });
          setIsAnalyzing(false);
          setStatusMessage('');
          fetchHistory();
        } else if (data.status === 'FAILED' || data.status === 'ERROR') {
          clearInterval(pollingTimer.current);
          pollingTimer.current = null;
          setIsAnalyzing(false);
          setStatusMessage('');
          alert(data.result_msg || "분석 실패");
        // [수정] WARNING 분기 추가 — 얼굴 미탐지 등 부분 실패, 상세보기 없이 메시지만 표시
        } else if (data.status === 'WARNING') {
          clearInterval(pollingTimer.current);
          pollingTimer.current = null;
          setResult({ ...data, status: 'WARNING' });
          setIsAnalyzing(false);
          setStatusMessage('');
          fetchHistory();
        } else {
          setStatusMessage(data.message || "이미지 분석 진행 중...");
        }
      } catch (err) { setResult(null); }
    }, 2000);
  }, [fetchHistory]);

  const handlePredict = async () => {
    if (modelType === 'pro' && !sessionUser) {
      alert("PRO 모델 분석은 로그인이 필요합니다.");
      navigate('/login');
      return;
    }
    if (!file) return alert("이미지를 먼저 업로드해주세요.");

    const formData = new FormData();
    formData.append('imagefile', file);
    formData.append('model_type', modelType);
    formData.append('domain_type', domainType === 'western' ? '서양인' : '동양인');
    formData.append('version_type', versionType);

    setIsAnalyzing(true);
    setStatusMessage('서버 접수 중...');
    setResult(null);

    try {
      const response = await axios.post('/inference/image', formData);
      const imageId = response.data?.image_id || response.data; 
      if (imageId) startPolling(imageId);
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
  };

  const toggleSelect = (id) => {
    setSelectedIds(prev => prev.includes(id) ? prev.filter(item => item !== id) : [...prev, id]);
  };

  const handleDeleteSelected = async () => {
    if (selectedIds.length === 0) return;
    if (!window.confirm(`${selectedIds.length}개의 기록을 삭제하시겠습니까?`)) return;
    
    try {
      const deletePromises = selectedIds.map(id => 
        axios.delete(`/image/history/${id}`)
      );
      await Promise.all(deletePromises);
      alert("선택한 이미지 기록이 성공적으로 삭제되었습니다.");
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
      navigate('/main');
    } catch (error) { alert("로그아웃 실패"); }
  };

  return (
    <div style={{ display: 'flex', backgroundColor: '#000', height: '100vh', width: '100vw', color: 'white', fontFamily: 'sans-serif', overflow: 'hidden' }}>
      <aside style={sideBarStyle}>
        <button onClick={() => { setFile(null); setPreviewUrl(null); setResult(null); setStatusMessage(''); setShowOptions(false); if (pollingTimer.current) clearInterval(pollingTimer.current); }} style={{ backgroundColor: '#1A2C50', color: 'white', padding: '14px', borderRadius: '10px', border: 'none', marginBottom: '35px', cursor: 'pointer', fontWeight: 'bold' }}>+ 새 분석 시작</button>
        
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <h3 style={{ fontSize: '18px', margin: 0 }}>내 작업 기록</h3>
          {sessionUser && history.length > 0 && (
            <button onClick={() => { setIsEditMode(!isEditMode); setSelectedIds([]); }} style={{ background: 'none', border: 'none', color: '#39FF14', cursor: 'pointer', fontSize: '12px' }}>
              {isEditMode ? '취소' : '편집'}
            </button>
          )}
        </div>

        <div style={{ flex: 1, overflowY: 'auto' }}>
          {sessionUser ? (
            history.map((item, index) => {
              const p = item.prob ?? item.score ?? item.analysis?.prob ?? -1;
              const isSelected = selectedIds.includes(item.image_id);
              const vType = item.version_type ? item.version_type.toUpperCase() : 'V2';
              const dType = item.domain_type || '서양인';
              const mType = item.model_type ? item.model_type.toUpperCase() : 'FAST';


              let itemLabel = 'UNKNOWN';
              if (item.label && item.label !== 'UNKNOWN') {
                itemLabel = item.label.toUpperCase();
              } else if (p !== -1) {
                itemLabel = p > 0.5 ? 'FAKE' : 'REAL';
              }

              return (
                <div key={item.image_id || index} style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '15px' }}>
                  {isEditMode && <input type="checkbox" checked={isSelected} onChange={() => toggleSelect(item.image_id)} style={{ accentColor: '#39FF14', width: '18px', height: '18px' }} />}
                  <div 
                    // [수정] image_id만 전달
                    onClick={() => !isEditMode && navigate('/analysis-detail', { 
                      state: { image_id: item.image_id } 
                    })}
                    style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '15px', padding: '12px', backgroundColor: '#111', borderRadius: '12px', border: isSelected ? '1px solid #39FF14' : '1px solid #222', cursor: isEditMode ? 'default' : 'pointer' }}
                  >
                    <div style={{ width: '45px', height: '45px', backgroundColor: '#222', borderRadius: '8px', overflow: 'hidden' }}>
                      {item.image_loc ? <img src={`${apiUrl}${item.image_loc}`} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} /> : '🖼️'}
                    </div>
                    <div>
                      <div style={{ fontSize: '12px', color: '#aaa', fontWeight: 'bold', marginBottom: '2px' }}>{vType} | {dType} | {mType}</div>
                      <div style={{ fontSize: '11px', color: '#555' }}>{item.created_at?.split('T')[0]}</div>
                      <div style={{ fontSize: '13px', color: '#fff', fontWeight: 'bold' }}>{itemLabel}</div>
                    </div>
                  </div>
                </div>
              );
            })
          ) : <p style={{ color: '#444', textAlign: 'center' }}>로그인 필요</p>}
        </div>

        {isEditMode && <button onClick={handleDeleteSelected} disabled={selectedIds.length === 0} style={{ width: '100%', padding: '12px', backgroundColor: selectedIds.length > 0 ? '#FF4B4B' : '#222', color: '#fff', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold', marginTop: '10px' }}>선택 삭제 ({selectedIds.length})</button>}

        <div style={{ borderTop: '1px solid #222', paddingTop: '20px', marginTop: '20px' }}>
          <button onClick={() => navigate('/main')} style={{ width: '100%', padding: '12px', backgroundColor: 'transparent', color: '#fff', border: '1px solid #444', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold', marginBottom: '15px' }}>메인 화면</button>
          {sessionUser ? (
            <div style={{ textAlign: 'center' }}>
              <p style={{ fontSize: '13px', color: '#666', marginBottom: '10px' }}>{sessionUser.name}님 접속 중</p>
              <button onClick={handleLogoutClick} style={{ width: '100%', padding: '12px', backgroundColor: 'transparent', color: '#FF4B4B', border: '1px solid #FF4B4B', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }}>로그아웃</button>
            </div>
          ) : <button onClick={() => navigate('/login')} style={{ width: '100%', padding: '12px', backgroundColor: '#1A2C50', color: 'white', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }}>로그인</button>}
        </div>
      </aside>

      <main style={centerZoneStyle}>
        <h2 style={{ fontSize: '28px', fontWeight: 'bold' }}>Deep Guard AI</h2>
        <div style={{ display: 'flex', flex: 1, gap: '20px' }}>
          <div style={{...innerBoxStyle, border: showOptions ? '2px solid #39FF14' : '2px dashed #333'}} onClick={() => { if(!previewUrl) setShowOptions(!showOptions); }}>
            {previewUrl ? <img src={previewUrl} alt="preview" style={{ maxWidth: '95%', maxHeight: '95%', objectFit: 'contain' }} /> : (
                <div style={{textAlign:'center', display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
                  <div style={plusBtnStyle}>+</div>
                  <p style={{fontSize:'20px', fontWeight: 'bold', marginBottom: '8px'}}>Image Upload</p>
                  <p style={{color:'#555', fontSize:'14px'}}>이미지를 업로드하세요</p>
                  {showOptions && (
                    <div style={{ marginTop: '25px', display: 'flex', gap: '12px' }}>
                      <button onClick={(e) => { e.stopPropagation(); document.getElementById('fIn').click(); }} style={{ padding: '10px 18px', backgroundColor: '#222', color: '#39FF14', border: '1px solid #39FF14', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }}>내 PC 파일</button>
                      <button onClick={(e) => { e.stopPropagation(); alert("준비 중입니다."); setShowOptions(false); }} style={{ padding: '10px 18px', backgroundColor: '#222', color: '#fff', border: '1px solid #444', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }}>Cloud Drive</button>
                    </div>
                  )}
                </div>
            )}
            <input id="fIn" type="file" hidden onChange={(e) => handleFileSelect(e.target.files[0])} />
          </div>

          <div style={{...innerBoxStyle, border: isAnalyzing ? 'none' : '2px dashed #333', cursor: 'default', position: 'relative', overflow: 'hidden'}}>
            {isAnalyzing ? (
              <div style={{ textAlign: 'center', width: '100%', height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', zIndex: 2, backgroundColor: '#050505' }}>
                <div style={{ position: 'absolute', top: 0, left: 0, right: 0, bottom: 0, borderRadius: '20px', padding: '2px', background: 'linear-gradient(45deg, #ff0000, #ff7300, #fffb00, #48ff00, #00ffd5, #002bff, #7a00ff, #ff00c8, #ff0000)', backgroundSize: '400%', zIndex: -1, animation: 'rainbow 15s linear infinite' }} />
                
                <div style={{ position: 'absolute', top: '2px', left: '2px', right: '2px', bottom: '2px', backgroundColor: '#0a0a0a', borderRadius: '18px', zIndex: -1, overflow: 'hidden' }}>
                  <div style={{ position: 'absolute', width: '100%', height: '2px', background: 'linear-gradient(90deg, transparent, #39FF14, transparent)', boxShadow: '0 0 15px #39FF14', top: '-10%', animation: 'scanLine 3s ease-in-out infinite' }} />
                </div>

                <div style={{ marginBottom: '40px' }}>
                  <p style={{ fontSize: '11px', letterSpacing: '4px', color: '#39FF14', opacity: 0.8, marginBottom: '5px', fontWeight: 'bold' }}>SYSTEM ENGINE</p>
                  <h3 style={{ fontSize: '22px', fontWeight: '900', margin: 0, color: '#fff', letterSpacing: '1px' }}>AI SCANNING...</h3>
                </div>
                
                <div style={{ width: '65%' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
                    <span style={{ fontSize: '11px', color: '#555', fontWeight: 'bold' }}>{statusMessage || "ANALYZING DATA..."}</span>
                    <span style={{ fontSize: '11px', color: '#39FF14', fontWeight: 'bold', animation: 'blink 1s step-end infinite' }}>LIVE_CORE</span>
                  </div>
                  <div style={{ width: '100%', height: '3px', backgroundColor: '#111', borderRadius: '2px', overflow: 'hidden' }}>
                    <div style={{ width: '100%', height: '100%', background: 'linear-gradient(90deg, #1A2C50, #39FF14, #1A2C50)', backgroundSize: '200% 100%', animation: 'loadingBar 1.5s linear infinite' }} />
                  </div>
                </div>

                <style>{`
                  @keyframes rainbow { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
                  @keyframes loadingBar { 0% { background-position: 100% 0%; } 100% { background-position: -100% 0%; } }
                  @keyframes scanLine { 0% { top: 0%; opacity: 0; } 10% { opacity: 1; } 90% { opacity: 1; } 100% { top: 100%; opacity: 0; } }
                  @keyframes blink { 50% { opacity: 0; } }
                `}</style>
              </div>
            ) : result ? (
              // [수정] WARNING 분기 — result_msg만 표시, 상세보기 없음
              result.status === 'WARNING' ? (
                <div style={{ textAlign: 'center', padding: '20px' }}>
                  <p style={{ fontSize: '36px', fontWeight: 'bold', color: '#FFA500', marginBottom: '16px' }}>
                    ⚠ UNDETECTED
                  </p>
                  <p style={{ fontSize: '14px', color: '#888', lineHeight: '1.7', maxWidth: '300px', margin: '0 auto' }}>
                    {result.result_msg || result.message || "얼굴을 탐지하지 못했습니다."}
                  </p>
                </div>
              ) : (
                <div style={{ textAlign: 'center' }}>
                  <p style={{ fontSize: '48px', fontWeight: 'bold', color: (result.prob ?? result.analysis?.prob ?? 0) > 0.5 ? '#FF4B4B' : '#39FF14' }}>
                    {(result.label || ( (result.prob ?? result.analysis?.prob ?? 0) > 0.5 ? 'FAKE' : 'REAL' ))}
                  </p>
                  <button 
                    // [수정] image_id만 전달
                    onClick={() => navigate('/analysis-detail', { 
                      state: { image_id: result.image_id } 
                    })} 
                    style={{ color: '#39FF14', background: 'none', border: 'none', textDecoration: 'underline', marginTop: '15px', cursor: 'pointer' }}
                  >
                    상세 결과 보기
                  </button>
                </div>
              )
            ) : <p style={{ color: '#222' }}>WAITING...</p>}
          </div>
        </div>
      </main>

      <aside style={rightPanelStyle}>
        <h3 style={{ fontSize: '22px', marginBottom: '30px', borderBottom: '1px solid #222', paddingBottom: '15px' }}>분석 설정</h3>
        <div style={{ marginBottom: '25px' }}>
          <p style={{ color: '#39FF14', fontSize: '14px', marginBottom: '10px', fontWeight: 'bold' }}>버전 선택</p>
          <div style={{ display: 'flex', backgroundColor: '#000', borderRadius: '12px', padding: '5px', border: '1px solid #333' }}>
            <button onClick={() => setVersionType('v1')} style={{ flex: 1, padding: '10px', borderRadius: '8px', border: 'none', backgroundColor: versionType === 'v1' ? '#222' : 'transparent', color: versionType === 'v1' ? '#39FF14' : '#666', cursor: 'pointer' }}>V1</button>
            <button onClick={() => setVersionType('v2')} style={{ flex: 1, padding: '10px', borderRadius: '8px', border: 'none', backgroundColor: versionType === 'v2' ? '#222' : 'transparent', color: versionType === 'v2' ? '#39FF14' : '#666', cursor: 'pointer' }}>V2</button>
          </div>
        </div>
        <div style={{ marginBottom: '25px' }}>
          <p style={{ color: '#39FF14', fontSize: '14px', marginBottom: '10px', fontWeight: 'bold' }}>대상 도메인</p>
          <div style={{ display: 'flex', gap: '10px' }}>
            <button onClick={() => setDomainType('western')} style={{ flex: 1, padding: '12px', borderRadius: '8px', border: 'none', backgroundColor: domainType === 'western' ? '#222' : '#000', color: domainType === 'western' ? '#39FF14' : '#666', cursor: 'pointer' }}>서양인</button>
            <button onClick={() => setDomainType('asian')} disabled={versionType === 'v1'} style={{ flex: 1, padding: '12px', borderRadius: '8px', border: 'none', backgroundColor: domainType === 'asian' ? '#222' : '#000', color: domainType === 'asian' ? '#39FF14' : '#666', cursor: versionType === 'v1' ? 'not-allowed' : 'pointer' }}>동양인</button>
          </div>
        </div>
        <div style={{ marginBottom: '30px' }}>
          <p style={{ color: '#39FF14', fontSize: '14px', marginBottom: '15px', fontWeight: 'bold' }}>모델 선택</p>
          <label style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px', cursor: 'pointer' }}>
            <input type="radio" checked={modelType === 'fast'} onChange={() => setModelType('fast')} style={{ accentColor: '#39FF14' }} />
            <span>FAST</span>
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer' }}>
            <input type="radio" checked={modelType === 'pro'} onChange={() => setModelType('pro')} style={{ accentColor: '#39FF14' }} />
            <span>PRO</span>
          </label>
        </div>
        <button onClick={handlePredict} disabled={isAnalyzing} style={{ marginTop: 'auto', padding: '18px', backgroundColor: isAnalyzing ? '#222' : '#1A2C50', color: 'white', border: 'none', borderRadius: '12px', fontWeight: 'bold', fontSize: '16px', cursor: isAnalyzing ? 'not-allowed' : 'pointer' }}>
          {isAnalyzing ? "분석 중..." : "분석 시작"}
        </button>
      </aside>
    </div>
  );
};

export default AnalysisPage;
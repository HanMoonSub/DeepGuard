import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

axios.defaults.withCredentials = true;

const VideoAnalysisPage = ({ sessionUser, onLogout, setSessionUser }) => {
  const navigate = useNavigate();
  
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [history, setHistory] = useState([]);
  const [showOptions, setShowOptions] = useState(false);
  const [isEditMode, setIsEditMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState([]);

  const [versionType, setVersionType] = useState('v1'); 
  const [modelType, setModelType] = useState('fast');   
  const [domainType, setDomainType] = useState('western'); 
  const [result, setResult] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');

  const pollingTimer = useRef(null);
  const isMounted = useRef(true);

  const fetchHistory = useCallback(async () => {
    if (!sessionUser) return;
    try {
      const response = await axios.get('http://localhost:8000/video/history');
      if (response.data.status === "success") {
        setHistory(response.data.context || []);
      }
    } catch (e) {
      console.log("히스토리 로드 실패:", e);
    }
  }, [sessionUser]);

  useEffect(() => {
    isMounted.current = true;
    const syncSession = async () => {
      if (!sessionUser) {
        try {
          const res = await axios.get('http://localhost:8000/home');
          if (res.data?.session_user) setSessionUser(res.data.session_user);
        } catch (error) { console.log("세션 확인 실패"); }
      }
    };
    syncSession();
    fetchHistory();
    return () => {
      isMounted.current = false;
      if (pollingTimer.current) { clearInterval(pollingTimer.current); pollingTimer.current = null; }
    };
  }, [sessionUser, setSessionUser, fetchHistory]);

  /*삭제 로직 (백엔드 연동 포함)*/
  const handleDeleteSelected = async () => {
    if (selectedIds.length === 0) return;
    if (!window.confirm(`${selectedIds.length}개의 기록을 삭제하시겠습니까?`)) return;

    /* -----------------------------------------------------------------
       try {
         await axios.post('http://localhost:8000/video/delete', { ids: selectedIds });
         alert("서버에서 삭제되었습니다.");
         fetchHistory();
       } catch (error) {
         alert("삭제 실패"); return;
       }
       ----------------------------------------------------------------- */

    const updatedHistory = history.filter(item => !selectedIds.includes(item.id));
    setHistory(updatedHistory);
    setSelectedIds([]);
    setIsEditMode(false);
    alert("화면에서 제거되었습니다.");
  };

  const toggleSelect = (id) => {
    setSelectedIds(prev => 
      prev.includes(id) ? prev.filter(item => item !== id) : [...prev, id]
    );
  };

  const startPolling = useCallback((videoId) => {
    if (!videoId) return;
    if (pollingTimer.current) clearInterval(pollingTimer.current);
    pollingTimer.current = setInterval(async () => {
      if (!isMounted.current) return;
      try {
        const response = await axios.get(`http://localhost:8000/inference/video/result/${videoId}`);
        const data = response.data;
        if (data.status === 'SUCCESS') {
          clearInterval(pollingTimer.current); pollingTimer.current = null;
          setResult(data); setIsAnalyzing(false); setStatusMessage(''); fetchHistory();
        } else if (data.status === 'WARNING' || data.status === 'FAILED') {
          clearInterval(pollingTimer.current); pollingTimer.current = null;
          setIsAnalyzing(false); setStatusMessage(''); alert(data.result_msg || data.message || "오류");
        } else { setStatusMessage(data.message || "AI 추론 중..."); }
      } catch (err) { setStatusMessage("서버 연결 확인 중..."); }
    }, 2500);
  }, [fetchHistory]);

  const handlePredict = async () => {
    if (modelType === 'pro' && !sessionUser) { alert("로그인이 필요합니다."); navigate('/login'); return; }
    if (!file) return alert("영상을 업로드해주세요.");
    
    const formData = new FormData();
    formData.append('videofile', file);
    formData.append('model_type', modelType);
    formData.append('version_type', versionType);
    formData.append('domain_type', domainType === 'western' ? '서양인' : '동양인');

    setIsAnalyzing(true); setStatusMessage('서버 접수 중...'); setResult(null);
    try {
      const response = await axios.post('http://localhost:8000/inference/video', formData);
      const videoId = response.data?.video_id;
      if (videoId) startPolling(videoId); else { alert("ID 실패"); setIsAnalyzing(false); }
    } catch (err) { setIsAnalyzing(false); setStatusMessage(''); }
  };

  // 설정 변경 핸들러
  const handleVersionChange = (v) => {
    setVersionType(v);
    if (v === 'v1') setDomainType('western');
  };

  const handleModelChange = (type) => {
    if (type === 'pro' && !sessionUser) {
      alert("PRO 모델 분석은 로그인이 필요합니다.");
      navigate('/login'); 
      return;
    }
    setModelType(type);
  };

  const handleFileSelect = (selectedFile) => {
    if (!selectedFile) return;
    setFile(selectedFile); setPreviewUrl(URL.createObjectURL(selectedFile));
    setResult(null); setStatusMessage(''); setShowOptions(false);
  };

  const handleLogoutClick = async () => {
    if (!window.confirm("로그아웃 하시겠습니까?")) return;
    try { await axios.get('http://localhost:8000/auth/logout'); onLogout(); navigate('/main'); } catch (error) {}
  };

  const sideBarStyle = { width: '280px', backgroundColor: '#050505', borderRight: '1px solid #222', display: 'flex', flexDirection: 'column', padding: '25px' };
  const centerZoneStyle = { flex: 1, padding: '40px', display: 'flex', flexDirection: 'column', gap: '20px', overflowY: 'auto' };
  const rightPanelStyle = { width: '340px', backgroundColor: '#0D0D0D', borderLeft: '1px solid #222', padding: '30px', display: 'flex', flexDirection: 'column' };
  const innerBoxStyle = { flex: 1, border: '2px dashed #333', borderRadius: '20px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', backgroundColor: '#0a0a0a', cursor: 'pointer', position: 'relative', transition: 'all 0.3s' };
  const plusBtnStyle = { width: '60px', height: '60px', borderRadius: '50%', backgroundColor: '#1A2C50', display: 'flex', justifyContent: 'center', alignItems: 'center', fontSize: '30px', color: '#39FF14', marginBottom: '20px', boxShadow: '0 0 15px rgba(57, 255, 20, 0.2)' };

  return (
    <div style={{ display: 'flex', backgroundColor: '#000', height: '100vh', width: '100vw', color: 'white', fontFamily: 'sans-serif', overflow: 'hidden' }}>
      <aside style={sideBarStyle}>
        <button onClick={() => { setFile(null); setPreviewUrl(null); setResult(null); setStatusMessage(''); setShowOptions(false); if (pollingTimer.current) clearInterval(pollingTimer.current); }} style={{ backgroundColor: '#1A2C50', color: 'white', padding: '14px', borderRadius: '10px', border: 'none', marginBottom: '35px', cursor: 'pointer', fontWeight: 'bold' }}>+ 새 영상 프로젝트</button>
        
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <h3 style={{ fontSize: '18px', margin: 0 }}>내 작업 기록</h3>
          {sessionUser && history.length > 0 && (
            <button onClick={() => { setIsEditMode(!isEditMode); setSelectedIds([]); }} style={{ background: 'none', border: 'none', color: '#39FF14', cursor: 'pointer', fontSize: '12px' }}>{isEditMode ? '취소' : '편집'}</button>
          )}
        </div>

        <div style={{ flex: 1, overflowY: 'auto' }}>
          {sessionUser ? (
            history.map((item, index) => {
              const score = item.score ?? item.prob ?? item.result_prob ?? -1;
              const vType = item.version_type ? item.version_type.toUpperCase() : 'V1';
              const dType = item.domain_type || '서양인';
              const mType = item.model_type ? item.model_type.toUpperCase() : 'FAST';
              return (
                <div key={item.id || index} style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '15px' }}>
                  {isEditMode && <input type="checkbox" checked={selectedIds.includes(item.id)} onChange={() => toggleSelect(item.id)} style={{ accentColor: '#39FF14', width: '18px', height: '18px', cursor: 'pointer' }} />}
                  <div onClick={() => !isEditMode && navigate('/analysis-detail', { state: { ...item, prob: score, video_loc: item.video_loc } })} style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '15px', padding: '12px', backgroundColor: '#111', borderRadius: '12px', border: selectedIds.includes(item.id) ? '1px solid #39FF14' : '1px solid #222', cursor: isEditMode ? 'default' : 'pointer' }}>
                    <div style={{ width: '45px', height: '45px', backgroundColor: '#222', borderRadius: '8px', display: 'flex', justifyContent: 'center', alignItems: 'center', overflow: 'hidden' }}><span style={{ fontSize: '18px' }}>🎥</span></div>
                    <div><div style={{ fontSize: '12px', color: '#aaa', fontWeight: 'bold', marginBottom: '2px' }}>{vType} | {dType} | {mType}</div><div style={{ fontSize: '11px', color: '#555' }}>{item.created_at}</div><div style={{ fontSize: '13px', color: '#fff', fontWeight: '500' }}>{item.label}</div></div>
                  </div>
                </div>
              );
            })
          ) : ( <div style={{ color: '#444', fontSize: '14px', textAlign: 'center', marginTop: '60px' }}>로그인 후 이용 가능합니다.</div> )}
        </div>

        {isEditMode && (
          <button onClick={handleDeleteSelected} disabled={selectedIds.length === 0} style={{ width: '100%', padding: '12px', backgroundColor: selectedIds.length > 0 ? '#FF4B4B' : '#222', color: '#fff', border: 'none', borderRadius: '8px', cursor: selectedIds.length > 0 ? 'pointer' : 'not-allowed', fontWeight: 'bold', marginTop: '10px' }}>{selectedIds.length}개 삭제하기</button>
        )}

        <div style={{ borderTop: '1px solid #222', paddingTop: '20px', marginTop: '20px' }}>
          <button onClick={() => navigate('/main')} style={{ width: '100%', padding: '12px', backgroundColor: 'transparent', color: '#fff', border: '1px solid #444', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold', marginBottom: '15px' }}>메인 화면</button>
          {sessionUser && (
            <div style={{ textAlign: 'center' }}>
              <p style={{ fontSize: '13px', color: '#666', marginBottom: '10px' }}>{sessionUser.name}님 접속 중</p>
              <button onClick={handleLogoutClick} style={{ width: '100%', padding: '12px', backgroundColor: 'transparent', color: '#FF4B4B', border: '1px solid #FF4B4B', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }}>로그아웃</button>
            </div>
          )}
        </div>
      </aside>

      <main style={centerZoneStyle}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 style={{ fontSize: '28px', fontWeight: 'bold' }}>Deep Guard Video AI</h2>
          <div style={{ color: '#39FF14', fontSize: '14px' }}>● 비디오 모드 가동 중</div>
        </div>
        <div style={{ display: 'flex', flex: 1, gap: '20px' }}>
          <div style={{...innerBoxStyle, border: showOptions ? '2px solid #39FF14' : '2px dashed #333'}} onClick={() => { if(!previewUrl) setShowOptions(!showOptions); }} onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); handleFileSelect(e.dataTransfer.files[0]); }}>
            {previewUrl ? <video src={previewUrl} style={{ maxWidth: '95%', maxHeight: '95%' }} controls /> : 
              <div style={{textAlign:'center', display: 'flex', flexDirection: 'column', alignItems: 'center'}}><div style={plusBtnStyle}>+</div><p style={{fontSize:'20px', fontWeight:'bold', marginBottom: '8px'}}>Video Upload</p><p style={{color:'#555', fontSize:'14px'}}>영상을 업로드하세요</p>
                 {showOptions && <div style={{ marginTop: '25px', display: 'flex', gap: '12px' }}><button onClick={(e) => { e.stopPropagation(); document.getElementById('vInput').click(); }} style={{ padding: '10px 18px', backgroundColor: '#222', color: '#39FF14', border: '1px solid #39FF14', borderRadius: '8px', cursor:'pointer', fontWeight:'bold' }}>내 PC 영상</button><button onClick={(e) => { e.stopPropagation(); alert("준비 중"); }} style={{ padding: '10px 18px', backgroundColor: '#222', color: '#fff', border: '1px solid #444', borderRadius: '8px', cursor:'pointer', fontWeight:'bold' }}>Cloud Drive</button></div>}
              </div>
            }
            <input id="vInput" type="file" accept="video/*" hidden onChange={(e) => handleFileSelect(e.target.files[0])} />
          </div>
          <div style={{...innerBoxStyle, cursor: 'default'}}>
            {isAnalyzing ? ( <div style={{textAlign:'center'}}><p>{statusMessage}</p></div> ) : result ? (
              <div style={{textAlign:'center'}}>
                <p style={{fontSize:'48px', fontWeight:'bold', color: result.label?.toLowerCase() === 'fake' ? '#FF4B4B' : '#39FF14'}}>{result.label?.toLowerCase() === 'fake' ? 'FAKE' : 'REAL'}</p>
                <button onClick={() => navigate('/analysis-detail', { state: { ...result, prob: result.score, video_loc: result.video_loc } })} style={{ color: '#39FF14', background:'none', border:'none', textDecoration: 'underline', marginTop: '15px', cursor:'pointer' }}>상세 결과 보기</button>
              </div>
            ) : <p style={{color:'#222'}}>WAITING...</p>}
          </div>
        </div>
      </main>

      <aside style={rightPanelStyle}>
        <h3 style={{ fontSize: '22px', marginBottom: '30px', borderBottom: '1px solid #222', paddingBottom: '15px' }}>분석 설정</h3>
        <div style={{ marginBottom: '25px' }}>
          <p style={{ color: '#39FF14', marginBottom: '10px', fontWeight: 'bold', fontSize: '14px' }}>버전 선택</p>
          <div style={{ display: 'flex', backgroundColor: '#000', borderRadius: '12px', padding: '5px', border: '1px solid #333' }}>
            <button onClick={() => handleVersionChange('v1')} style={{ flex: 1, padding: '10px', borderRadius: '8px', border: 'none', backgroundColor: versionType === 'v1' ? '#222' : 'transparent', color: versionType === 'v1' ? '#39FF14' : '#666', cursor: 'pointer' }}>V1</button>
            <button onClick={() => handleVersionChange('v2')} style={{ flex: 1, padding: '10px', borderRadius: '8px', border: 'none', backgroundColor: versionType === 'v2' ? '#222' : 'transparent', color: versionType === 'v2' ? '#39FF14' : '#666', cursor: 'pointer' }}>V2</button>
          </div>
        </div>
        <div style={{ marginBottom: '25px' }}>
          <p style={{ color: '#39FF14', marginBottom: '10px', fontWeight: 'bold', fontSize: '14px' }}>대상 도메인</p>
          <div style={{ display: 'flex', gap: '10px' }}>
            <button onClick={() => setDomainType('western')} style={{ flex: 1, padding: '12px', borderRadius: '8px', border: 'none', backgroundColor: domainType === 'western' ? '#222' : '#000', color: domainType === 'western' ? '#39FF14' : '#666', cursor: 'pointer' }}>서양인</button>
            <button disabled={versionType === 'v1'} onClick={() => setDomainType('asian')} style={{ flex: 1, padding: '12px', borderRadius: '8px', border: 'none', backgroundColor: domainType === 'asian' ? '#222' : '#000', color: domainType === 'asian' ? '#39FF14' : '#666', cursor: versionType === 'v1' ? 'not-allowed' : 'pointer', opacity: versionType === 'v1' ? 0.3 : 1 }}>동양인</button>
          </div>
        </div>
        <div style={{ marginBottom: '30px' }}>
          <p style={{ color: '#39FF14', marginBottom: '15px', fontWeight: 'bold', fontSize: '14px' }}>모델 선택</p>
          <label style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px', cursor: 'pointer' }}>
            <input type="radio" checked={modelType === 'fast'} onChange={() => handleModelChange('fast')} style={{ accentColor: '#39FF14' }} />
            <span style={{ color: modelType === 'fast' ? '#fff' : '#666' }}>FAST - 일반 분석</span>
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer' }}>
            <input type="radio" checked={modelType === 'pro'} onChange={() => handleModelChange('pro')} style={{ accentColor: '#39FF14' }} />
            <span style={{ color: modelType === 'pro' ? '#fff' : '#666' }}>PRO - 정밀 분석</span>
          </label>
        </div>
        <button onClick={handlePredict} disabled={isAnalyzing} style={{ marginTop: 'auto', padding: '18px', backgroundColor: isAnalyzing ? '#222' : '#1A2C50', color: 'white', border: 'none', borderRadius: '12px', fontWeight: 'bold', fontSize: '16px', cursor: isAnalyzing ? 'not-allowed' : 'pointer' }}>{isAnalyzing ? "분석 중..." : "분석 시작 (PREDICT)"}</button>
      </aside>
    </div>
  );
};

export default VideoAnalysisPage;
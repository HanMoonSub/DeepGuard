import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const apiUrl = process.env.REACT_APP_API_URL || "http://localhost:8000";
axios.defaults.withCredentials = true;

const AnalysisPage = ({ sessionUser, onLogout, setSessionUser }) => {
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
          setResult(data);
          setIsAnalyzing(false);
          setStatusMessage('');
          fetchHistory();
        } else if (data.status === 'FAILED' || data.status === 'ERROR') {
          clearInterval(pollingTimer.current);
          pollingTimer.current = null;
          setIsAnalyzing(false);
          setStatusMessage('');
          alert(data.result_msg || "분석 실패");
        } else {
          setStatusMessage(data.message || "이미지 분석 진행 중...");
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

      if (imageId) {
        startPolling(imageId);
      } else {
        alert("분석 ID 발급 실패");
        setIsAnalyzing(false);
      }
    } catch (err) {
      alert("서버 통신 오류");
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
    setSelectedIds(prev => 
      prev.includes(id) ? prev.filter(item => item !== id) : [...prev, id]
    );
  };

  const handleDeleteSelected = async () => {
    if (selectedIds.length === 0) return;
    if (!window.confirm(`${selectedIds.length}개의 기록을 삭제하시겠습니까?`)) return;
    setHistory(history.filter(item => !selectedIds.includes(item.id)));
    setSelectedIds([]);
    setIsEditMode(false);
    alert("삭제되었습니다.");
  };

  const handleLogoutClick = async () => {
    if (!window.confirm("로그아웃 하시겠습니까?")) return;
    try {
      await axios.get('/auth/logout');
      onLogout(); 
      navigate('/main');
    } catch (error) { alert("로그아웃 실패"); }
  };

  // 스타일 상수 (UI 일관성 유지)
  const innerBoxStyle = { flex: 1, border: '2px dashed #333', borderRadius: '20px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', backgroundColor: '#0a0a0a', cursor: 'pointer', position: 'relative', transition: 'all 0.3s' };
  const plusBtnStyle = { width: '60px', height: '60px', borderRadius: '50%', backgroundColor: '#1A2C50', display: 'flex', justifyContent: 'center', alignItems: 'center', fontSize: '30px', color: '#39FF14', marginBottom: '20px', boxShadow: '0 0 15px rgba(57, 255, 20, 0.2)' };

  return (
    <div style={{ display: 'flex', backgroundColor: '#000', height: '100vh', width: '100vw', color: 'white', fontFamily: 'sans-serif', overflow: 'hidden' }}>
      <aside style={{ width: '280px', backgroundColor: '#050505', borderRight: '1px solid #222', display: 'flex', flexDirection: 'column', padding: '25px' }}>
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
              const p = item.prob ?? item.score ?? -1;
              const isSelected = selectedIds.includes(item.id);
              return (
                <div key={item.id || index} style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '15px' }}>
                  {isEditMode && (
                    <input type="checkbox" checked={isSelected} onChange={() => toggleSelect(item.id)} style={{ accentColor: '#39FF14', width: '18px', height: '18px' }} />
                  )}
                  <div onClick={() => !isEditMode && navigate('/analysis-detail', { state: { ...item, prob: p } })} style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '15px', padding: '12px', backgroundColor: '#111', borderRadius: '12px', border: isSelected ? '1px solid #39FF14' : '1px solid #222', cursor: isEditMode ? 'default' : 'pointer' }}>
                    <div style={{ width: '45px', height: '45px', backgroundColor: '#222', borderRadius: '8px', overflow: 'hidden' }}>
                      {item.image_loc ? <img src={`${apiUrl}${item.image_loc}`} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} /> : '🖼️'}
                    </div>
                    <div>
                      <div style={{ fontSize: '11px', color: '#555' }}>{item.created_at?.split('T')[0]}</div>
                      <div style={{ fontSize: '13px', color: '#fff' }}>{item.label || (p > 0.5 ? 'FAKE' : 'REAL')}</div>
                    </div>
                  </div>
                </div>
              );
            })
          ) : <p style={{ color: '#444', textAlign: 'center' }}>로그인 필요</p>}
        </div>

        {isEditMode && (
          <button onClick={handleDeleteSelected} disabled={selectedIds.length === 0} style={{ width: '100%', padding: '12px', backgroundColor: selectedIds.length > 0 ? '#FF4B4B' : '#222', color: '#fff', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold', marginTop: '10px' }}>선택 삭제</button>
        )}

        <div style={{ borderTop: '1px solid #222', paddingTop: '20px', marginTop: '20px' }}>
          <button onClick={() => navigate('/main')} style={{ width: '100%', padding: '12px', backgroundColor: 'transparent', color: '#fff', border: '1px solid #444', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold', marginBottom: '15px' }}>메인 화면</button>
          {sessionUser ? (
            <div style={{ textAlign: 'center' }}>
              <p style={{ fontSize: '13px', color: '#666', marginBottom: '10px' }}>{sessionUser.name}님 접속 중</p>
              <button onClick={handleLogoutClick} style={{ width: '100%', padding: '12px', backgroundColor: 'transparent', color: '#FF4B4B', border: '1px solid #FF4B4B', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }}>로그아웃</button>
            </div>
          ) : (
            <button onClick={() => navigate('/login')} style={{ width: '100%', padding: '12px', backgroundColor: '#1A2C50', color: '#fff', border: 'none', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }}>로그인</button>
          )}
        </div>
      </aside>

      <main style={{ flex: 1, padding: '40px', display: 'flex', flexDirection: 'column', gap: '20px', overflowY: 'auto' }}>
        <h2 style={{ fontSize: '28px', fontWeight: 'bold' }}>Deep Guard AI</h2>
        <div style={{ display: 'flex', flex: 1, gap: '20px' }}>
          {/* 업로드 섹션: 비디오 페이지와 동일하게 수정 */}
          <div 
            style={{...innerBoxStyle, border: showOptions ? '2px solid #39FF14' : '2px dashed #333'}} 
            onClick={() => { if(!previewUrl) setShowOptions(!showOptions); }}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => { e.preventDefault(); handleFileSelect(e.dataTransfer.files[0]); }}
          >
            {previewUrl ? <img src={previewUrl} alt="preview" style={{ maxWidth: '95%', maxHeight: '95%', objectFit: 'contain' }} /> : (
               <div style={{textAlign:'center', display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
                 <div style={plusBtnStyle}>+</div>
                 <p style={{fontSize:'20px', fontWeight:'bold', marginBottom: '8px'}}>Image Upload</p>
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

          {/* 결과 섹션: 추론 중일 때 디자인 강화 */}
          <div style={{...innerBoxStyle, border: isAnalyzing ? '2px solid #39FF14' : '2px dashed #333', cursor: 'default'}}>
            {isAnalyzing ? (
              <div style={{textAlign:'center'}}>
                <div style={{ fontSize: '40px', marginBottom: '15px', animation: 'imagePulse 1.5s infinite' }}>🖼️</div>
                <p style={{ color: '#39FF14', fontWeight: 'bold' }}>{statusMessage || "분석 중..."}</p>
                <style>{`@keyframes imagePulse { 0% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.1); opacity: 0.7; } 100% { transform: scale(1); opacity: 1; } }`}</style>
              </div>
            ) : result ? (
              <div style={{ textAlign: 'center' }}>
                <p style={{ fontSize: '48px', fontWeight: 'bold', color: (result.prob ?? result.analysis?.prob ?? 0) > 0.5 ? '#FF4B4B' : '#39FF14' }}>
                  {(result.label || ( (result.prob ?? result.analysis?.prob ?? 0) > 0.5 ? 'FAKE' : 'REAL' ))}
                </p>
                <button onClick={() => navigate('/analysis-detail', { state: { ...result, ...result.analysis, image_loc: previewUrl } })} style={{ color: '#39FF14', background: 'none', border: 'none', textDecoration: 'underline', marginTop: '15px', cursor: 'pointer' }}>상세 보기</button>
              </div>
            ) : <p style={{ color: '#222' }}>WAITING...</p>}
          </div>
        </div>
      </main>

      <aside style={{ width: '340px', backgroundColor: '#0D0D0D', padding: '30px', display: 'flex', flexDirection: 'column' }}>
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
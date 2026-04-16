import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const AnalysisPage = ({ sessionUser, onLogout, setSessionUser }) => {
  const navigate = useNavigate();
  
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [history, setHistory] = useState([]);

  const [versionType, setVersionType] = useState('v1'); 
  const [modelType, setModelType] = useState('fast');   
  const [domainType, setDomainType] = useState('western'); 
  
  const [result, setResult] = useState(null); 

  // 히스토리 로드
  const fetchHistory = useCallback(async () => {
    if (!sessionUser) return;
    try {
      const response = await axios.get('http://localhost:8000/image/history');
      if (response.data.status === "success") {
        setHistory(response.data.context); 
      }
    } catch (e) {
      console.log("히스토리 로드 실패");
    }
  }, [sessionUser]);

  useEffect(() => {
    const syncSession = async () => {
      if (!sessionUser) {
        try {
          const response = await axios.get('http://localhost:8000/auth/check');
          if (response.data && response.data.user) setSessionUser(response.data.user);
        } catch (error) { console.log("세션 확인 실패"); }
      }
    };
    syncSession();
    fetchHistory();
  }, [sessionUser, setSessionUser, fetchHistory]);

  // [수정] ESLint 경고 해결: 아래 버튼들에서 이 함수들을 호출하도록 연결했습니다.
  const handleVersionChange = (v) => {
    setVersionType(v);
    if (v === 'v1') setDomainType('western');
  };

  const handleModelChange = (type) => {
    if (type === 'pro' && !sessionUser) {
      alert("PRO 모델 정밀 분석 기능은 로그인이 필요합니다.");
      navigate('/login'); 
      return;
    }
    setModelType(type);
  };

  const handleFileSelect = (selectedFile) => {
    if (!selectedFile) return;
    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setResult(null);
  };

  const handleLogoutClick = async () => {
    if (!window.confirm("로그아웃 하시겠습니까?")) return;
    try {
      await axios.get('http://localhost:8000/auth/logout');
      onLogout(); 
      alert("성공적으로 로그아웃되었습니다.");
      navigate('/main');
    } catch (error) { alert("로그아웃 실패"); }
  };

  const handlePredict = async () => {
    if (modelType === 'pro' && !sessionUser) {
      alert("PRO 모델 분석은 로그인이 필요한 기능입니다.");
      navigate('/login');
      return;
    }
    if (!file) return alert("분석할 파일을 먼저 업로드해주세요.");

    const formData = new FormData();
    formData.append('imagefile', file);
    formData.append('model_type', modelType);
    formData.append('domain_type', domainType === 'western' ? '서양인' : '동양인');
    formData.append('version_type', versionType);

    setIsAnalyzing(true);
    try {
      const response = await axios.post('http://localhost:8000/inference/image', formData);
      setResult(response.data);
      fetchHistory(); 
    } catch (err) {
      alert(err.response?.data?.detail || "분석 중 오류가 발생했습니다.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  // 스타일 정의
  const sideBarStyle = { width: '280px', backgroundColor: '#050505', borderRight: '1px solid #222', display: 'flex', flexDirection: 'column', padding: '25px' };
  const centerZoneStyle = { flex: 1, padding: '40px', display: 'flex', flexDirection: 'column', gap: '20px', overflowY: 'auto' };
  const rightPanelStyle = { width: '340px', backgroundColor: '#0D0D0D', borderLeft: '1px solid #222', padding: '30px', display: 'flex', flexDirection: 'column' };
  const innerBoxStyle = { flex: 1, border: '2px dashed #333', borderRadius: '20px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', backgroundColor: '#0a0a0a', cursor: 'pointer', position: 'relative' };

  return (
    <div style={{ display: 'flex', backgroundColor: '#000', height: '100vh', width: '100vw', color: 'white', fontFamily: 'sans-serif', overflow: 'hidden' }}>
      <aside style={sideBarStyle}>
        <button onClick={() => { setFile(null); setPreviewUrl(null); setResult(null); }} style={{ backgroundColor: '#1A2C50', color: 'white', padding: '14px', borderRadius: '10px', border: 'none', marginBottom: '35px', cursor: 'pointer', fontWeight: 'bold' }}>+ 새 프로젝트 시작</button>
        <h3 style={{ fontSize: '18px', marginBottom: '20px' }}>내 작업 기록</h3>
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {sessionUser ? (
            history.map((item) => {
              const prob = item.prob ?? item.analysis?.prob ?? -1;
              const probNum = Number(prob);
              const displayProb = (prob !== undefined && prob !== null && prob !== -1) ? (probNum * 100).toFixed(1) + '%' : 'N/A';
              const thumbUrl = item.image_loc ? `http://localhost:8000${item.image_loc}` : null;

              // 히스토리 클릭 시 보낼 데이터 규격화
              const transportState = {
                ...item,
                prob: prob,
                face_conf: item.face_conf ?? item.analysis?.face_conf ?? item.conf ?? 0,
                face_ratio: item.face_ratio ?? item.analysis?.face_ratio ?? 0,
                face_brightness: item.face_brightness ?? item.analysis?.face_brightness ?? 0,
                label: item.label || (probNum > 0.5 ? 'FAKE' : 'REAL')
              };

              return (
                <div key={item.id} onClick={() => navigate('/analysis-detail', { state: transportState })} style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '15px', padding: '12px', backgroundColor: '#111', borderRadius: '12px', border: '1px solid #222', cursor: 'pointer' }}>
                  <div style={{ width: '45px', height: '45px', backgroundColor: '#222', borderRadius: '8px', display: 'flex', justifyContent: 'center', alignItems: 'center', overflow: 'hidden' }}>
                    {thumbUrl ? <img src={thumbUrl} alt="thumb" style={{ width: '100%', height: '100%', objectFit: 'cover' }} onError={(e) => { e.target.style.display = 'none'; }} /> : <span style={{ fontSize: '18px' }}>🖼️</span>}
                  </div>
                  <div>
                    <div style={{ fontSize: '12px', color: probNum > 0.5 ? '#FF4B4B' : (prob === -1 ? '#888' : '#39FF14'), fontWeight: 'bold', marginBottom: '2px' }}>{displayProb}</div>
                    <div style={{ fontSize: '11px', color: '#555' }}>{item.created_at}</div>
                    <div style={{ fontSize: '13px', color: '#fff', fontWeight: '500' }}>{transportState.label}</div>
                  </div>
                </div>
              );
            })
          ) : (
            <div style={{ color: '#444', fontSize: '14px', textAlign: 'center', marginTop: '60px' }}>로그인하시면 작업 기록을 저장할 수 있습니다.</div>
          )}
        </div>
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
          <h2 style={{ fontSize: '28px', fontWeight: 'bold' }}>Deep Guard AI</h2>
          <div style={{ color: '#39FF14', fontSize: '14px' }}>● 시스템 가동 중</div>
        </div>
        <div style={{ display: 'flex', flex: 1, gap: '20px' }}>
          <div style={innerBoxStyle} onClick={() => document.getElementById('fileInput').click()}>
             {previewUrl ? <img src={previewUrl} alt="preview" style={{ maxWidth: '95%', maxHeight: '95%', objectFit: 'contain' }} /> : <div style={{textAlign:'center'}}><p style={{fontSize:'20px', fontWeight:'bold'}}>Drag & Drop</p><p style={{color:'#555', fontSize:'14px'}}>Media Upload</p></div>}
             <input id="fileInput" type="file" hidden onChange={(e) => handleFileSelect(e.target.files[0])} />
          </div>
          <div style={{...innerBoxStyle, cursor: 'default'}}>
             {isAnalyzing ? <div style={{textAlign:'center'}}><p>분석 중...</p></div> : result ? (
               <div style={{textAlign:'center'}}>
                 <p style={{fontSize:'32px', fontWeight:'bold', color: result.analysis.prob > 0.5 ? '#FF4B4B' : (result.analysis.prob === -1 ? '#888' : '#39FF14')}}>
                   {result.analysis.prob === -1 ? 'UNKNOWN' : (result.analysis.prob > 0.5 ? 'FAKE' : 'REAL')}
                 </p>
                 <button onClick={() => navigate('/analysis-detail', { state: { ...result.analysis, image_loc: previewUrl, label: result.analysis.prob === -1 ? 'UNKNOWN' : (result.analysis.prob > 0.5 ? 'FAKE' : 'REAL'), message: result.message, version_type: versionType, domain_type: domainType } })} style={{ color: '#39FF14', cursor: 'pointer', background: 'none', border: 'none', textDecoration: 'underline', marginTop: '10px' }}>상세 결과 보기</button>
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
            {/* handleVersionChange를 여기서 호출합니다 */}
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
          {/* handleModelChange를 여기서 호출합니다 */}
          <label style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px', cursor: 'pointer' }}>
            <input type="radio" checked={modelType === 'fast'} onChange={() => handleModelChange('fast')} style={{ accentColor: '#39FF14' }} />
            <span style={{ color: modelType === 'fast' ? '#fff' : '#666' }}>FAST - 일반 분석</span>
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer' }}>
            <input type="radio" checked={modelType === 'pro'} onChange={() => handleModelChange('pro')} style={{ accentColor: '#39FF14' }} />
            <span style={{ color: modelType === 'pro' ? '#fff' : '#666' }}>PRO - 정밀 분석</span>
          </label>
        </div>
        <button onClick={handlePredict} disabled={isAnalyzing} style={{ marginTop: 'auto', padding: '18px', backgroundColor: isAnalyzing ? '#222' : '#1A2C50', color: 'white', border: 'none', borderRadius: '12px', fontWeight: 'bold', fontSize: '16px', cursor: 'pointer' }}>{isAnalyzing ? "분석 중..." : "분석 시작 (PREDICT)"}</button>
      </aside>
    </div>
  );
};

export default AnalysisPage;
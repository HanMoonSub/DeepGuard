import React, { useState, useEffect } from 'react';
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

  // 세션 체크
  useEffect(() => {
    const syncSession = async () => {
      if (!sessionUser) {
        try {
          const response = await axios.get('http://localhost:8000/auth/check', { withCredentials: true });
          if (response.data && response.data.user) setSessionUser(response.data.user);
        } catch (error) { console.log("세션 확인 실패"); }
      }
    };
    syncSession();
  }, [sessionUser, setSessionUser]);

  // V1/V2 변경 시 도메인 제어
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

  const handlePredict = async () => {
    if (modelType === 'pro' && !sessionUser) {
      alert("PRO 모델 분석은 로그인이 필요한 기능입니다.");
      navigate('/login');
      return;
    }
    if (!file) return alert("분석할 파일을 먼저 업로드해주세요.");

    const formData = new FormData();
    formData.append('imagefile', file);
    formData.append('version_type', versionType);
    // 요구사항: 백엔드 변수 한글 매핑
    formData.append('domain_type', domainType === 'western' ? '서양인' : '동양인');
    formData.append('model_type', modelType);

    setIsAnalyzing(true);
    try {
      const response = await axios.post('http://localhost:8000/inference/image', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        withCredentials: true
      });
      
      const data = response.data; // { status, prob, message, context: { image, face_ratio, ... } }
      
      // 라벨 판정 로직
      let finalLabel = "UNKNOWN";
      if (data.status === "success") {
        finalLabel = data.prob > 0.5 ? "FAKE" : "REAL";
      }

      // 결과 상태 업데이트
      const analysisResult = {
        ...data,
        label: finalLabel,
        // 이미지 경로는 응답의 context.image 사용
        image_loc: data.context?.image || previewUrl 
      };
      setResult(analysisResult);

      // 히스토리 추가
      const newRecord = {
        id: Date.now(),
        date: new Date().toLocaleString(),
        ...analysisResult
      };
      setHistory(prev => [newRecord, ...prev]);

      if (data.status === "warning") alert(`[주의] ${data.message}`);

    } catch (err) {
      alert(err.response?.data?.detail || "오류가 발생했습니다.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleHistoryClick = (item) => {
    setResult(item);
    setPreviewUrl(item.image_loc);
  };

  const handleLogoutClick = async () => {
    if (!window.confirm("로그아웃 하시겠습니까?")) return;
    try {
      await axios.get('http://localhost:8000/auth/logout', { withCredentials: true });
      onLogout(); 
      alert("성공적으로 로그아웃되었습니다.");
      navigate('/main');
    } catch (error) { alert("로그아웃 실패"); }
  };

  // 스타일 정의
  const sideBarStyle = { width: '280px', backgroundColor: '#050505', borderRight: '1px solid #222', display: 'flex', flexDirection: 'column', padding: '25px' };
  const centerZoneStyle = { flex: 1, padding: '40px', display: 'flex', flexDirection: 'column', gap: '20px', overflowY: 'auto' };
  const rightPanelStyle = { width: '340px', backgroundColor: '#0D0D0D', borderLeft: '1px solid #222', padding: '30px', display: 'flex', flexDirection: 'column' };
  const innerBoxStyle = { flex: 1, border: '2px dashed #333', borderRadius: '20px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', backgroundColor: '#0a0a0a', cursor: 'pointer', position: 'relative', overflow: 'hidden' };

  return (
    <div style={{ display: 'flex', backgroundColor: '#000', height: '100vh', width: '100vw', color: 'white', fontFamily: 'sans-serif', overflow: 'hidden' }}>
      
      {/* LEFT SIDEBAR: 히스토리 */}
      <aside style={sideBarStyle}>
        <button onClick={() => { setFile(null); setPreviewUrl(null); setResult(null); }} style={{ backgroundColor: '#1A2C50', color: 'white', padding: '14px', borderRadius: '10px', border: 'none', marginBottom: '35px', cursor: 'pointer', fontWeight: 'bold' }}>
          + 새 프로젝트 시작
        </button>
        <h3 style={{ fontSize: '18px', marginBottom: '20px' }}>내 작업 기록</h3>
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {sessionUser ? (
            history.map((item) => (
              <div key={item.id} onClick={() => handleHistoryClick(item)} style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '15px', padding: '12px', backgroundColor: '#111', borderRadius: '12px', border: '1px solid #222', cursor: 'pointer' }}>
                <img src={item.image_loc} alt="thumb" style={{ width: '45px', height: '45px', borderRadius: '8px', objectFit: 'cover' }} />
                <div>
                  <div style={{ fontSize: '12px', color: '#888' }}>{item.date}</div>
                  <div style={{ fontSize: '14px', color: item.label === 'FAKE' ? '#FF4B4B' : item.label === 'REAL' ? '#39FF14' : '#666', fontWeight: 'bold' }}>
                    {item.label}
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div style={{ color: '#444', fontSize: '14px', textAlign: 'center', marginTop: '60px' }}>로그인하시면 작업 기록을 저장할 수 있습니다.</div>
          )}
        </div>
        {/* 하단 버튼부 */}
        <div style={{ borderTop: '1px solid #222', paddingTop: '20px', marginTop: '20px' }}>
          <button onClick={() => navigate('/main')} style={{ width: '100%', padding: '12px', backgroundColor: 'transparent', color: '#fff', border: '1px solid #444', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold', marginBottom: '15px' }}>메인 화면으로 돌아가기</button>
          {sessionUser ? (
            <div style={{ textAlign: 'center' }}>
              <p style={{ fontSize: '13px', color: '#666', marginBottom: '10px' }}>{sessionUser.name}님 접속 중</p>
              <button onClick={handleLogoutClick} style={{ width: '100%', padding: '12px', backgroundColor: 'transparent', color: '#FF4B4B', border: '1px solid #FF4B4B', borderRadius: '8px', cursor: 'pointer', fontWeight: 'bold' }}>로그아웃</button>
            </div>
          ) : (
            <button onClick={() => navigate('/login')} style={{ width: '100%', padding: '12px', backgroundColor: '#39FF14', color: '#000', border: 'none', borderRadius: '8px', fontWeight: 'bold', cursor: 'pointer' }}>로그인하기</button>
          )}
        </div>
      </aside>

      {/* CENTER: 분석 영역 */}
      <main style={centerZoneStyle}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 style={{ fontSize: '28px', fontWeight: 'bold' }}>Deep Guard AI</h2>
          <div style={{ color: '#39FF14', fontSize: '14px' }}>● 시스템 가동 중</div>
        </div>
        
        <div style={{ display: 'flex', flex: 1, gap: '20px' }}>
          {/* 업로드 박스 */}
          <div style={innerBoxStyle} onClick={() => document.getElementById('fileInput').click()}>
             {previewUrl ? <img src={previewUrl} alt="preview" style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }} /> : <div style={{textAlign:'center'}}><p style={{fontSize:'20px', fontWeight:'bold'}}>Drag & Drop</p><p style={{color:'#555', fontSize:'14px'}}>Media Upload</p></div>}
             <input id="fileInput" type="file" hidden onChange={(e) => handleFileSelect(e.target.files[0])} />
          </div>
          {/* 결과 박스 (매트릭 포함) */}
          <div style={{...innerBoxStyle, cursor: 'default', padding: '20px'}}>
             {isAnalyzing ? (
               <p style={{color: '#39FF14'}}>분석 진행 중...</p>
             ) : result ? (
               <div style={{ width: '100%', textAlign: 'center' }}>
                 <p style={{ fontSize: '14px', color: '#888' }}>{result.version_type?.toUpperCase()} / {result.model_type?.toUpperCase()} / {result.domain_type}</p>
                 <h1 style={{ fontSize: '48px', color: result.label === 'FAKE' ? '#FF4B4B' : result.label === 'REAL' ? '#39FF14' : '#888', margin: '10px 0' }}>{result.label}</h1>
                 
                 {result.status === "success" && (
                    <div style={{ marginTop: '30px', display: 'flex', flexDirection: 'column', gap: '15px', alignItems: 'center' }}>
                      <div style={{ width: '80%', height: '8px', backgroundColor: '#222', borderRadius: '4px' }}>
                        <div style={{ width: `${result.prob * 100}%`, height: '100%', backgroundColor: result.prob > 0.5 ? '#FF4B4B' : '#39FF14', borderRadius: '4px' }} />
                      </div>
                      <p style={{ fontSize: '18px' }}>신뢰도: {(result.prob * 100).toFixed(1)}%</p>
                      
                      {/* 매트릭 상세 */}
                      <div style={{ display: 'flex', gap: '20px', marginTop: '20px' }}>
                        <div style={{ padding: '10px', border: '1px solid #333', borderRadius: '10px' }}>
                          <p style={{ fontSize: '12px', color: '#888' }}>Face Ratio</p>
                          <p style={{ color: result.context?.face_ratio > 3 ? '#39FF14' : '#FF4B4B', fontWeight: 'bold' }}>{result.context?.face_ratio}%</p>
                        </div>
                        <div style={{ padding: '10px', border: '1px solid #333', borderRadius: '10px' }}>
                          <p style={{ fontSize: '12px', color: '#888' }}>Brightness</p>
                          <p style={{ color: result.context?.face_brightness > 20 ? '#39FF14' : '#FF4B4B', fontWeight: 'bold' }}>{result.context?.face_brightness}%</p>
                        </div>
                      </div>
                    </div>
                 )}
                 {result.status === "warning" && <p style={{ color: '#FFBD39', marginTop: '20px' }}>{result.message}</p>}
               </div>
             ) : <p style={{color:'#222'}}>WAITING FOR ANALYSIS...</p>}
          </div>
        </div>
      </main>

      {/* RIGHT: 분석 설정 */}
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

        <button onClick={handlePredict} disabled={isAnalyzing} style={{ marginTop: 'auto', padding: '18px', backgroundColor: isAnalyzing ? '#222' : '#1A2C50', color: 'white', border: 'none', borderRadius: '12px', fontWeight: 'bold', fontSize: '16px', cursor: 'pointer' }}>
          {isAnalyzing ? "분석 중..." : "분석 시작 (PREDICT)"}
        </button>
      </aside>
    </div>
  );
};

export default AnalysisPage;
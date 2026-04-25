import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const VideoAnalysisPage = ({ sessionUser, onLogout, setSessionUser }) => {
  const navigate = useNavigate();
  
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [history, setHistory] = useState([]);

  const [versionType, setVersionType] = useState('v1'); 
  const [modelType, setModelType] = useState('fast');   
  const [domainType, setDomainType] = useState('western'); 
  
  const [result, setResult] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');

  const pollingTimer = useRef(null);
  const isMounted = useRef(true);

  // 히스토리 로드
  const fetchHistory = useCallback(async () => {
    if (!sessionUser) return;
    try {
      const response = await axios.get('/video/history');
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
          const res = await axios.get('/auth/check');
          if (res.data?.user) setSessionUser(res.data.user);
        } catch (error) {
          console.log("세션 확인 실패");
        }
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

  // 폴링 함수 (백엔드 스펙에 맞춤)
  const startPolling = useCallback((videoId) => {
    if (!videoId) return;

    if (pollingTimer.current) clearInterval(pollingTimer.current);

    console.log(`[Video Polling Start] videoId = ${videoId}`);

    pollingTimer.current = setInterval(async () => {
      if (!isMounted.current) return;

      try {
        const response = await axios.get(`/inference/video/result/${videoId}`);
        const data = response.data;

        console.log('[VIDEO RESULT]', data);

        if (data.status === 'SUCCESS') {
          clearInterval(pollingTimer.current);
          pollingTimer.current = null;
          setResult(data);
          setIsAnalyzing(false);
          setStatusMessage('');
          fetchHistory();
        } 
        else if (data.status === 'WARNING' || data.status === 'FAILED') {
          clearInterval(pollingTimer.current);
          pollingTimer.current = null;
          setIsAnalyzing(false);
          setStatusMessage('');
          alert(data.result_msg || data.message || "영상 분석 중 오류가 발생했습니다.");
        } 
        else {
          setStatusMessage(data.message || "AI 추론 중...");
        }
      } catch (err) {
        console.error("폴링 에러:", err);
        setStatusMessage("서버 연결 확인 중...");
      }
    }, 2500);
  }, [fetchHistory]);

  // 분석 시작 함수
  const handlePredict = async () => {
    if (modelType === 'pro' && !sessionUser) {
      alert("PRO 모델 분석은 로그인이 필요합니다.");
      navigate('/login');
      return;
    }
    if (!file) return alert("분석할 영상을 먼저 업로드해주세요.");

    const formData = new FormData();
    formData.append('videofile', file);
    formData.append('model_type', modelType);
    formData.append('version_type', versionType);
    formData.append('domain_type', domainType === 'western' ? '서양인' : '동양인');   // 백엔드와 정확히 일치

    setIsAnalyzing(true);
    setStatusMessage('서버 접수 중...');
    setResult(null);

    try {
      const response = await axios.post('/inference/video', formData);

      console.log('[VIDEO POST RESPONSE]', response.status, response.data);

      const videoId = response.data?.video_id;

      if (videoId) {
        startPolling(videoId);
      } else {
        alert("video_id를 받지 못했습니다.");
        setIsAnalyzing(false);
      }
    } catch (err) {
      console.error("분석 요청 에러:", err);
      const errorMsg = err.response?.data?.detail || 
                      err.response?.data?.message || 
                      "분석 요청에 실패했습니다.";
      alert(errorMsg);
      setIsAnalyzing(false);
      setStatusMessage('');
    }
  };

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
    setFile(selectedFile);
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setResult(null);
    setStatusMessage('');
  };

  const handleLogoutClick = async () => {
    if (!window.confirm("로그아웃 하시겠습니까?")) return;
    try {
      await axios.get('/auth/logout');
      onLogout(); 
      alert("성공적으로 로그아웃되었습니다.");
      navigate('/main');
    } catch (error) { 
      alert("로그아웃 실패"); 
    }
  };

  const sideBarStyle = { width: '280px', backgroundColor: '#050505', borderRight: '1px solid #222', display: 'flex', flexDirection: 'column', padding: '25px' };
  const centerZoneStyle = { flex: 1, padding: '40px', display: 'flex', flexDirection: 'column', gap: '20px', overflowY: 'auto' };
  const rightPanelStyle = { width: '340px', backgroundColor: '#0D0D0D', borderLeft: '1px solid #222', padding: '30px', display: 'flex', flexDirection: 'column' };
  const innerBoxStyle = { flex: 1, border: '2px dashed #333', borderRadius: '20px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', backgroundColor: '#0a0a0a', cursor: 'pointer', position: 'relative' };

  return (
    <div style={{ display: 'flex', backgroundColor: '#000', height: '100vh', width: '100vw', color: 'white', fontFamily: 'sans-serif', overflow: 'hidden' }}>
      
      {/* 왼쪽 히스토리바 */}
      <aside style={sideBarStyle}>
        <button 
          onClick={() => { 
            setFile(null); 
            setPreviewUrl(null); 
            setResult(null); 
            setStatusMessage(''); 
            if (pollingTimer.current) clearInterval(pollingTimer.current);
          }} 
          style={{ backgroundColor: '#1A2C50', color: 'white', padding: '14px', borderRadius: '10px', border: 'none', marginBottom: '35px', cursor: 'pointer', fontWeight: 'bold' }}
        >
          + 새 영상 프로젝트
        </button>

        <h3 style={{ fontSize: '18px', marginBottom: '20px' }}>내 작업 기록</h3>
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {sessionUser ? (
            history.map((item) => {
              const score = item.score ?? -1;
              const displayProb = score !== -1 ? (Number(score) * 100).toFixed(1) + '%' : 'N/A';

              return (
                <div 
                  key={item.id} 
                  onClick={() => navigate('/analysis-detail', { state: { ...item, prob: score } })} 
                  style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '15px', padding: '12px', backgroundColor: '#111', borderRadius: '12px', border: '1px solid #222', cursor: 'pointer' }}
                >
                  <div style={{ width: '45px', height: '45px', backgroundColor: '#222', borderRadius: '8px', display: 'flex', justifyContent: 'center', alignItems: 'center', overflow: 'hidden' }}>
                    <span style={{ fontSize: '18px' }}>🎥</span>
                  </div>
                  <div>
                    <div style={{ fontSize: '12px', color: Number(score) > 0.5 ? '#FF4B4B' : '#39FF14', fontWeight: 'bold', marginBottom: '2px' }}>{displayProb}</div>
                    <div style={{ fontSize: '11px', color: '#555' }}>{item.created_at}</div>
                    <div style={{ fontSize: '13px', color: '#fff', fontWeight: '500' }}>{item.label}</div>
                  </div>
                </div>
              );
            })
          ) : (
            <div style={{ color: '#444', fontSize: '14px', textAlign: 'center', marginTop: '60px' }}>로그인 후 이용 가능합니다.</div>
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

      {/* 중앙 분석 영역 */}
      <main style={centerZoneStyle}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 style={{ fontSize: '28px', fontWeight: 'bold' }}>Deep Guard Video AI</h2>
          <div style={{ color: '#39FF14', fontSize: '14px' }}>● 비디오 모드 가동 중</div>
        </div>
        <div style={{ display: 'flex', flex: 1, gap: '20px' }}>
          <div style={innerBoxStyle} onClick={() => document.getElementById('vInput').click()}>
            {previewUrl ? <video src={previewUrl} style={{ maxWidth: '95%', maxHeight: '95%' }} controls /> : 
              <div style={{textAlign:'center'}}>
                <p style={{fontSize:'20px', fontWeight:'bold'}}>Video Upload</p>
                <p style={{color:'#555', fontSize:'14px'}}>영상을 업로드하세요</p>
              </div>
            }
            <input id="vInput" type="file" accept="video/*" hidden onChange={(e) => handleFileSelect(e.target.files[0])} />
          </div>

          <div style={{...innerBoxStyle, cursor: 'default'}}>
            {isAnalyzing ? (
              <div style={{textAlign:'center'}}>
                <p style={{fontSize:'20px', fontWeight:'bold'}}>{statusMessage}</p>
              </div>
            ) : result ? (
              <div style={{textAlign:'center'}}>
                <p style={{fontSize:'48px', fontWeight:'bold', color: result.label?.toLowerCase() === 'fake' ? '#FF4B4B' : '#39FF14'}}>
                  {result.label?.toLowerCase() === 'fake' ? 'FAKE' : 'REAL'}
                </p>
                <button 
                  onClick={() => navigate('/analysis-detail', { state: { ...result, prob: result.score } })} 
                  style={{ color: '#39FF14', cursor: 'pointer', background: 'none', border: 'none', textDecoration: 'underline', marginTop: '15px' }}
                >
                  상세 결과 보기
                </button>
              </div>
            ) : <p style={{color:'#222'}}>WAITING...</p>}
          </div>
        </div>
      </main>

      {/* 오른쪽 설정 패널 */}
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

        <button 
          onClick={handlePredict} 
          disabled={isAnalyzing} 
          style={{ 
            marginTop: 'auto', 
            padding: '18px', 
            backgroundColor: isAnalyzing ? '#222' : '#1A2C50', 
            color: 'white', 
            border: 'none', 
            borderRadius: '12px', 
            fontWeight: 'bold', 
            fontSize: '16px', 
            cursor: isAnalyzing ? 'not-allowed' : 'pointer' 
          }}
        >
          {isAnalyzing ? "분석 중..." : "분석 시작 (PREDICT)"}
        </button>
      </aside>
    </div>
  );
};

export default VideoAnalysisPage;
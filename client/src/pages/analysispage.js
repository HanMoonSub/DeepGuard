import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const AnalysisPage = ({ sessionUser, onLogout, setSessionUser }) => {
  const navigate = useNavigate();
  
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [modelType, setModelType] = useState('FAST');
  const [domainType, setDomainType] = useState('ASIAN'); 
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const syncSession = async () => {
      if (!sessionUser) {
        try {
          const response = await axios.get('http://localhost:8000/auth/check', { withCredentials: true });
          if (response.data && response.data.user) {
            setSessionUser(response.data.user);
          }
        } catch (error) {
          console.log("세션 확인 실패");
        }
      }
    };
    syncSession();
  }, [sessionUser, setSessionUser]);

  useEffect(() => {
    if (sessionUser) {
      setHistory([
        { id: 1, type: '이미지', date: '2026-04-07 10:30 오전', thumbnail: null },
        { id: 2, type: '동영상', date: '2026-04-06 02:15 오후', thumbnail: null },
      ]);
    } else {
      setHistory([]);
    }
  }, [sessionUser]);

  const handleModelChange = (type) => {
    if (type === 'PRO' && !sessionUser) {
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

  const handleDragOver = (e) => { e.preventDefault(); e.stopPropagation(); };
  const handleDrop = (e) => {
    e.preventDefault(); e.stopPropagation();
    const files = e.dataTransfer.files;
    if (files && files.length > 0) handleFileSelect(files[0]);
  };

  const handleLogoutClick = async () => {
    if (!window.confirm("로그아웃 하시겠습니까?")) return;
    try {
      await axios.get('http://localhost:8000/auth/logout', { withCredentials: true });
      onLogout(); 
      alert("성공적으로 로그아웃되었습니다.");
      navigate('/main');
    } catch (error) {
      alert("로그아웃 실패");
    }
  };

  const handlePredict = async () => {
    if (modelType === 'PRO' && !sessionUser) {
      alert("PRO 모델 분석은 로그인이 필요한 기능입니다.");
      navigate('/login');
      return;
    }
    if (!file) return alert("분석할 파일을 먼저 업로드해주세요.");

    setIsAnalyzing(true);
    setTimeout(() => {
      const isFake = Math.random() > 0.5;
      setResult({ is_deepfake: isFake, confidence: (Math.random() * (99 - 85) + 85).toFixed(1) });
      setIsAnalyzing(false);

      if (sessionUser) {
        const newHistoryItem = {
          id: Date.now(),
          type: file.type.includes('video') ? '동영상' : '이미지',
          date: '방금 전',
          thumbnail: previewUrl
        };
        setHistory(prev => [newHistoryItem, ...prev]);
      }
    }, 2000);
  };

  const sideBarStyle = { width: '280px', backgroundColor: '#050505', borderRight: '1px solid #222', display: 'flex', flexDirection: 'column', padding: '25px' };
  const centerZoneStyle = { flex: 2, padding: '40px', display: 'flex', flexDirection: 'column', overflowY: 'auto' };
  const rightPanelStyle = { width: '340px', backgroundColor: '#0D0D0D', borderLeft: '1px solid #222', padding: '30px', display: 'flex', flexDirection: 'column' };

  return (
    <div style={{ display: 'flex', backgroundColor: '#000', height: '100vh', width: '100vw', color: 'white', fontFamily: 'sans-serif', overflow: 'hidden' }}>
      
      <aside style={sideBarStyle}>
        <button onClick={() => { setFile(null); setPreviewUrl(null); setResult(null); }} style={{ backgroundColor: '#1A2C50', color: 'white', padding: '14px', borderRadius: '10px', border: 'none', marginBottom: '35px', cursor: 'pointer', fontWeight: 'bold' }}>
          + 새 프로젝트 시작
        </button>
        <h3 style={{ fontSize: '18px', marginBottom: '20px' }}>내 작업 기록</h3>
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {sessionUser ? (
            history.map((item) => (
              <div key={item.id} style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '15px', padding: '12px', backgroundColor: '#111', borderRadius: '12px', border: '1px solid #222' }}>
                <div style={{ width: '45px', height: '45px', backgroundColor: '#222', borderRadius: '8px', overflow: 'hidden', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                  {item.thumbnail ? <img src={item.thumbnail} alt="thumb" style={{ width: '100%', height: '100%', objectFit: 'cover' }} /> : <span style={{ fontSize: '20px' }}>🖼️</span>}
                </div>
                <div>
                  <div style={{ fontSize: '12px', color: '#888' }}>{item.date}</div>
                  <div style={{ fontSize: '14px', color: '#39FF14', fontWeight: 'bold' }}>분석 완료</div>
                </div>
              </div>
            ))
          ) : (
            <div style={{ color: '#444', fontSize: '14px', textAlign: 'center', marginTop: '60px' }}>로그인하시면 작업 기록을 저장할 수 있습니다.</div>
          )}
        </div>
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

      <main style={centerZoneStyle}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px' }}>
          <h2 style={{ fontSize: '28px', fontWeight: 'bold' }}>Deep Guard AI</h2>
          <div style={{ color: '#39FF14', fontSize: '14px' }}>● 시스템 가동 중</div>
        </div>
        <div 
          style={{ flex: 1, border: '2px dashed #333', borderRadius: '20px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', backgroundColor: '#0a0a0a', cursor: 'pointer', position: 'relative' }}
          onClick={() => document.getElementById('fileInput').click()}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
          {isAnalyzing ? (
            <div style={{ textAlign: 'center' }}>⚙️ 분석 중...</div>
          ) : previewUrl ? (
            <img src={previewUrl} alt="preview" style={{ maxWidth: '95%', maxHeight: '95%', borderRadius: '12px', objectFit: 'contain' }} />
          ) : (
            <div style={{ textAlign: 'center' }}>
              <p style={{ fontSize: '20px', fontWeight: 'bold' }}>Drag & Drop</p>
              <p style={{ color: '#555', marginTop: '10px' }}>or Click to Upload Your Image</p>
            </div>
          )}
          <input id="fileInput" type="file" hidden onChange={(e) => handleFileSelect(e.target.files[0])} />
        </div>
        <div style={{ marginTop: '30px', padding: '25px', backgroundColor: '#111', borderRadius: '15px', border: '1px solid #222', textAlign: 'center' }}>
          {result ? (
            <div>결과: <span style={{ color: result.is_deepfake ? '#FF4B4B' : '#39FF14', fontWeight: 'bold' }}>{result.is_deepfake ? 'FAKE' : 'REAL'}</span> ({result.confidence}%)</div>
          ) : (
            <p style={{ color: '#555' }}>분석 버튼을 눌러 결과를 확인하세요.</p>
          )}
        </div>
      </main>

      <aside style={rightPanelStyle}>
        <h3 style={{ fontSize: '22px', marginBottom: '40px', borderBottom: '1px solid #222', paddingBottom: '15px' }}>분석 설정</h3>
        <div style={{ marginBottom: '35px' }}>
          <p style={{ color: '#39FF14', marginBottom: '15px', fontWeight: 'bold' }}>대상 도메인</p>
          <div style={{ display: 'flex', backgroundColor: '#000', borderRadius: '12px', padding: '6px', border: '1px solid #333' }}>
            <button onClick={() => setDomainType('WESTERN')} style={{ flex: 1, padding: '10px', borderRadius: '8px', border: 'none', backgroundColor: domainType === 'WESTERN' ? '#222' : 'transparent', color: domainType === 'WESTERN' ? '#39FF14' : '#666', cursor: 'pointer' }}>서양인</button>
            <button onClick={() => setDomainType('ASIAN')} style={{ flex: 1, padding: '10px', borderRadius: '8px', border: 'none', backgroundColor: domainType === 'ASIAN' ? '#222' : 'transparent', color: domainType === 'ASIAN' ? '#39FF14' : '#666', cursor: 'pointer' }}>동양인</button>
          </div>
        </div>
        <div style={{ marginBottom: '45px' }}>
          <p style={{ color: '#39FF14', marginBottom: '20px', fontWeight: 'bold' }}>모델 선택</p>
          
          <label style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '15px', cursor: 'pointer' }}>
            <input 
              type="radio" 
              checked={modelType === 'FAST'} 
              onChange={() => handleModelChange('FAST')} 
              style={{ accentColor: '#39FF14' }} 
            />
            <span style={{ color: modelType === 'FAST' ? '#fff' : '#666' }}>FAST - 일반 분석</span>
          </label>

          <label style={{ display: 'flex', alignItems: 'center', gap: '12px', cursor: 'pointer' }}>
            <input 
              type="radio" 
              checked={modelType === 'PRO'} 
              onChange={() => handleModelChange('PRO')} 
              style={{ accentColor: '#39FF14' }} 
            />
            <span style={{ color: modelType === 'PRO' ? '#fff' : '#666' }}>PRO - 정밀 분석</span>
          </label>
        </div>
        <button onClick={handlePredict} disabled={isAnalyzing} style={{ marginTop: 'auto', padding: '20px', backgroundColor: isAnalyzing ? '#222' : '#1A2C50', color: 'white', border: 'none', borderRadius: '12px', fontWeight: 'bold', fontSize: '18px', cursor: isAnalyzing ? 'not-allowed' : 'pointer' }}>
          {isAnalyzing ? "분석 중..." : "분석 시작 (PREDICT)"}
        </button>
      </aside>
    </div>
  );
};

export default AnalysisPage;
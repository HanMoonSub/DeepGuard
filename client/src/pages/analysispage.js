import React from 'react';
import { useNavigate } from 'react-router-dom';

const AnalysisPage = ({ sessionUser }) => {
  const navigate = useNavigate();

  return (
    <div style={{ backgroundColor: '#000', minHeight: '100vh', color: 'white', padding: '100px' }}>
      <h1>분석 페이지</h1>
      {sessionUser ? (
        <p style={{ color: '#39FF14' }}>{sessionUser.name}님, 정밀 분석을 시작합니다.</p>
      ) : (
        <p style={{ color: '#888' }}>게스트 모드로 분석 중입니다.</p>
      )}
      
      <div style={{ 
        width: '100%', height: '300px', border: '2px dashed #333', 
        display: 'flex', justifyContent: 'center', alignItems: 'center', borderRadius: '20px', marginTop: '20px'
      }}>
        파일을 여기로 드래그하거나 클릭하여 업로드 (분석 대기 중...)
      </div>
      <button 
        onClick={() => navigate('/main')}
        style={{ marginTop: '30px', backgroundColor: 'transparent', color: '#39FF14', border: '1px solid #39FF14', padding: '10px 20px', borderRadius: '20px', cursor: 'pointer' }}
      >
        ❮ 메인으로 돌아가기
      </button>
    </div>
  );
};

export default AnalysisPage;
import React from 'react';

const AnalysisPage = () => {
  return (
    <div style={{ backgroundColor: '#000', minHeight: '100vh', color: 'white', padding: '100px' }}>
      <h1>분석 페이지</h1>
      <p>여기에 파일을 업로드하거나 분석 결과를 확인할 수 있는 기능을 구현하세요.</p>
      <div style={{ 
        width: '100%', height: '300px', border: '2px dashed #333', 
        display: 'flex', justifyContent: 'center', alignItems: 'center', borderRadius: '20px' 
      }}>
        파일을 여기로 드래그하거나 클릭하여 업로드 (분석 대기 중...)
      </div>
    </div>
  );
};

export default AnalysisPage;
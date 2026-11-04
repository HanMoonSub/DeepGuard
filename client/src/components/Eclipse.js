import React from 'react';
import './Eclipse.css';

// 어두운 배경 위 장식용 이클립스 링. 위치·크기는 부모 쪽 CSS에서 .dg-eclipse로 조정
const Eclipse = ({ className = '' }) => (
  <div className={`dg-eclipse ${className}`.trim()} aria-hidden="true">
    <span className="dg-eclipse__halo" />
    <span className="dg-eclipse__spin dg-eclipse__spin--blur" />
    <span className="dg-eclipse__spin" />
    <span className="dg-eclipse__core" />
  </div>
);

export default Eclipse;

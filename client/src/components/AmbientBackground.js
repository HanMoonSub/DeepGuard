import React from 'react';
import './AmbientBackground.css';

// 밝은 페이지용 배경: 위쪽 스포트라이트 + 천천히 흐르는 오로라 + 미세 그레인
// 부모에 isolation: isolate 가 있어야 콘텐츠 뒤(z-index: -1)에 깔림
const AmbientBackground = () => (
  <div className="dg-ambient" aria-hidden="true">
    <span className="dg-ambient__aurora" />
    <span className="dg-ambient__spot" />
    <span className="dg-ambient__beam" />
    <span className="dg-ambient__grain" />
  </div>
);

export default AmbientBackground;

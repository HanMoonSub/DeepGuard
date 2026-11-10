import React from 'react';
import './AmbientBackground.css';

// 페이지 배경
// - light: 하늘색 그라데이션 + 천천히 흐르는 뭉게구름
// - dark: 기존 남색 배경 위 천천히 흐르는 오로라
// 부모에 isolation: isolate 가 있어야 콘텐츠 뒤(z-index: -1)에 깔림
const AmbientBackground = () => (
  <div className="dg-ambient" aria-hidden="true">
    <span className="dg-ambient__aurora" />
    <span className="dg-ambient__cloud dg-ambient__cloud--1" />
    <span className="dg-ambient__cloud dg-ambient__cloud--2" />
    <span className="dg-ambient__cloud dg-ambient__cloud--3" />
    <span className="dg-ambient__spot" />
    <span className="dg-ambient__beam" />
    <span className="dg-ambient__grain" />
  </div>
);

export default AmbientBackground;

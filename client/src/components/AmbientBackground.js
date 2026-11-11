import React from 'react';
import './AmbientBackground.css';

// Page background
// - light: sky-blue gradient + slowly drifting clouds
// - dark: slowly drifting aurora over the existing navy background
// The parent needs isolation: isolate for it to sit behind the content (z-index: -1)
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

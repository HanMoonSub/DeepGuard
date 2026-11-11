import React from 'react';
import './Eclipse.css';

// Decorative eclipse ring on dark backgrounds. Position and size are set in the parent's CSS via .dg-eclipse
const Eclipse = ({ className = '' }) => (
  <div className={`dg-eclipse ${className}`.trim()} aria-hidden="true">
    <span className="dg-eclipse__halo" />
    <span className="dg-eclipse__spin dg-eclipse__spin--blur" />
    <span className="dg-eclipse__spin" />
    <span className="dg-eclipse__core" />
  </div>
);

export default Eclipse;

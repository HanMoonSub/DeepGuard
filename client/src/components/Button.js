import React from 'react';
import './Button.css';

// variant: primary | ghost (for dark backgrounds) | outline (for light backgrounds) | inverse (for blue backgrounds)
const Button = ({ variant = 'primary', href, onClick, className = '', children, ...rest }) => {
  const cls = `dg-btn dg-btn--${variant} ${className}`.trim();

  if (href) {
    return (
      <a className={cls} href={href} onClick={onClick} {...rest}>
        {children}
      </a>
    );
  }
  return (
    <button type="button" className={cls} onClick={onClick} {...rest}>
      {children}
    </button>
  );
};

export default Button;

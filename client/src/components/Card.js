import React from 'react';
import './Card.css';

const Card = ({ icon, title, children, action, className = '' }) => (
  <div className={`dg-card ${className}`.trim()}>
    {icon && <div className="dg-card__icon">{icon}</div>}
    {title && <h3 className="dg-card__title">{title}</h3>}
    {children && <div className="dg-card__body">{children}</div>}
    {action && <div className="dg-card__action">{action}</div>}
  </div>
);

export default Card;

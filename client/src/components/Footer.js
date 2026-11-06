import React from 'react';
import logoMark from '../assets/logo-shield.svg';
import './Footer.css';

// 섹션 링크는 다른 페이지에서도 동작하도록 /main#... 형태로 건다
const COLUMNS = [
  {
    title: '서비스',
    links: [
      { label: '이미지 분석', href: '/analysis' },
      { label: '동영상 분석', href: '/video-analysis' },
    ],
  },
  {
    title: '리소스',
    links: [
      { label: 'Benchmark', href: '/main#bench' },
      { label: 'Explainability', href: '/main#xai' },
      { label: 'Docs (API)', href: '/main#api' },
      { label: 'GitHub ↗', href: 'https://github.com/HanMoonSub/DeepGuard', external: true },
      {
        label: 'Hugging Face Spaces ↗',
        href: 'https://huggingface.co/spaces/KoreaPeter/DeepFake-Image-Detection',
        external: true,
      },
    ],
  },
  {
    title: '프로젝트',
    links: [
      { label: 'Team', href: '/main#team' },
      { label: '충북대학교 소프트웨어학과' },
      { label: 'MIT License' },
    ],
  },
];

const Footer = () => (
  <footer className="dg-footer">
    <div className="dg-footer__inner">
      <div className="dg-footer__top">
        <div className="dg-footer__brand">
          <strong className="dg-footer__logo">
            <img className="dg-footer__mark" src={logoMark} alt="" aria-hidden="true" />DeepGuard
          </strong>
          <p>이미지와 영상 속 위조 흔적을 탐지하고, 시각적 근거까지 제공하는 딥페이크 탐지 서비스.</p>
        </div>

        {COLUMNS.map(({ title, links }) => (
          <div key={title} className="dg-footer__col">
            <h4>{title}</h4>
            <ul>
              {links.map(({ label, href, external }) => (
                <li key={label}>
                  {href ? (
                    <a href={href} {...(external ? { target: '_blank', rel: 'noreferrer' } : {})}>{label}</a>
                  ) : (
                    <span>{label}</span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div className="dg-footer__bottom">
        <span>© 2026 DeepGuard. All rights reserved.</span>
        <span>MIT License · Chungbuk National University</span>
      </div>
    </div>
  </footer>
);

export default Footer;

import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import Button from '../components/Button';
import Card from '../components/Card';
import ScrollHero from '../components/ScrollHero';
import xaiLow from '../assets/xai-low.png';
import xaiHigh from '../assets/xai-high.png';
import benchParamsAcc from '../assets/bench-params-acc.png';
import './mainpage.css';

const NAV_LINKS = [
  { href: '#bench', label: 'Benchmark' },
  { href: '#xai', label: 'Explainability' },
  { href: '#team', label: 'Team' },
];

const TEAM = [
  { name: '한문섭', role: 'Data & Backend Engineering', github: 'HanMoonSub', email: 'hanmoon3054@gmail.com' },
  { name: '이예솔', role: 'UI/UX & Frontend Engineering', github: 'ieeyesoi', email: 'yesol4138@chungbuk.ac.kr' },
  { name: '서윤제', role: 'AI Engineering', github: 'SEOYUNJE', email: 'seoyunje2001@gmail.com' },
];

const GitHubIcon = () => (
  <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor" aria-hidden="true">
    <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
  </svg>
);

const MailIcon = () => (
  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <rect x="3" y="5" width="18" height="14" rx="2" />
    <path d="M3 7l9 6 9-6" />
  </svg>
);

const HF_SPACE = 'https://huggingface.co/spaces/KoreaPeter';

// Secondary button: Hugging Face Spaces demo (new tab)
const HfDemo = ({ space }) => (
  <Button variant="outline" href={`${HF_SPACE}/${space}`} target="_blank" rel="noreferrer" title="Hugging Face Spaces에서 열기">
    🤗 Open in Spaces ↗
  </Button>
);

// Mini illustrations for the color header at the top of the feature cards (decorative)
const VIDEO_WAVE = [30, 45, 35, 80, 100, 90, 40, 25, 55, 85, 30, 20];
const FeatArt = ({ kind }) => (
  <span className="dg-feat__art" aria-hidden="true">
    {kind === 'image' && (
      <>
        <span className="dg-feat__frame">
          <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="4" width="18" height="16" rx="2" />
            <circle cx="9" cy="10" r="2" />
            <path d="M21 16l-5-5-9 9" />
          </svg>
        </span>
        <span className="dg-feat__badge"><small>판정 결과</small><b>FAKE 97.4%</b></span>
      </>
    )}
    {kind === 'video' && (
      <>
        <span className="dg-feat__frames">
          {[0, 1, 2, 3, 4].map((i) => <i key={i} />)}
          <span className="dg-feat__scan" />
        </span>
        <span className="dg-feat__wave">
          {VIDEO_WAVE.map((h, i) => <i key={i} className={h >= 80 ? 'is-hot' : undefined} style={{ height: `${h}%` }} />)}
        </span>
      </>
    )}
    {kind === 'xai' && (
      <span className="dg-feat__heat">
        <i className="dg-feat__blob" />
        <i className="dg-feat__blob dg-feat__blob--sm" />
        <i className="dg-feat__bbox" />
      </span>
    )}
  </span>
);

// SOTA comparison (from the table in deepguard/MS_EffGCViT.md, Latency: Tesla T4 × 1 · Batch 1 · Video 20 frames)
const SOTA_MODELS = [
  { id: 'b0', name: 'Fast (B0)', params: 8.7, time: 0.051, ours: true },
  { id: 'b5', name: 'Pro (B5)', params: 50.34, time: 0.479, ours: true },
  { id: 'sia', name: 'SIA', venue: "ECCV'22", params: 17.57, time: 0.094 },
  { id: 'core', name: 'CORE', venue: "CVPR'22", params: 20.81, time: 0.153 },
  { id: 'iid', name: 'IID', venue: "CVPR'23", params: 65.5, time: 0.181 },
  { id: 'effort', name: 'Effort', venue: "ICML'25", params: 303.4, time: 1.041 },
];

// Accuracy (%) · AUC (%) per dataset
const BENCH = [
  {
    key: 'celeb',
    label: 'Celeb-DF (v2)',
    acc: { b0: 98.4, b5: 99.8, sia: 99.0, core: 99.6, iid: 99.8, effort: 99.8 },
    auc: { b0: 99.7, b5: 99.8, sia: 99.8, core: 99.7, iid: 99.9, effort: 99.7 },
  },
  {
    key: 'ffpp',
    label: 'FaceForensics++',
    acc: { b0: 98.1, b5: 98.5, sia: 95.0, core: 98.3, iid: 97.3, effort: 98.4 },
    auc: { b0: 99.7, b5: 99.7, sia: 97.5, core: 99.8, iid: 99.6, effort: 99.8 },
  },
  {
    key: 'kodf',
    label: 'KoDF',
    acc: { b0: 96.6, b5: 97.9, sia: 97.0, core: 96.3, iid: 96.5, effort: 99.0 },
    auc: { b0: 97.9, b5: 98.3, sia: 98.8, core: 99.2, iid: 97.6, effort: 99.5 },
  },
];

const MAX_PARAMS = Math.max(...SOTA_MODELS.map((m) => m.params));
const MAX_TIME = Math.max(...SOTA_MODELS.map((m) => m.time));
// Performance section: switch between chart and table in the same spot
const BENCH_VIEWS = [
  { key: 'chart', label: '📈 한눈에 보기' },
  { key: 'table', label: '📋 상세 비교' },
];
const pct = (v, min, max) => `${Math.max(4, Math.min(100, ((v - min) / (max - min)) * 100))}%`;

// Analysis screen guide: numbered pins and description boxes
const GUIDE_ITEMS = [
  { n: 1, title: '예시로 체험하기', desc: '샘플 이미지를 누르면 파일 없이도 바로 판정을 체험할 수 있어요.' },
  { n: 2, title: '업로드 · 웹캠 촬영', desc: '얼굴이 담긴 이미지를 올리거나 웹캠으로 바로 찍어요.' },
  { n: 3, title: '대상 도메인', desc: '사진 속 인물에 맞춰 서양인 또는 동양인 도메인을 고를 수 있어요.' },
  { n: 4, title: '모델 선택', desc: 'FAST는 빠른 속도, PRO는 더 높은 정확도예요. PRO는 회원 전용이에요.' },
  { n: 5, title: '분석 결과', desc: 'REAL / FAKE 판정이 바로 뜨고, 상세 결과 보기로 이어져요.' },
];
const GUIDE_LEFT = [1, 2, 5];
const GUIDE_RIGHT = [3, 4];
const GUIDE_SAMPLES = ['동양인', '동양인', '서양인', '서양인'];
// Pin positions: 800 × 460 mockup coordinates (px / 8 = cqw)
const GUIDE_PINS = [
  [1, 6, 30],
  [2, 6, 96],
  [3, 598, 34],
  [4, 598, 104],
  [5, 286, 96],
];
const cq = (px) => `${px / 8}cqw`;

const GuideBox = ({ n, title, desc }) => (
  <div className="dg-guide__box" style={{ order: n }}>
    <span className="dg-guide__num">{n}</span>
    <h3>{title}</h3>
    <p>{desc}</p>
  </div>
);

// Simplified mockup of the analysis page (decorative illustration)
const GuideMock = () => (
  <div className="dg-gm" role="img" aria-label="이미지 분석 화면 예시: 예시 이미지, 업로드 영역, 분석 결과, 도메인과 모델 선택 패널">
    <div className="dg-gm__bar"><i /><i /><i /></div>
    <span className="dg-gm__lbl dg-gm__lbl--samples">예시로 체험하기</span>
    {GUIDE_SAMPLES.map((d, i) => (
      <div key={i} className={`dg-gm__th${d === '서양인' ? ' is-w' : ''}`} style={{ left: cq(120 + i * 70) }}>
        <b>FAKE</b>
        <span>{d}</span>
      </div>
    ))}
    <div className="dg-gm__up">
      <div className="dg-gm__ph">
        <svg viewBox="0 0 120 150" aria-hidden="true">
          <circle cx="60" cy="52" r="30" fill="#8b98ad" />
          <path d="M8 150c4-40 26-58 52-58s48 18 52 58z" fill="#6b7890" />
        </svg>
      </div>
      <div className="dg-gm__tool">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" />
        </svg>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
          <circle cx="12" cy="10" r="7" />
          <circle cx="12" cy="10" r="3" />
          <path d="M8 21h8" />
        </svg>
      </div>
    </div>
    <div className="dg-gm__res">
      <small>분석 결과</small>
      <strong>REAL</strong>
      <em>상세 결과 보기 →</em>
    </div>
    <div className="dg-gm__side">
      <span className="dg-gm__lbl dg-gm__lbl--domain">대상 도메인</span>
      <div className="dg-gm__tg"><span className="is-on">서양인</span><span>동양인</span></div>
      <span className="dg-gm__lbl dg-gm__lbl--model">모델 선택</span>
      <div className="dg-gm__md is-on dg-gm__md--fast">FAST<b>RECOMMENDED</b><small>빠른 속도</small></div>
      <div className="dg-gm__md dg-gm__md--pro">PRO<b className="is-member">회원 전용</b><small>높은 정확도</small></div>
      <div className="dg-gm__go">분석 시작</div>
    </div>
    {GUIDE_PINS.map(([n, x, y]) => (
      <span key={n} className="dg-gm__pin" style={{ left: cq(x), top: cq(y) }}>{n}</span>
    ))}
  </div>
);

// Video analysis guide: analyzing screen → result screen
const VGUIDE_ITEMS = [
  { n: 1, title: '예시로 체험하기', desc: '샘플 영상을 누르면 파일 없이도 바로 분석을 체험할 수 있어요.' },
  { n: 2, title: '영상 업로드 · 재생', desc: '영상을 올리면 가운데 재생 버튼으로 바로 내용을 확인할 수 있어요.' },
  { n: 3, title: '실시간 진행 상황', desc: '업로드·접수 → 프레임 분석 → 판정 단계와 경과 시간을 보여줘요. 페이지를 떠나도 분석은 계속돼요.' },
  { n: 4, title: '도메인 · 모델 선택', desc: '인물에 맞는 도메인과 FAST / PRO 모델을 고른 뒤 분석을 시작해요.' },
  { n: 5, title: '위조 비율', desc: '분석한 프레임 중 위조로 판정된 비율을 한눈에 보여줘요.' },
  { n: 6, title: '영상 정보', desc: 'FPS, 전체 프레임 수, 얼굴이 탐지된 프레임 수를 함께 알려줘요.' },
  { n: 7, title: '프레임 타임라인', desc: '프레임별 위조 확률을 막대로 보여주고, 50%를 넘으면 빨간색으로 표시해요. 막대에 올리면 해당 장면이 보여요.' },
];
const VM_W = 800;
// Timeline bars: fake probability per frame (all above 50% = 14 of 14 fake)
const VM_PROBS = [0.92, 0.71, 0.58, 0.84, 0.97, 0.63, 0.993, 0.88, 0.55, 0.76, 0.95, 0.67, 0.81, 0.6];
const VM_BAR_X = VM_PROBS.map((_, i) => 44 + i * (672 / (VM_PROBS.length - 1)));
const VM_BAR_H = VM_PROBS.map((p) => p * 76);
const VM_LINE = VM_BAR_X.map((x, i) => `${x.toFixed(1)},${(98 - VM_BAR_H[i]).toFixed(1)}`).join(' ');

// Scale a fixed-coordinate mockup (default 800 × 460) down to the parent width
const VideoMockFit = ({ label, children, width = VM_W, className = 'dg-vm-fit', innerClassName = 'dg-vm' }) => {
  const ref = useRef(null);
  const [scale, setScale] = useState(1);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return undefined;
    const ro = new ResizeObserver(([entry]) => setScale(entry.contentRect.width / width));
    ro.observe(el);
    return () => ro.disconnect();
  }, [width]);
  return (
    <div ref={ref} className={className} role="img" aria-label={label}>
      <div className={innerClassName} style={{ transform: `scale(${scale})` }}>{children}</div>
    </div>
  );
};

const VmPlayer = ({ x, y, w, h, person = 90 }) => (
  <div className="pl" style={{ left: x, top: y, width: w, height: h }}>
    <div className="scr">
      <svg width={person} height={person} viewBox="0 0 120 120" aria-hidden="true">
        <circle cx="60" cy="44" r="24" fill="#d9c3b0" />
        <path d="M14 120c4-34 22-48 46-48s42 14 46 48z" fill="#2b3446" />
        <path d="M52 74h16l-8 30z" fill="#e8eef8" />
      </svg>
    </div>
    <span className="play" />
    <div className="ctl"><em /><span>0:00 / 0:14</span><i /></div>
  </div>
);

const VmPins = ({ pins }) =>
  pins.map(([n, x, y]) => (
    <span key={n} className="pin" style={{ left: x, top: y }}>{n}</span>
  ));

const VideoMockAnalyzing = () => (
  <VideoMockFit label="영상 분석 중 화면 예시: 예시 영상, 재생 버튼이 있는 업로드 영상, 분석 진행 상황, 도메인과 모델 선택 패널">
    <div className="bar"><i /><i /><i /></div>
    <span className="lbl" style={{ left: 20, top: 58 }}>예시로 체험하기</span>
    <div className="th" style={{ left: 120 }}><b>FAKE</b><span>서양인</span></div>
    <div className="th r" style={{ left: 190 }}><b>REAL</b><span>서양인</span></div>
    <div className="card" style={{ left: 20, top: 100, width: 270, height: 340 }}>
      <span className="badge">● 분석 중</span>
      <div className="tools">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#8a94a8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12" />
        </svg>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#8a94a8" strokeWidth="2" aria-hidden="true">
          <circle cx="12" cy="10" r="7" />
          <circle cx="12" cy="10" r="3" />
          <path d="M8 21h8" />
        </svg>
      </div>
    </div>
    <VmPlayer x={34} y={170} w={242} h={140} />
    <div className="prog">
      <div className="eb">딥페이크 탐지 엔진</div>
      <h4>AI가 영상을 분석하고 있어요</h4>
      <div className="stp">
        <div className="done" style={{ left: 24 }}><b>✓</b>업로드·접수</div>
        <div className="now" style={{ left: 110 }}><b>2</b>프레임 분석</div>
        <div style={{ left: 196 }}><b>3</b>판정</div>
      </div>
      <div className="live"><span>비디오 분석 진행 중...</span><em>LIVE</em></div>
      <div className="lbar" />
      <div className="tm">경과 시간<b>00:10</b></div>
      <p>영상 길이에 따라 수십 초~수 분이 걸릴 수 있어요.<br />페이지를 떠나도 분석은 계속되고, 결과는<br />내 영상 기록에서 확인할 수 있어요.</p>
    </div>
    <div className="side">
      <span className="lbl" style={{ left: 14, top: 20, color: '#0b1220' }}>대상 도메인</span>
      <div className="tg"><span className="on">서양인</span><span>동양인</span></div>
      <span className="lbl" style={{ left: 14, top: 92, color: '#0b1220' }}>모델 선택</span>
      <div className="md" style={{ top: 112 }}>FAST<b>RECOMMENDED</b><small>빠른 속도</small></div>
      <div className="md on" style={{ top: 178 }}>PRO<b className="m">회원 전용</b><small>높은 정확도</small></div>
      <div className="go">분석 중...</div>
    </div>
    <VmPins pins={[[1, 0, 24], [2, 134, 230], [3, 286, 86], [4, 592, 28]]} />
  </VideoMockFit>
);

const VideoMockResult = () => (
  <VideoMockFit label="영상 분석 결과 화면 예시: 재생 버튼이 있는 영상, 위조 비율 100%, 영상 정보, 프레임별 위조 확률 타임라인">
    <div className="bar"><i /><i /><i /></div>
    <div className="card" style={{ left: 20, top: 40, width: 440, height: 226 }} />
    <VmPlayer x={36} y={54} w={408} h={198} person={120} />
    <div className="card" style={{ left: 476, top: 40, width: 304, height: 226 }}>
      <span className="cap" style={{ left: 18, top: 18 }}>위조 비율</span>
      <div className="dn">
        <svg width="96" height="96" viewBox="0 0 96 96" aria-hidden="true">
          <circle cx="48" cy="48" r="38" fill="none" stroke="#ff4d4d" strokeWidth="11" />
        </svg>
        <b>100%</b>
      </div>
      <div className="lg">
        <div><i style={{ background: '#ff4d4d' }} />FAKE<em style={{ color: '#ff4d4d' }}>100.0%</em></div>
        <div><i style={{ background: '#16a34a' }} />REAL<em style={{ color: '#16a34a', marginLeft: 17 }}>0.0%</em></div>
        <small>14개 프레임 중 14개 위조</small>
      </div>
      <div className="stat">
        <div>FPS<b>30.0</b></div>
        <div>전체 프레임<b>422</b></div>
        <div>얼굴 탐지<b>14 / 14</b></div>
      </div>
    </div>
    <div className="card tl">
      <span className="cap" style={{ left: 18, top: 13 }}>타임라인</span>
      <span className="lgd"><i style={{ background: '#ff4d4d' }} />변조 의심 (50% 초과)<i style={{ background: '#16a34a' }} />정상</span>
      <div className="area">
        <svg className="ln" width="724" height="118" aria-hidden="true">
          <polyline points={VM_LINE} fill="none" stroke="#206cfd" strokeWidth="2" strokeLinejoin="round" />
        </svg>
        <div className="th50" />
        <div className="bars">
          {VM_BAR_X.map((x, i) => <i key={i} style={{ left: x, height: VM_BAR_H[i] }} />)}
        </div>
        {VM_BAR_X.map((x, i) => (
          <span key={i} className="t" style={{ left: x }}>{`00:${String(i).padStart(2, '0')}`}</span>
        ))}
        <div className="tip" style={{ left: VM_BAR_X[6] }}><span /><small>00:06<b>99.3%</b></small></div>
      </div>
    </div>
    <VmPins pins={[[5, 464, 28], [6, 464, 192], [7, 6, 268]]} />
  </VideoMockFit>
);

const VGuideBox = ({ n, title, desc, wide }) => (
  <div className={`dg-guide__box dg-vguide__box${wide ? ' is-wide' : ''}`}>
    <span className="dg-guide__num">{n}</span>
    <h3>{title}</h3>
    <p>{desc}</p>
  </div>
);

// XAI guide: result/settings screen on the left, description on the right (linked by numbered circles and dashed lines)
const XAI_CHECKS = [
  ['High-level', '조명 불일치, 얼굴 기하, 그림자'],
  ['Low-level', '피부 질감, 경계 블렌딩, 압축 흔적'],
];
// top: vertical position (px) set so the dashed line ends at the number on wide screens
const XAI_STEPS = [
  { n: 1, top: 204, title: '브랜치 선택', desc: 'Low-level과 High-level 중 어떤 시선으로 볼지 골라요.' },
  { n: 2, top: 290, title: '위조 흔적 표시', desc: 'Heatmap, BBox, Heatmap + BBox 중 보기 방식을 바꿔요.' },
  { n: 3, top: 375, title: '원본 이미지 투명도', desc: '히트맵과 원본 얼굴이 겹치는 정도를 조절해요.' },
  { n: 4, top: 445, title: '위조 흔적 임계값', desc: '값을 높이면 더 확실한 영역만 남겨 보여줘요.' },
];
const XAI_W = 600;

const XaiStage = () => (
  <VideoMockFit
    label="XAI 결과 화면 예시: High-level과 Low-level 위조 흔적 이미지, 브랜치·표시 방식·투명도·임계값 설정 패널"
    width={XAI_W}
    className="dg-xai__fit"
    innerClassName="dg-xai__stage"
  >
    <div className="fx"><b /><b /><i /><i /><i /></div>
    <div className="card img">
      <img src={xaiHigh} alt="" width={250} height={303} loading="lazy" />
      <span className="bdg">High-level</span>
      <span className="dl">⬇ 다운로드</span>
    </div>
    <div className="card img lo">
      <img src={xaiLow} alt="" width={150} height={182} loading="lazy" />
      <span className="bdg">Low-level</span>
    </div>
    <div className="card set">
      <h4>시각화 설정</h4>
      <div><h5>브랜치</h5><div className="tg"><span>LOW</span><span className="on">HIGH</span></div></div>
      <div><h5>위조 흔적 표시</h5><div className="rd"><span>Heatmap</span><span>BBox</span><span className="on">Heatmap + BBox</span></div></div>
      <div><div className="row">원본 이미지 투명도<em>0.70</em></div><div className="sl"><i style={{ width: '50%' }} /><b style={{ left: '50%' }} /></div></div>
      <div><div className="row">위조 흔적 임계값<em>0.80</em></div><div className="sl"><i style={{ width: '40%' }} /><b style={{ left: '40%' }} /></div></div>
      <div className="go">적용</div>
      {[54, 140, 243, 285].map((top, i) => (
        <span key={top} className="pin" style={{ top }}>{i + 1}</span>
      ))}
    </div>
  </VideoMockFit>
);

const MainPage = ({ sessionUser, onLogout }) => {
  const navigate = useNavigate();
  const [benchKey, setBenchKey] = useState('celeb');
  const [benchView, setBenchView] = useState('chart');
  const bench = BENCH.find((b) => b.key === benchKey);
  // Accuracy · AUC bars: put zero slightly below the lowest value in the tab so differences are visible
  const range = (obj) => {
    const vals = Object.values(obj);
    return [Math.min(...vals) - 0.5, Math.max(...vals)];
  };
  const [accMin, accMax] = range(bench.acc);
  const [aucMin, aucMax] = range(bench.auc);

  const handleBasicAnalysis = () => navigate('/analysis');

  const handleVideoAnalysis = () => navigate('/video-analysis');

  // Navigate without a reload from buttons rendered as <a href>
  const linkTo = (handler) => (e) => {
    e.preventDefault();
    handler();
  };

  return (
    <div className="dg-landing">
      <Navbar variant="dark" solid sessionUser={sessionUser} onLogout={onLogout} links={NAV_LINKS} />

      {/* Scroll-driven hero (AI-generated photos + zoom-in video frames) */}
      <ScrollHero onStart={linkTo(handleBasicAnalysis)} />

      <section id="features" className="dg-section dg-section--alt">
        <div className="dg-wrap dg-center">
          <h2>탐지부터 설명까지, 한 번에</h2>
          <div className="dg-grid3 dg-features">
            <Card
              className="dg-feat dg-feat--image"
              icon={<FeatArt kind="image" />}
              title="이미지 탐지"
              action={
                <>
                  <Button variant="primary" href="/analysis" onClick={linkTo(handleBasicAnalysis)}>지금 분석하기</Button>
                  <HfDemo space="DeepFake-Image-Detection" />
                </>
              }
            >
              <p>사진을 올리면 진짜인지 가짜인지 바로 알려드려요.</p>
            </Card>
            <Card
              className="dg-feat dg-feat--video"
              icon={<FeatArt kind="video" />}
              title="영상 탐지"
              action={
                <>
                  <Button variant="primary" href="/video-analysis" onClick={linkTo(handleVideoAnalysis)}>지금 분석하기</Button>
                  <HfDemo space="DeepFake-Video-Detection" />
                </>
              }
            >
              <p>영상을 올리면 여러 장면을 살펴보고 조작 여부를 판단해드려요.</p>
            </Card>
            <Card
              className="dg-feat dg-feat--xai"
              icon={<FeatArt kind="xai" />}
              title="설명 가능한 XAI"
              action={
                <>
                  <Button variant="primary" href="#xai">예시 보기</Button>
                  <HfDemo space="DeepFake-Detection-XAI" />
                </>
              }
            >
              <p>왜 가짜로 판단했는지, 의심되는 부분을 색과 박스로 표시해드려요.</p>
            </Card>
          </div>
        </div>
      </section>

      <section id="bench" className="dg-section">
        <div className="dg-wrap">
          <div className="dg-center">
            <h2>신뢰할 수 있는 딥페이크 탐지기</h2>
          </div>

          <div className="dg-bench__card dg-sota">
            <div className="dg-sota__top">
              <div className="dg-sota__head">
                <h3>SOTA Model 성능 비교</h3>
              </div>
              <div className="dg-bench__seg" role="tablist" aria-label="성능 보기 방식">
                {BENCH_VIEWS.map((v) => (
                  <button
                    key={v.key}
                    id={`bench-tab-${v.key}`}
                    type="button"
                    role="tab"
                    aria-selected={v.key === benchView}
                    aria-controls={`bench-view-${v.key}`}
                    className={v.key === benchView ? 'is-active' : ''}
                    onClick={() => setBenchView(v.key)}
                  >
                    {v.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="dg-bench__views">
              <figure
                id="bench-view-chart"
                role="tabpanel"
                aria-labelledby="bench-tab-chart"
                className={`dg-bench__view dg-sota__fig${benchView === 'chart' ? ' is-active' : ''}`}
              >
                <img src={benchParamsAcc} alt="Celeb-DF, FaceForensics++, KoDF에서 파라미터 수 대비 정확도 산점도. MS-EffGCViT가 적은 파라미터로 SOTA 모델과 대등한 정확도를 보임" loading="lazy" />
              </figure>

              <div
                id="bench-view-table"
                role="tabpanel"
                aria-labelledby="bench-tab-table"
                className={`dg-bench__view${benchView === 'table' ? ' is-active' : ''}`}
              >
                <div className="dg-bench__tabs" role="tablist" aria-label="벤치마크 데이터셋">
                  {BENCH.map((b) => (
                    <button
                      key={b.key}
                      type="button"
                      role="tab"
                      aria-selected={b.key === benchKey}
                      className={`dg-bench__tab${b.key === benchKey ? ' is-active' : ''}`}
                      onClick={() => setBenchKey(b.key)}
                    >
                      {b.label}
                    </button>
                  ))}
                </div>
                <table className="dg-sota__table" role="tabpanel">
                  <thead>
                    <tr>
                      <th scope="col">Model</th>
                      <th scope="col">Params (M) ↓</th>
                      <th scope="col">Acc (%) ↑</th>
                      <th scope="col">AUC (%) ↑</th>
                      <th scope="col">Inference Latency (s) ↓</th>
                    </tr>
                  </thead>
                  <tbody>
                    {SOTA_MODELS.map((m) => {
                      const acc = bench.acc[m.id];
                      const auc = bench.auc[m.id];
                      return (
                        <tr key={m.id} className={m.ours ? 'is-ours' : ''}>
                          <th scope="row">
                            {m.name}
                            {m.ours ? <span className="dg-sota__badge">OURS</span> : <small>{m.venue}</small>}
                          </th>
                          <td>
                            <span className="dg-sota__bar"><i style={{ width: pct(m.params, 0, MAX_PARAMS) }} /></span>
                            {m.params.toFixed(2)}
                          </td>
                          <td>
                            <span className="dg-sota__bar"><i style={{ width: pct(acc, accMin, accMax) }} /></span>
                            {acc.toFixed(1)}
                          </td>
                          <td>
                            <span className="dg-sota__bar"><i style={{ width: pct(auc, aucMin, aucMax) }} /></span>
                            {auc.toFixed(1)}
                          </td>
                          <td>
                            <span className="dg-sota__bar"><i style={{ width: pct(m.time, 0, MAX_TIME) }} /></span>
                            {m.time.toFixed(3)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="guide" className="dg-section dg-section--alt">
        <div className="dg-wrap">
          <div className="dg-center">
            <h2>클릭 몇 번이면 끝나는 분석</h2>
            <p className="dg-sub">분석 화면의 기능을 번호를 따라 살펴보세요.</p>
          </div>
          <div className="dg-guide">
            <div className="dg-guide__col">
              {GUIDE_LEFT.map((n) => <GuideBox key={n} {...GUIDE_ITEMS[n - 1]} />)}
            </div>
            <div className="dg-guide__frame">
              <GuideMock />
            </div>
            <div className="dg-guide__col">
              {GUIDE_RIGHT.map((n) => <GuideBox key={n} {...GUIDE_ITEMS[n - 1]} />)}
            </div>
          </div>
        </div>
      </section>

      <section id="video-guide" className="dg-section dg-section--alt dg-vguide">
        <div className="dg-wrap">
          <div className="dg-center">
            <h2>올리면, 분석하고, 보여줘요</h2>
            <p className="dg-sub">영상 분석의 두 화면을 나란히 비교해 보세요.</p>
          </div>
          <div className="dg-vguide__grid">
            <div className="dg-vguide__col">
              <div className="dg-vguide__ttl"><span>STEP 1</span>분석 중 화면</div>
              <div className="dg-vguide__frame"><VideoMockAnalyzing /></div>
              <div className="dg-vguide__boxes">
                {VGUIDE_ITEMS.slice(0, 4).map((it) => <VGuideBox key={it.n} {...it} />)}
              </div>
            </div>
            <div className="dg-vguide__arw" aria-hidden="true">→</div>
            <div className="dg-vguide__col">
              <div className="dg-vguide__ttl"><span>STEP 2</span>결과 화면</div>
              <div className="dg-vguide__frame"><VideoMockResult /></div>
              <div className="dg-vguide__boxes">
                {VGUIDE_ITEMS.slice(4).map((it) => <VGuideBox key={it.n} {...it} wide={it.n === 7} />)}
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="xai" className="dg-section dg-section--alt dg-xai">
        <div className="dg-wrap">
          <div className="dg-center">
            <h2>왜 가짜인지, 보여드립니다</h2>
            <p className="dg-sub">
              단순히, 딥페이크 확률 값 제공을 넘어서 위조 흔적 영역을 시각적으로 제공해줍니다
            </p>
          </div>

          <div className="dg-xai__grid">
            {/* Numbered circles → dashed lines to the description on the right (wide screens only) */}
            <svg className="dg-xai__lnk" width="660" height="540" aria-hidden="true">
              <path d="M553 217H652" />
              <path d="M553 303H652" />
              <path d="M553 406H600V388H652" />
              <path d="M553 448H600V458H652" />
            </svg>
            <XaiStage />
            <div className="dg-xai__txt">
              <p className="dg-xai__lead">DeepGuard XAI는 두 가지 시선으로 근거를 보여줘요:</p>
              <ul className="dg-xai__chk">
                {XAI_CHECKS.map(([name, desc]) => (
                  <li key={name}>
                    <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">
                      <circle cx="12" cy="12" r="11" fill="#206cfd" />
                      <path d="M7 12.5l3.2 3.2L17 9" fill="none" stroke="#fff" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    <span><b>{name}</b> · {desc}</span>
                  </li>
                ))}
              </ul>
              <ol className="dg-xai__steps">
                {XAI_STEPS.map(({ n, top, title, desc }) => (
                  <li key={n} style={{ '--top': `${top}px` }}>
                    <span>{n}</span>
                    <p><b>{title}</b>{desc}</p>
                  </li>
                ))}
              </ol>
            </div>
          </div>
        </div>
      </section>

      <section id="team" className="dg-section dg-section--alt">
        <div className="dg-wrap dg-center">
          <h2>충북대학교 소프트웨어학과 졸업 프로젝트</h2>
          <div className="dg-grid3">
            {TEAM.map(({ name, role, github, email }) => (
              <Card key={github} className="dg-member">
                <img
                  className="dg-member__avatar"
                  src={`https://github.com/${github}.png?size=160`}
                  alt={`${name} 프로필 사진`}
                  width={80}
                  height={80}
                  loading="lazy"
                />
                <h3 className="dg-member__name">{name}</h3>
                <p className="dg-member__role">{role}</p>
                <div className="dg-member__links">
                  <a href={`https://github.com/${github}`} target="_blank" rel="noreferrer">
                    <GitHubIcon />{github}
                  </a>
                  <a href={`mailto:${email}`}>
                    <MailIcon />{email}
                  </a>
                </div>
              </Card>
            ))}
          </div>
        </div>
      </section>

      <div className="dg-band">
        <div className="dg-wrap">
          <h2>딥페이크 탐지, 지금 시작하세요</h2>
          <p className="dg-sub dg-band__sub">이미지 한 장으로 바로 체험해볼 수 있어요.</p>
          <Button variant="primary" href="/analysis" onClick={linkTo(handleBasicAnalysis)}>Try Now</Button>
        </div>
      </div>

      <Footer />
    </div>
  );
};

export default MainPage;

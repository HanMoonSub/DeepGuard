import React from 'react';
import { useNavigate } from 'react-router-dom';

import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import Button from '../components/Button';
import Card from '../components/Card';
import './mainpage.css';

const NAV_LINKS = [
  { href: '#bench', label: 'Benchmark' },
  { href: '#xai', label: 'Explainability' },
  { href: '#api', label: 'Docs' },
  { href: '#team', label: 'Team' },
];

// 히어로/XAI 섹션의 얼굴 스캔 일러스트
const ScanVisual = ({ tag, result, className = '' }) => (
  <div className={`dg-scan ${className}`.trim()}>
    <span className="dg-scan__tag">{tag}</span>
    <div className="dg-scan__face" />
    <div className="dg-scan__heat dg-scan__heat--1" />
    <div className="dg-scan__heat dg-scan__heat--2" />
    <div className="dg-scan__box"><b>FAKE</b></div>
    {result && (
      <div className="dg-scan__result">
        <span>Deepfake Probability</span>
        <strong>{result}</strong>
      </div>
    )}
  </div>
);

const MainPage = ({ sessionUser, onLogout }) => {
  const navigate = useNavigate();

  const handleBasicAnalysis = () => navigate('/analysis');

  const handleVideoAnalysis = () => navigate('/video-analysis');

  // <a href>로 렌더링된 버튼에서 새로고침 없이 이동
  const linkTo = (handler) => (e) => {
    e.preventDefault();
    handler();
  };

  return (
    <div className="dg-landing">
      <Navbar variant="dark" sessionUser={sessionUser} onLogout={onLogout} links={NAV_LINKS} />

      <header className="dg-hero">
        <div className="dg-wrap dg-hero__grid">
          <div>
            <span className="dg-pill">MS-EffGCViT · Celeb-DF-v2 AUC 0.9984</span>
            <h1>딥페이크,<br /><em>눈으로 확인</em>하세요</h1>
            <p className="dg-hero__lead">
              이미지와 영상 속 조작 흔적을 탐지하고, 어디가 왜 가짜인지 시각적 근거까지 제공하는 딥페이크 탐지 서비스.
            </p>
            <div className="dg-hero__cta">
              <Button variant="primary" href="/analysis" onClick={linkTo(handleBasicAnalysis)}>지금 분석해보기</Button>
              <Button variant="ghost" href="#api">API 문서 →</Button>
            </div>
          </div>
          <ScanVisual tag="SAMPLE · Fake" result="0.9873" className="dg-scan--hero" />
        </div>
      </header>

      <section id="features" className="dg-section">
        <div className="dg-wrap dg-center">
          <h2>탐지부터 설명까지, 한 번에</h2>
          <p className="dg-sub">CNN의 국소 감각과 Vision Transformer의 전역 문맥을 결합한 하이브리드 모델로 미세한 조작을 잡아냅니다.</p>
          <div className="dg-grid3">
            <Card
              icon="🖼"
              title="이미지 탐지"
              action={<a href="/analysis" onClick={linkTo(handleBasicAnalysis)}>자세히 보기 →</a>}
            >
              <p>얼굴을 자동 검출하고 딥페이크 확률을 즉시 반환합니다.</p>
            </Card>
            <Card
              icon="🎬"
              title="영상 탐지"
              action={<a href="/video-analysis" onClick={linkTo(handleVideoAnalysis)}>자세히 보기 →</a>}
            >
              <p>프레임을 샘플링해 conf / mean / vote 방식으로 종합 판정합니다.</p>
            </Card>
            <Card icon="🔍" title="설명 가능한 AI" action={<a href="#xai">자세히 보기 →</a>}>
              <p>Grad-CAM 계열 히트맵과 박스로 조작 의심 영역을 시각화합니다.</p>
            </Card>
          </div>
        </div>
      </section>

      <section id="bench" className="dg-section dg-section--alt">
        <div className="dg-wrap dg-center">
          <h2>검증된 성능</h2>
          <p className="dg-sub">Celeb-DF-v2, FaceForensics++, KoDF 3개 벤치마크로 학습·평가했습니다.</p>
          <div className="dg-stats">
            <div><b>99.84%</b><span>AUC · Celeb-DF-v2 (b5)</span></div>
            <div><b>99.81%</b><span>Accuracy · Celeb-DF-v2 (b5)</span></div>
            <div><b>3</b><span>벤치마크 데이터셋</span></div>
            <div><b>62K+</b><span>KoDF 원본 영상</span></div>
          </div>
          <div className="dg-chips dg-chips--center">
            <span>Celeb-DF-v2</span><span>FaceForensics++</span><span>KoDF</span>
          </div>
        </div>
      </section>

      <section id="xai" className="dg-section">
        <div className="dg-wrap dg-row">
          <div>
            <h2>왜 가짜인지,<br />보여드립니다</h2>
            <p className="dg-sub">Dual-Branch XAI가 모델의 멀티스케일 추론을 그대로 따라가며 국소 아티팩트와 전역 구조 이상을 각각 시각화합니다.</p>
            <ul className="dg-list">
              <li><b>Low-level</b> · 피부 질감, 경계 블렌딩, 압축 흔적</li>
              <li><b>High-level</b> · 조명 불일치, 얼굴 기하, 그림자</li>
              <li>HiResCAM · LayerCAM · GradCAM++ · XGradCAM 외 6종</li>
            </ul>
          </div>
          <ScanVisual tag="Heatmap + BBox" className="dg-scan--xai" />
        </div>
      </section>

      <section id="api" className="dg-section dg-section--alt">
        <div className="dg-wrap dg-row">
          <div>
            <h2>코드 몇 줄이면 끝</h2>
            <p className="dg-sub">PyTorch·timm 인터페이스와 Hugging Face Hub로 바로 불러와 쓸 수 있습니다.</p>
            <div className="dg-chips">
              <span>ms_eff_gcvit_b0 · 8.7M</span><span>ms_eff_gcvit_b5 · 50.3M</span>
            </div>
            <Button
              variant="outline"
              className="dg-api__github"
              href="https://github.com/HanMoonSub/DeepGuard"
              target="_blank"
              rel="noreferrer"
            >
              GitHub 보기
            </Button>
          </div>
          <pre className="dg-code">
            <span className="k">from</span>{' deepguard '}<span className="k">import</span>{' ms_eff_gcvit_b0\n\nmodel = ms_eff_gcvit_b0(\n    pretrained='}
            <span className="k">True</span>{',\n    dataset='}<span className="s">"celeb_df_v2"</span>{'\n)\n'}
            <span className="c"># 영상 판정</span>{'\nresult = predictor.predict_video(\n    video_path='}
            <span className="s">"video.mp4"</span>{',\n    num_frames=20, agg_mode='}<span className="s">"conf"</span>{'\n)'}
          </pre>
        </div>
      </section>

      <section id="team" className="dg-section">
        <div className="dg-wrap dg-center">
          <h2>충북대학교 소프트웨어학과 졸업 프로젝트</h2>
          <div className="dg-grid3">
            <Card title="한문섭"><p>Data &amp; Backend Engineering</p></Card>
            <Card title="이예솔"><p>UI/UX &amp; Frontend Engineering</p></Card>
            <Card title="서윤제"><p>AI Engineering</p></Card>
          </div>
        </div>
      </section>

      <div className="dg-band">
        <div className="dg-wrap">
          <h2>딥페이크 탐지, 지금 시작하세요</h2>
          <p className="dg-sub dg-band__sub">이미지 한 장으로 바로 체험해볼 수 있어요.</p>
          <Button variant="inverse" href="/analysis" onClick={linkTo(handleBasicAnalysis)}>Try Now</Button>
        </div>
      </div>

      <Footer />
    </div>
  );
};

export default MainPage;

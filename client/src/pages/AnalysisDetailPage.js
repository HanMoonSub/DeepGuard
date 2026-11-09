import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';

import Navbar from '../components/Navbar';
import AmbientBackground from '../components/AmbientBackground';
import useTheme from '../hooks/useTheme';
import Footer from '../components/Footer';
import Button from '../components/Button';
import Card from '../components/Card';
import useCountUp from '../utils/useCountUp';
import './AnalysisDetailPage.css';

// 공통 레이아웃: Navbar(light) + 본문 + Footer
const DetailLayout = ({ sessionUser, children }) => {
  const [theme, setTheme] = useTheme();
  return (
  <div className="dg-detail">
    <AmbientBackground />
    <Navbar variant="light" sessionUser={sessionUser} theme={theme} onThemeChange={setTheme} />
    <main className="dg-detail__main">
      <div className="dg-wrap">{children}</div>
    </main>
    <Footer />
  </div>
  );
};

// 결과 옆 정보 목록의 한 줄 (라벨 · 값)
const InfoRow = ({ label, value, tone }) => (
  <div className="dg-detail__info-row">
    <dt>{label}</dt>
    <dd className={tone ? `dg-detail__tone--${tone}` : ''}>{value}</dd>
  </div>
);

// 위조 확률 원형 게이지
const GAUGE_R = 76;
const GAUGE_C = 2 * Math.PI * GAUGE_R;
const ProbGauge = ({ prob, tone }) => {
  const valid = prob !== -1;
  const pct = valid ? Math.min(Math.max(Number(prob), 0), 1) : 0;
  const shown = useCountUp(pct);
  return (
    <svg className="dg-detail__gauge" width="180" height="180" viewBox="0 0 180 180" role="img"
      aria-label={valid ? `위조 확률 ${(pct * 100).toFixed(1)}%` : '위조 확률 없음'}>
      <circle className="dg-detail__gauge-track" cx="90" cy="90" r={GAUGE_R} fill="none" strokeWidth="16" />
      {valid && (
        <circle className={`dg-detail__gauge-arc dg-detail__gauge-arc--${tone}`} cx="90" cy="90" r={GAUGE_R}
          fill="none" strokeWidth="16" strokeLinecap="round"
          strokeDasharray={`${shown * GAUGE_C} ${GAUGE_C}`} transform="rotate(-90 90 90)"
          style={{ opacity: shown > 0.0005 ? 1 : 0 }} />
      )}
      <text className="dg-detail__gauge-value" x="90" y="94" textAnchor="middle">
        {valid ? `${(shown * 100).toFixed(1)}%` : 'N/A'}
      </text>
      <text className="dg-detail__gauge-label" x="90" y="120" textAnchor="middle">위조 확률</text>
    </svg>
  );
};

// 기록 항목의 판정 라벨 (분석 페이지 기록 목록과 같은 규칙)
const itemLabelOf = (item) => {
  if (item.label && item.label !== 'UNKNOWN') return item.label.toUpperCase();
  const p = item.prob ?? item.score ?? item.analysis?.prob ?? -1;
  if (p === -1) return 'UNKNOWN';
  return p > 0.5 ? 'FAKE' : 'REAL';
};

// 판정별 안내 문구
const VERDICT_COPY = {
  fake: {
    badge: '딥페이크로 의심돼요',
    image: '얼굴 영역에서 위조 흔적이 감지되었어요. 히트맵에서 모델이 주목한 부위를 확인해 보세요.',
    video: '영상의 얼굴 영역에서 위조 흔적이 감지되었어요. 타임라인에서 의심 구간을 확인해 보세요.',
  },
  real: {
    badge: '실제 얼굴로 보여요',
    image: '뚜렷한 위조 흔적이 발견되지 않았어요. 히트맵에서 모델이 본 부위를 확인할 수 있어요.',
    video: '뚜렷한 위조 흔적이 발견되지 않았어요. 타임라인에서 구간별 결과를 확인할 수 있어요.',
  },
  unknown: {
    badge: '판정할 수 없어요',
    image: '위조 확률을 계산하지 못했어요.',
    video: '위조 확률을 계산하지 못했어요.',
  },
};

const AnalysisDetailPage = ({ sessionUser }) => {
  const { state, pathname } = useLocation();
  const navigate = useNavigate();
  const [history, setHistory] = useState([]);

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchDetail = async () => {
      const imageId = state?.image_id;
      const videoId = state?.video_id;

      if (!imageId && !videoId) { setLoading(false); return; }

      try {
        if (imageId) {
          const res = await axios.get(`/image/history/${imageId}`);
          setData(res.data.context);
        } else {
          const res = await axios.get(`/video/history/${videoId}`);
          setData(res.data.context);
        }
      } catch (e) {
        console.error("상세 조회 실패", e);
      } finally {
        setLoading(false);
      }
    };
    fetchDetail();
  }, [state]);

  // 하단 '최근 분석' 줄 — 로그인 사용자만, 이미지/동영상 기록 중 보고 있는 종류로
  const isVideoRoute = !!state?.video_id;
  useEffect(() => {
    if (!sessionUser) { setHistory([]); return; }
    axios.get(isVideoRoute ? '/video/history' : '/image/history')
      .then((res) => { if (res.data.status === 'success') setHistory(res.data.context || []); })
      .catch(() => console.log('히스토리 로드 실패'));
  }, [sessionUser, isVideoRoute]);

  if (loading) {
    return (
      <DetailLayout sessionUser={sessionUser}>
        <div className="dg-detail__state">
          <div className="dg-detail__spinner" />
          <p>불러오는 중...</p>
        </div>
      </DetailLayout>
    );
  }

  if (!data) {
    return (
      <DetailLayout sessionUser={sessionUser}>
        <div className="dg-detail__state">
          <p>분석 데이터를 찾을 수 없거나 비정상적인 접근입니다.</p>
          <Button variant="primary" onClick={() => navigate(-1)}>돌아가기</Button>
        </div>
      </DetailLayout>
    );
  }

  const prob = data.prob ?? data.score ?? data.analysis?.prob ?? data.result_prob ?? -1;

  let label = 'UNKNOWN';
  if (data.label && data.label !== 'UNKNOWN') {
    label = data.label.toUpperCase();
  } else if (prob !== -1) {
    label = prob > 0.5 ? 'FAKE' : 'REAL';
  }

  const isInvalid = prob === -1;

  const mediaLoc = data.video_loc || data.media_loc || data.image_loc || '';
  const mediaSrc = mediaLoc.startsWith('blob') ? mediaLoc : mediaLoc;
  const isVideo = !!data.video_loc || (data.score !== undefined && !data.image_loc) || window.location.pathname.includes('video');

  const isWarning = data.status?.toUpperCase() === 'WARNING';

  // 라벨 색: FAKE=danger, REAL=success, 판정 불가=muted
  const verdictTone = label === 'FAKE' ? 'fake' : (isInvalid ? 'unknown' : 'real');

  const handleImageHeatmap = () => {
    if (!sessionUser) {
      alert("로그인이 필요한 기능이에요.");
      return;
    }

    navigate('/image-heatmap', {
      state: {
        image_id: data.image_id,
        image_loc: data.image_loc,
        model_type: data.model_type || 'fast',
        version_type: data.version_type || 'v1',
        prob,
        label,
      },
    });
  };

  const handleVideoHeatmap = () => {
    if (!sessionUser) {
      alert("로그인이 필요한 기능이에요.");
      return;
    }
    navigate('/video-timeline', {
      state: {
        ...data
      }
    });
  };


  return (
    <DetailLayout sessionUser={sessionUser}>
      {/* 상단: 뒤로가기 + 분석 조건 태그 */}
      <header className="dg-detail__head">
        <button type="button" className="dg-detail__back" onClick={() => navigate(-1)}>
          <span aria-hidden="true">←</span> 뒤로가기
        </button>
      </header>

      <h2 className="dg-detail__title">{isVideo ? '동영상' : '이미지'} 분석 결과</h2>

      <div className="dg-detail__grid">
        {/* 왼쪽: 미디어 */}
        <Card className="dg-card--static dg-detail__media-card">
          <div className="dg-detail__media">
            {isVideo && mediaLoc ? (
              <video src={mediaSrc} controls autoPlay muted />
            ) : mediaLoc ? (
              <img src={mediaSrc} alt="Analyzed media"
                onError={(e) => { e.target.src = 'https://via.placeholder.com/600x400?text=No+Image'; }} />
            ) : (
              <div className="dg-detail__media-empty">미디어가 존재하지 않습니다.</div>
            )}
          </div>

        </Card>

        {/* 오른쪽: 판정 카드 or WARNING */}
        <div className="dg-detail__side">
          {isWarning ? (
            <Card className="dg-card--static dg-detail__warning">
              <div className="dg-detail__warn-icon" aria-hidden="true">!</div>
              <p className="dg-detail__warn-title">분석을 완료하지 못했어요</p>
              <p className="dg-detail__warn-msg">{data.result_msg}</p>
            </Card>
          ) : (
            <>
              {/* 결과 중심 스플릿: 위조 확률 게이지 + 판정 + 근거 보기 */}
              <section className="dg-detail__panel dg-detail__verdict-card">
                <ProbGauge prob={prob} tone={verdictTone} />
                <span className={`dg-detail__badge dg-detail__badge--${verdictTone}`}>{VERDICT_COPY[verdictTone].badge}</span>
                <p className="dg-detail__desc">{VERDICT_COPY[verdictTone][isVideo ? 'video' : 'image']}</p>
                {mediaLoc && (
                  <Button variant="primary" className="dg-btn--block dg-detail__xai-btn"
                    onClick={isVideo ? handleVideoHeatmap : handleImageHeatmap}>
                    {isVideo ? '타임라인으로 근거 보기' : '히트맵으로 근거 보기'}
                  </Button>
                )}
              </section>

              <dl className="dg-detail__panel dg-detail__info">
                <InfoRow label="모델" value={`${(data.version_type || 'v1').toUpperCase()} · ${data.model_type?.toUpperCase() || 'FAST'}`} />
                <InfoRow label="대상 도메인" value={data.domain_type || '서양인'} />
                <InfoRow label="판정" value={label} tone={verdictTone} />
              </dl>
            </>
          )}
        </div>
      </div>

      {/* 최근 분석 */}
      {history.length > 0 && (
        <section className="dg-detail__recent">
          <h3 className="dg-detail__recent-title">최근 분석</h3>
          <div className="dg-detail__recent-list">
            {history.slice(0, 12).map((item, index) => {
              const id = isVideoRoute ? (item.video_id || item.id) : item.image_id;
              const itemLabel = itemLabelOf(item);
              const tone = itemLabel === 'FAKE' ? 'fake' : itemLabel === 'REAL' ? 'real' : 'unknown';
              const isCurrent = String(id) === String(isVideoRoute ? state?.video_id : state?.image_id);
              return (
                <button key={id || index} type="button"
                  className={`dg-detail__recent-item${isCurrent ? ' is-current' : ''}`}
                  aria-current={isCurrent ? 'page' : undefined}
                  onClick={() => navigate(pathname, { state: isVideoRoute ? { video_id: id } : { image_id: id } })}>
                  <span className="dg-detail__recent-thumb">
                    {isVideoRoute
                      ? (item.video_loc ? <video src={`${item.video_loc}#t=0.1`} muted preload="metadata" /> : null)
                      : (item.image_loc ? <img src={item.image_loc} alt="" /> : null)}
                  </span>
                  <span className="dg-detail__recent-meta">
                    <b className={`dg-detail__tone--${tone}`}>{tone === 'unknown' ? '판정 불가' : itemLabel}</b>
                    <span>{item.created_at?.slice(0, 10)}</span>
                  </span>
                </button>
              );
            })}
          </div>
        </section>
      )}
    </DetailLayout>
  );
};

export default AnalysisDetailPage;

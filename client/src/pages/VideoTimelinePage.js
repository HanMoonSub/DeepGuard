import React, { useEffect, useState, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';

import Navbar from '../components/Navbar';
import useTheme from '../hooks/useTheme';
import AmbientBackground from '../components/AmbientBackground';
import Footer from '../components/Footer';
import Button from '../components/Button';
import Card from '../components/Card';
import VideoSegmentTrack from '../components/VideoSegmentTrack';
import useCountUp from '../utils/useCountUp';
import './VideoTimelinePage.css';

// 도넛 색 (tokens.css의 danger와 동일 값)
const COLOR_FAKE = '#ff4d4d';
const DONUT_R = 70;

// 공통 레이아웃: Navbar(light, 다크/라이트 전환) + 본문 + Footer
const TimelineLayout = ({ sessionUser, children }) => {
  const [theme, setTheme] = useTheme();
  return (
    <div className="dg-timeline">
      <AmbientBackground />
      <Navbar variant="light" sessionUser={sessionUser} theme={theme} onThemeChange={setTheme} />
      <main className="dg-timeline__main">
        <div className="dg-timeline__wrap">{children}</div>
      </main>
      <Footer />
    </div>
  );
};

const normalizeFrame = (f, i) => {
  const raw = f.score ?? f.fake_score ?? 0;
  const fake_score = raw <= 1 ? raw * 100 : raw;
  const ts = f.frame_time != null
    ? new Date(f.frame_time * 1000).toISOString().substr(14, 5)
    : f.timestamp ?? `00:${String(i * 2).padStart(2, '0')}`;
  return { ...f, fake_score, frame_index: f.frame_index ?? i, timestamp: ts };
};

// /detail 응답 상태 코드 → 에러 화면 type
const errorTypeFromStatus = (status) => {
  if (status === 401) return 'auth';       // 로그인 필수 API
  if (status === 400) return 'incomplete'; // 추론이 SUCCESS가 아닌 영상
  if (status === 404) return 'notfound';   // 기록/메타 없음
  return 'server';                         // 5xx, 네트워크 오류 등
};

// 에러 화면 문구/버튼 — actions: home(동영상 분석) | login | retry | back
const ERROR_VIEWS = {
  nostate: {
    tone: 'primary',
    title: '분석 정보가 전달되지 않았어요',
    desc: () => '분석 상세 화면의 ‘비디오 위조 흔적 분석하기’로 들어와 주세요.',
    actions: () => ['home', 'back'],
  },
  auth: {
    tone: 'primary',
    title: '로그인이 필요해요',
    desc: () => '구간별 분석은 회원만 볼 수 있어요.',
    actions: () => ['login', 'back'],
  },
  incomplete: {
    tone: 'warning',
    title: '프레임 분석을 볼 수 없는 영상이에요',
    desc: (data) => (data?.status === 'PENDING'
      ? '아직 분석이 진행 중이에요. 잠시 후 다시 시도해 주세요.'
      : '분석이 정상 완료된 영상만 프레임별 결과를 제공해요.'),
    actions: (data) => (data?.status === 'PENDING' ? ['retry', 'back'] : ['back']),
  },
  notfound: {
    tone: 'primary',
    title: '분석 기록을 찾을 수 없어요',
    desc: () => '삭제되었거나 보관 기간이 지난 기록일 수 있어요.',
    actions: () => ['home', 'back'],
  },
  empty: {
    tone: 'warning',
    title: '프레임별 결과가 없어요',
    desc: () => '이 영상은 프레임 단위 결과가 저장되지 않았어요.',
    actions: () => ['back'],
  },
  server: {
    tone: 'danger',
    title: '결과를 불러오지 못했어요',
    desc: () => '잠시 후 다시 시도해 주세요.',
    actions: () => ['retry', 'back'],
  },
};

const VideoTimelinePage = ({ sessionUser }) => {
  const { state: data } = useLocation();
  const navigate = useNavigate();
  const [timelineData, setTimelineData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError]         = useState(null); // { type, detail }
  const [retryKey, setRetryKey]   = useState(0);    // 다시 시도 시 fetch 재실행용
  const [thumbs, setThumbs]       = useState({});   // { frame_index: dataURL } 그래프 hover 미리보기용
  const [hoverIdx, setHoverIdx]   = useState(null); // 그래프에서 커서가 올라간 프레임 순번
  const [duration, setDuration]   = useState(null); // 영상 길이(초) — video 메타데이터에서
  const [isGraphOpen, setIsGraphOpen] = useState(false); // 타임라인 그래프 펼침 (기본 접힘)

  const videoId = data?.video_id || data?.id;
  const mediaLoc = data?.video_loc || '';
  const mediaSrc = mediaLoc.startsWith('blob') ? mediaLoc : mediaLoc;

  // hover 툴팁 위치 계산용: 그래프 영역과 각 구간(열) ref
  const chartRef = useRef(null);
  const colRefs  = useRef([]);
  const videoRef = useRef(null);

  useEffect(() => {
    const fetchTimelineDetails = async () => {
      if (!videoId) { setIsLoading(false); return; }
      setIsLoading(true);
      setError(null);
      try {
        const response = await axios.get(`/inference/video/${videoId}/detail`);
        const d = response.data;
        let frames = [];
        if (d?.frames?.length)        frames = d.frames;
        else if (d?.timeline?.length) frames = d.timeline;
        if (frames.length) {
          setTimelineData(frames.map(normalizeFrame));
        } else {
          setTimelineData([]);
          setError({ type: 'empty' });
        }
      } catch (err) {
        setTimelineData([]);
        const detail = err.response?.data?.detail;
        setError({
          type: errorTypeFromStatus(err.response?.status),
          detail: typeof detail === 'string' ? detail : null,
        });
      } finally {
        setIsLoading(false);
      }
    };
    fetchTimelineDetails();
  }, [videoId, retryKey]);

  // ── fake/real 비율 (프레임 카운트 기준) ──
  const fakeCount  = timelineData.filter(f => f.fake_score > 50).length;
  const totalCount = timelineData.length;
  const fakeRatio  = totalCount ? (fakeCount / totalCount) * 100 : 0;
  // 도넛: 로딩이 끝나면 0%에서 차오름 (상세 화면 원형 차트와 같은 효과)
  const shownRatio = useCountUp(isLoading ? 0 : fakeRatio);

  // ── 프레임 썸네일 추출 ──
  // 숨은 video로 각 frame_time에 seek → canvas에 그려 작은 jpg로 저장 (동일 출처라 canvas 오염 없음)
  useEffect(() => {
    if (!mediaSrc || !timelineData.length) return;
    let cancelled = false;
    const video = document.createElement('video');
    video.muted = true;
    video.preload = 'auto';
    video.src = mediaSrc;
    const canvas = document.createElement('canvas');

    const seek = (t) => new Promise((resolve, reject) => {
      const onSeeked = () => { cleanup(); resolve(); };
      const onError = () => { cleanup(); reject(new Error('seek failed')); };
      const cleanup = () => { video.removeEventListener('seeked', onSeeked); video.removeEventListener('error', onError); };
      video.addEventListener('seeked', onSeeked);
      video.addEventListener('error', onError);
      video.currentTime = t;
    });

    const run = async () => {
      try {
        await new Promise((resolve, reject) => {
          if (video.readyState >= 1) return resolve();
          video.addEventListener('loadedmetadata', resolve, { once: true });
          video.addEventListener('error', reject, { once: true });
        });
        const W = 192;
        canvas.width = W;
        canvas.height = Math.round(W * (video.videoHeight / video.videoWidth || 9 / 16));
        const ctx = canvas.getContext('2d');
        for (const frame of timelineData) {
          if (cancelled) return;
          if (frame.frame_time == null) continue;
          await seek(Math.min(frame.frame_time + 0.01, Math.max(video.duration - 0.05, 0)));
          if (cancelled) return;
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          const url = canvas.toDataURL('image/jpeg', 0.75);
          setThumbs(prev => ({ ...prev, [frame.frame_index]: url }));
        }
      } catch (e) {
        // 썸네일은 부가 기능 — 실패해도 시각/확률 툴팁은 그대로 표시
      }
    };
    run();
    return () => { cancelled = true; video.removeAttribute('src'); video.load(); };
  }, [mediaSrc, timelineData]);

  // 에러 화면 (state 없음 포함) — 정상 화면 외의 유일한 분기
  const errorType = !videoId ? 'nostate' : error?.type;
  if (errorType) {
    const view = ERROR_VIEWS[errorType];
    const actionButtons = {
      home:  { label: '동영상 분석으로', onClick: () => navigate('/video-analysis') },
      login: { label: '로그인', onClick: () => navigate('/login') },
      retry: { label: '다시 시도', onClick: () => setRetryKey(k => k + 1) },
      back:  { label: '뒤로가기', onClick: () => navigate(-1) },
    };
    return (
      <TimelineLayout sessionUser={sessionUser}>
        <header className="dg-timeline__head">
          <button type="button" className="dg-timeline__back" onClick={() => navigate(-1)}>
            <span aria-hidden="true">←</span> 뒤로가기
          </button>
        </header>

        <Card className={`dg-card--static dg-timeline__error dg-timeline__error--${view.tone}`}>
          <div className="dg-timeline__error-icon" aria-hidden="true">!</div>
          <p className="dg-timeline__error-title">{view.title}</p>
          <p className="dg-timeline__error-desc">{view.desc(data)}</p>
          {error?.detail && <p className="dg-timeline__error-detail">{error.detail}</p>}
          <div className="dg-timeline__error-actions">
            {view.actions(data).map((key, i) => (
              <Button key={key} variant={i === 0 ? 'primary' : 'outline'} onClick={actionButtons[key].onClick}>
                {actionButtons[key].label}
              </Button>
            ))}
          </div>
        </Card>
      </TimelineLayout>
    );
  }

  const goHeatmap = (frame) => navigate('/heatmap', {
    state: {
      video_id: videoId,
      frame_index: frame.frame_index,
      timestamp: frame.timestamp,
      frame_time: frame.frame_time,
      fake_score: frame.fake_score,
      model_type: data.model_type || 'fast',
      version_type: data.version_type || 'v1',
      video_name: data.video_name,
      // 히트맵 화면에서 이전/다음 구간으로 바로 이동할 수 있게 구간 목록을 함께 전달
      frames: timelineData.map(f => ({ frame_index: f.frame_index, timestamp: f.timestamp, frame_time: f.frame_time, fake_score: f.fake_score })),
    }
  });

  const circumference = 2 * Math.PI * DONUT_R;
  // 영상 전체 판정: 상세 화면과 같은 기준(label → prob), 없으면 위조 구간 비율
  const prob = data.prob ?? data.score ?? data.result_prob;
  const isFakeVideo = data.label && data.label !== 'UNKNOWN'
    ? data.label.toUpperCase() === 'FAKE'
    : (prob != null && prob !== -1 ? prob > 0.5 : fakeRatio > 50);

  // 영상 길이: video 메타데이터 우선, 없으면 마지막 프레임 시각 + 샘플 간격으로 추정
  const times = timelineData.map(f => f.frame_time).filter(t => t != null);
  const step = times.length > 1 ? (times[times.length - 1] - times[0]) / (times.length - 1) : 0;
  const videoSec = duration || (times.length ? times[times.length - 1] + step : 0);
  const fakeSec = totalCount ? Math.round((fakeCount / totalCount) * videoSec) : 0;
  const ratioSentence = !fakeCount
    ? '동영상 내 위조 흔적이 보이지 않습니다'
    : (videoSec
      ? <>동영상 내 <b className="dg-timeline__tone--fake">{Math.max(fakeSec, 1)}초</b> 동안 위조 흔적이 보입니다</>
      : <>{totalCount}개 구간 중 <b className="dg-timeline__tone--fake">{fakeCount}개</b> 구간에서 위조 흔적이 보입니다</>);

  // 가장 의심되는 장면: 딥페이크 확률이 가장 높은 프레임 2개
  const topFrames = [...timelineData].sort((a, b) => b.fake_score - a.fake_score).slice(0, 2);

  return (
    <TimelineLayout sessionUser={sessionUser}>
      {/* 상단: 왼쪽 뒤로가기 + 제목, 오른쪽 분석 조건 태그 */}
      <header className="dg-timeline__head dg-timeline__head--title">
        <div className="dg-timeline__head-main">
          <button type="button" className="dg-timeline__back" onClick={() => navigate(-1)}>
            <span aria-hidden="true">←</span> 뒤로가기
          </button>
          <h2 className="dg-timeline__title">동영상 구간별 분석</h2>
        </div>
      </header>

      {/* 상단: 영상 | 오른쪽 패널 (A안 위조 비율 + D안 TOP 3) */}
      <div className="dg-timeline__grid">
        <Card className="dg-card--static dg-timeline__media-card">
          <div className="dg-timeline__media">
            <video ref={videoRef} src={mediaSrc} playsInline
              onClick={e => { const v = e.currentTarget; if (v.paused) v.play().catch(() => {}); else v.pause(); }}
              onLoadedMetadata={e => { if (Number.isFinite(e.currentTarget.duration)) setDuration(e.currentTarget.duration); }} />
          </div>
          {/* 영상 아래: 재생 컨트롤 + Deepfake / Real 구간 (드래그해서 초 단위 이동) */}
          <VideoSegmentTrack videoRef={videoRef} duration={videoSec} frames={timelineData} />
        </Card>

        <aside className="dg-timeline__side">
          <Card className="dg-card--static dg-timeline__ratio-card">
            <div className="dg-timeline__ratio">
              <svg width="172" height="172" viewBox="0 0 172 172" className="dg-timeline__donut" aria-hidden="true">
                <circle cx="86" cy="86" r={DONUT_R} fill="none" className="dg-timeline__donut-track" strokeWidth="16" />
                <circle cx="86" cy="86" r={DONUT_R} fill="none" stroke={COLOR_FAKE} strokeWidth="16"
                  strokeDasharray={`${(shownRatio / 100) * circumference} ${circumference}`}
                  strokeLinecap={fakeCount ? 'round' : 'butt'}
                  style={{ opacity: shownRatio > 0.05 ? 1 : 0 }}
                  transform="rotate(-90 86 86)" className="dg-timeline__donut-fake" />
                <text x="86" y="80" textAnchor="middle" dominantBaseline="central" className="dg-timeline__donut-text">
                  {shownRatio.toFixed(0)}%
                </text>
                <text x="86" y="110" textAnchor="middle" className="dg-timeline__donut-label">위조 구간</text>
              </svg>
              <div className="dg-timeline__verdict">
                <span className={`dg-timeline__verdict-badge dg-timeline__verdict-badge--${isFakeVideo ? 'fake' : 'real'}`}>
                  {isFakeVideo ? '딥페이크 의심' : '실제 영상으로 보여요'}
                </span>
                <p className="dg-timeline__verdict-count">{ratioSentence}</p>
              </div>
            </div>
          </Card>

          <section className="dg-timeline__top" aria-labelledby="tl-top-title">
            <h3 id="tl-top-title" className="dg-timeline__top-title">가장 의심되는 장면 TOP 2</h3>
            {isLoading ? (
              <div className="dg-timeline__top-empty">구간을 불러오는 중...</div>
            ) : topFrames.map(frame => {
              const tone = frame.fake_score > 50 ? 'fake' : 'real';
              return (
                <div key={frame.frame_index} className="dg-timeline__top-row">
                  <span className={`dg-timeline__top-mark dg-timeline__top-mark--${tone}`} aria-hidden="true" />
                  <span className="dg-timeline__top-info">
                    <b className="dg-timeline__top-time">{frame.timestamp}</b>
                    <span className="dg-timeline__top-meta">
                      딥페이크 확률: <b className={`dg-timeline__tone--${tone}`}>{Number(frame.fake_score).toFixed(1)}%</b>
                    </span>
                  </span>
                  <button type="button" className="dg-timeline__top-btn" onClick={() => goHeatmap(frame)}
                    aria-label={`${frame.timestamp} 장면 위조 흔적 보기`}>
                    위조 흔적 <span aria-hidden="true">→</span>
                  </button>
                </div>
              );
            })}
          </section>
        </aside>
      </div>

      {/* 하단: 타임라인 그래프 & 프레임 데이터 보드 */}
      <Card className="dg-card--static dg-timeline__board">
        <button type="button" className="dg-timeline__board-toggle" onClick={() => setIsGraphOpen(o => !o)}
          aria-expanded={isGraphOpen} aria-controls="tl-graph">
          <span className="dg-timeline__board-title">
            <span className="dg-timeline__eyebrow">타임라인 그래프</span>
            <span className="dg-timeline__board-desc">구간별 딥페이크 확률을 상세히 알 수 있어요</span>
          </span>
          <span className="dg-timeline__board-action">
            {isGraphOpen ? '접기' : '펼치기'}
            <svg viewBox="0 0 24 24" aria-hidden="true" className={isGraphOpen ? 'is-open' : ''}><path d="M6 9l6 6 6-6" /></svg>
          </span>
        </button>

        {isGraphOpen && (
        <div id="tl-graph" className="dg-timeline__board-body">
        <div className="dg-timeline__board-legend">
          <span><i className="dg-timeline__swatch dg-timeline__swatch--fake" />변조 의심 (50% 초과)</span>
          <span><i className="dg-timeline__swatch dg-timeline__swatch--normal" />정상 (50% 이하)</span>
        </div>

        {isLoading ? (
          <div className="dg-timeline__state dg-timeline__state--inline">
            <div className="dg-timeline__spinner" />
            <p>프레임별 수치를 계산하는 중...</p>
          </div>
        ) : (
          <div>
            {/* 차트 (E안): 50% 판정 기준선 + 구간별 그라데이션 막대 */}
            <div ref={chartRef} className="dg-timeline__chart">
              <div className="dg-timeline__threshold" aria-hidden="true">
                <span>판정 기준 50%</span>
              </div>

              <div className="dg-timeline__bars" onMouseLeave={() => setHoverIdx(null)}>
                {timelineData.map((frame, i) => {
                  const score = Number(frame.fake_score);
                  const tone = score > 50 ? 'fake' : 'real';
                  const h = Math.max(score, 3);
                  return (
                    <button
                      key={i}
                      type="button"
                      ref={el => { colRefs.current[i] = el; }}
                      className={`dg-timeline__bar-col${hoverIdx === i ? ' is-hover' : ''}`}
                      onMouseEnter={() => setHoverIdx(i)}
                      onFocus={() => setHoverIdx(i)}
                      onBlur={() => setHoverIdx(null)}
                      onClick={() => goHeatmap(frame)}
                      aria-label={`${frame.timestamp} 위조 확률 ${score.toFixed(1)}%, 위조 흔적 보기`}
                    >
                      <b className={`dg-timeline__bar-value dg-timeline__bar-value--${tone}`} style={{ bottom: `calc(${h}% + 6px)` }}>
                        {score.toFixed(1)}
                      </b>
                      <span className={`dg-timeline__bar dg-timeline__bar--${tone}`} style={{ height: `${h}%` }} />
                    </button>
                  );
                })}
              </div>

              {/* hover 툴팁: 장면 썸네일 + 시각/확률 + 히트맵 안내 */}
              {hoverIdx != null && timelineData[hoverIdx] && (() => {
                const frame = timelineData[hoverIdx];
                const colEl = colRefs.current[hoverIdx];
                const chartEl = chartRef.current;
                const TIP_W = 176;
                const GAP = 12;
                let left = 0;
                if (colEl && chartEl) {
                  const chartRect = chartEl.getBoundingClientRect();
                  const colRect = colEl.getBoundingClientRect();
                  const right = colRect.right - chartRect.left + 4;
                  // 기본은 막대 오른쪽, 공간이 없으면 왼쪽
                  left = right + TIP_W + GAP <= chartRect.width ? right : colRect.left - chartRect.left - 4 - TIP_W;
                  left = Math.max(left, GAP);
                }
                const tone = frame.fake_score > 50 ? 'fake' : 'real';
                const thumb = thumbs[frame.frame_index];
                return (
                  <div className="dg-timeline__preview" style={{ left, width: TIP_W }} aria-hidden="true">
                    {thumb
                      ? <img src={thumb} alt="" />
                      : <div className="dg-timeline__preview-empty">장면 불러오는 중...</div>}
                    <div className="dg-timeline__preview-info">
                      <span>{frame.timestamp}</span>
                      <b className={`dg-timeline__tone--${tone}`}>{Number(frame.fake_score).toFixed(1)}%</b>
                    </div>
                    <span className="dg-timeline__preview-cta">클릭해서 위조 흔적 보기</span>
                  </div>
                );
              })()}
            </div>

          </div>
        )}
        </div>
        )}
      </Card>
    </TimelineLayout>
  );
};

export default VideoTimelinePage;

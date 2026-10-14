import React, { useEffect, useState, useRef, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';

import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import Button from '../components/Button';
import Card from '../components/Card';
import './VideoTimelinePage.css';

// 차트 색 (tokens.css와 동일 값 — canvas는 CSS 변수를 직접 못 씀)
const COLOR_FAKE = '#ff4d4d';
const COLOR_REAL = '#16a34a';
const COLOR_LINE = '#206cfd';

// 공통 레이아웃: Navbar(light) + 본문 + Footer
const TimelineLayout = ({ sessionUser, children }) => (
  <div className="dg-timeline">
    <Navbar variant="light" sessionUser={sessionUser} />
    <main className="dg-timeline__main">
      <div className="dg-timeline__wrap">{children}</div>
    </main>
    <Footer />
  </div>
);

const normalizeFrame = (f, i) => {
  const raw = f.score ?? f.fake_score ?? 0;
  const fake_score = raw <= 1 ? raw * 100 : raw;
  const ts = f.frame_time != null
    ? new Date(f.frame_time * 1000).toISOString().substr(14, 5)
    : f.timestamp ?? `00:${String(i * 2).padStart(2, '0')}`;
  return { ...f, fake_score, frame_index: f.frame_index ?? i, timestamp: ts };
};

const VideoTimelinePage = ({ sessionUser }) => {
  const { state: data } = useLocation();
  const navigate = useNavigate();
  const [timelineData, setTimelineData] = useState([]);
  const [videoMeta, setVideoMeta]       = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  const videoId = data?.video_id || data?.id;
  const mediaLoc = data?.video_loc || '';
  const mediaSrc = mediaLoc.startsWith('blob') ? mediaLoc : mediaLoc;

  // Canvas는 막대 div 위에 absolute로 얹히므로
  // 막대 div 자체를 ref로 잡아서 크기/위치를 측정
  const canvasRef   = useRef(null);
  const barsRef     = useRef(null); // 막대들이 들어있는 flex div
  const barRefs     = useRef([]);   // 각 막대 div ref 배열

  useEffect(() => {
    const fetchTimelineDetails = async () => {
      if (!videoId) { setIsLoading(false); return; }
      try {
        const response = await axios.get(`/inference/video/${videoId}/detail`);
        const d = response.data;
        let frames = [];
        if (d?.frames?.length)        frames = d.frames;
        else if (d?.timeline?.length) frames = d.timeline;
        if (d?.meta) setVideoMeta(d.meta);
        if (frames.length) {
          setTimelineData(frames.map(normalizeFrame));
        } else {
          setTimelineData(Array.from({ length: 10 }, (_, i) => normalizeFrame({
            frame_index: i, frame_time: i * 2,
            score: (i >= 3 && i <= 6) ? 0.855 : 0.123,
          }, i)));
        }
      } catch {
        setTimelineData(Array.from({ length: 10 }, (_, i) => normalizeFrame({
          frame_index: i, frame_time: i * 2,
          score: (i >= 3 && i <= 6) ? 0.855 : 0.123,
        }, i)));
      } finally {
        setIsLoading(false);
      }
    };
    fetchTimelineDetails();
  }, [videoId]);

  // ── fake/real 비율 (프레임 카운트 기준) ──
  const fakeCount  = timelineData.filter(f => f.fake_score > 50).length;
  const totalCount = timelineData.length;
  const fakeRatio  = totalCount ? (fakeCount / totalCount) * 100 : 0;
  const realRatio  = 100 - fakeRatio;

  // ── Canvas 렌더링 ──
  // 핵심: 각 막대 div의 getBoundingClientRect()로 실제 화면 위치를 측정해
  //       캔버스 좌표로 변환 → 완벽하게 막대 상단 중앙을 통과하는 꺾은선
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const barsEl = barsRef.current;
    if (!canvas || !barsEl || timelineData.length < 2) return;
    if (barRefs.current.length !== timelineData.length) return;
    if (barRefs.current.some(r => !r)) return;

    const barsRect = barsEl.getBoundingClientRect();
    const W = barsRect.width;
    const H = barsRect.height;
    const dpr = window.devicePixelRatio || 1;

    canvas.width  = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width  = W + 'px';
    canvas.style.height = H + 'px';

    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    // 각 막대 상단 중앙 좌표 계산
    // barRef는 막대 div (색깔 있는 직사각형)
    const points = barRefs.current.map((barEl, i) => {
      const barRect = barEl.getBoundingClientRect();
      // canvas 기준 좌표 = barRect - barsRect (offset)
      const x = barRect.left - barsRect.left + barRect.width / 2;
      const y = barRect.top  - barsRect.top;  // 막대 상단
      return { x, y, isFake: timelineData[i].fake_score > 50 };
    });

    // 50% 기준선 y좌표: fake_score=50일 때의 막대 높이 비율로 계산
    // 막대 영역 높이 = barsEl에서 timestamp span 제외한 부분
    // barsEl padding-top=30px → 막대 영역 시작 y=30, 막대 바닥 y=H-23(timestamp)
    const BAR_TOP    = 30;  // padding-top
    const BAR_BOTTOM = H - 23; // timestamp 텍스트 영역 제외
    const BAR_H      = BAR_BOTTOM - BAR_TOP;
    const y50 = BAR_BOTTOM - (50 / 100) * BAR_H;

    // 50% 기준선
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = 'rgba(255, 77, 77, 0.45)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(points[0].x, y50);
    ctx.lineTo(points[points.length - 1].x, y50);
    ctx.stroke();
    ctx.restore();

    // 꺾은선 (밝은 배경용: primary 선 + 옅은 글로우)
    ctx.save();
    ctx.shadowColor = 'rgba(32, 108, 253, 0.35)';
    ctx.shadowBlur  = 8;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
    ctx.strokeStyle = COLOR_LINE;
    ctx.lineWidth   = 2;
    ctx.lineJoin    = 'round';
    ctx.lineCap     = 'round';
    ctx.stroke();
    ctx.restore();

    // 데이터 포인트 dot
    points.forEach((pt) => {
      // 외곽 글로우 원
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 7, 0, Math.PI * 2);
      ctx.fillStyle = pt.isFake ? 'rgba(255, 77, 77, 0.18)' : 'rgba(22, 163, 74, 0.16)';
      ctx.fill();
      // 내부 dot (흰 테두리로 막대와 분리)
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = pt.isFake ? COLOR_FAKE : COLOR_REAL;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();
    });
  }, [timelineData]);

  useEffect(() => {
    if (isLoading || !timelineData.length) return;
    // DOM 렌더 완료 후 실행 (두 번 대기로 안전하게)
    const t1 = setTimeout(drawCanvas, 100);
    const t2 = setTimeout(drawCanvas, 400);
    const ro = new ResizeObserver(() => setTimeout(drawCanvas, 50));
    if (barsRef.current) ro.observe(barsRef.current);
    return () => { clearTimeout(t1); clearTimeout(t2); ro.disconnect(); };
  }, [timelineData, isLoading, drawCanvas]);

  if (!data) {
    return (
      <TimelineLayout sessionUser={sessionUser}>
        <div className="dg-timeline__state">
          <p>전송된 비디오 분석 데이터가 없습니다.</p>
          <Button variant="primary" onClick={() => navigate(-1)}>뒤로가기</Button>
        </div>
      </TimelineLayout>
    );
  }

  const goHeatmap = (frame) => navigate('/heatmap', {
    state: {
      video_id: videoId,
      frame_index: frame.frame_index,
      timestamp: frame.timestamp,
      fake_score: frame.fake_score,
      model_type: data.model_type || 'fast',
      video_name: data.video_name,
    }
  });

  const circumference = 2 * Math.PI * 56;

  return (
    <TimelineLayout sessionUser={sessionUser}>
      {/* 상단: 뒤로가기 + 분석 조건 태그 */}
      <header className="dg-timeline__head">
        <button type="button" className="dg-timeline__back" onClick={() => navigate(-1)}>
          <span aria-hidden="true">←</span> 뒤로가기
        </button>
        <div className="dg-timeline__chips">
          {data.version_type && <span>{data.version_type.toUpperCase()}</span>}
          {data.domain_type && <span>{data.domain_type}</span>}
          {data.model_type && <span>{data.model_type.toUpperCase()}</span>}
        </div>
      </header>

      <h2 className="dg-timeline__title">프레임별 위조 분석</h2>
      <p className="dg-timeline__sub">구간별 변조 확률을 확인하고, 의심 구간은 히트맵으로 근거를 살펴보세요.</p>

      {/* 상단: 영상 + 위조 비율 */}
      <div className="dg-timeline__grid">
        <Card className="dg-card--static dg-timeline__media-card">
          <div className="dg-timeline__media">
            <video src={mediaSrc} controls />
          </div>
        </Card>

        <Card className="dg-card--static dg-timeline__ratio-card">
          <p className="dg-timeline__eyebrow">FORGERY RATIO</p>

          <div className="dg-timeline__ratio">
            {/* 도넛 차트 */}
            <svg width="140" height="140" viewBox="0 0 140 140" className="dg-timeline__donut">
              {/* real 전체 배경 링 */}
              <circle cx="70" cy="70" r="56" fill="none" stroke={COLOR_REAL} strokeWidth="16" />
              {/* fake 비율만큼 덮어쓰기 */}
              <circle
                cx="70" cy="70" r="56" fill="none"
                stroke={COLOR_FAKE} strokeWidth="16"
                strokeDasharray={`${(fakeRatio / 100) * circumference} ${circumference}`}
                strokeLinecap="butt"
                className="dg-timeline__donut-fake"
              />
              {/* 중앙 fake % 텍스트 (rotate 보정) */}
              <text
                x="70" y="70"
                textAnchor="middle" dominantBaseline="central"
                className="dg-timeline__donut-text"
              >
                {fakeRatio.toFixed(0)}%
              </text>
            </svg>

            {/* 범례 */}
            <div className="dg-timeline__legend">
              <div className="dg-timeline__legend-row">
                <i className="dg-timeline__swatch dg-timeline__swatch--fake" />
                <span>FAKE</span>
                <strong className="dg-timeline__tone--fake">{fakeRatio.toFixed(1)}%</strong>
              </div>
              <div className="dg-timeline__legend-row">
                <i className="dg-timeline__swatch dg-timeline__swatch--real" />
                <span>REAL</span>
                <strong className="dg-timeline__tone--real">{realRatio.toFixed(1)}%</strong>
              </div>
              <div className="dg-timeline__legend-count">
                {fakeCount} / {totalCount} FRAMES
              </div>
            </div>
          </div>

          {videoMeta && (
            <dl className="dg-timeline__meta">
              <div><dt>FPS</dt><dd>{Number(videoMeta.fps).toFixed(1)}</dd></div>
              <div><dt>전체 프레임</dt><dd>{videoMeta.total_frames}</dd></div>
              <div><dt>얼굴 탐지</dt><dd>{videoMeta.num_detected} / {videoMeta.num_sampled}</dd></div>
            </dl>
          )}
        </Card>
      </div>

      {/* 하단: 타임라인 그래프 & 프레임 데이터 보드 */}
      <Card className="dg-card--static dg-timeline__board">
        <div className="dg-timeline__board-head">
          <p className="dg-timeline__eyebrow">CHRONOLOGICAL FORGERY RISK</p>
          <div className="dg-timeline__board-legend">
            <span><i className="dg-timeline__swatch dg-timeline__swatch--fake" />변조 의심 (50% 초과)</span>
            <span><i className="dg-timeline__swatch dg-timeline__swatch--real" />정상</span>
          </div>
        </div>

        {isLoading ? (
          <div className="dg-timeline__state dg-timeline__state--inline">
            <div className="dg-timeline__spinner" />
            <p>프레임별 수치를 계산하는 중...</p>
          </div>
        ) : (
          <div>
            {/* 차트 영역: 막대 + Canvas 꺾은선 */}
            <div className="dg-timeline__chart">

              {/* 막대 그래프 (치수는 drawCanvas 기준선 계산과 맞물려 있으므로 유지) */}
              <div ref={barsRef} className="dg-timeline__bars">
                {timelineData.map((frame, i) => {
                  const isFakeUnit = frame.fake_score > 50;
                  return (
                    <div key={i} className="dg-timeline__bar-col">
                      {/* 막대 div — ref 배열로 각각 잡음 */}
                      <div
                        ref={el => barRefs.current[i] = el}
                        className={`dg-timeline__bar dg-timeline__bar--${isFakeUnit ? 'fake' : 'real'}`}
                        style={{ height: `${Math.max(frame.fake_score, 4)}%` }}
                      />
                      <span className="dg-timeline__bar-time">{frame.timestamp}</span>
                    </div>
                  );
                })}
              </div>

              {/* Canvas: barsRef와 완전히 동일한 위치/크기 */}
              <canvas ref={canvasRef} className="dg-timeline__canvas" />
            </div>

            {/* 구간별 히트맵 버튼 행 */}
            <div className="dg-timeline__heat-row">
              {timelineData.map((frame, i) => {
                const isFake = frame.fake_score > 50;
                return (
                  <div key={i} className="dg-timeline__heat-cell">
                    <button
                      type="button"
                      onClick={() => goHeatmap(frame)}
                      title={`${frame.timestamp} 히트맵`}
                      className={`dg-timeline__heat dg-timeline__heat--${isFake ? 'fake' : 'real'}`}
                    >
                      HEAT
                    </button>
                  </div>
                );
              })}
            </div>

            {/* 구간별 텍스트 그리드 */}
            <div className="dg-timeline__frames">
              {timelineData.map((frame, idx) => {
                const isFake = frame.fake_score > 50;
                const tone = isFake ? 'fake' : 'real';
                return (
                  <div key={idx} className="dg-timeline__frame">
                    <span className="dg-timeline__frame-time">TIMESTAMP {frame.timestamp}</span>
                    <div className="dg-timeline__frame-info">
                      <span className="dg-timeline__frame-prob">
                        변조 확률 <b className={`dg-timeline__tone--${tone}`}>{Number(frame.fake_score).toFixed(1)}%</b>
                      </span>
                      <span className={`dg-timeline__badge dg-timeline__badge--${tone}`}>
                        {isFake ? 'MANIPULATED' : 'VERIFIED'}
                      </span>
                      <button type="button" className="dg-timeline__frame-btn" onClick={() => goHeatmap(frame)}>
                        히트맵
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </Card>
    </TimelineLayout>
  );
};

export default VideoTimelinePage;

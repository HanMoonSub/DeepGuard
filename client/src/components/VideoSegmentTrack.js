import React, { useState, useEffect, useRef, useCallback } from 'react';
import './VideoSegmentTrack.css';

const SPEEDS = [0.5, 1, 1.5, 2];
const FAKE_THRESHOLD = 50;

// 초 → mm:ss
const clock = (sec) => {
  const s = Math.max(Math.floor(sec || 0), 0);
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
};

// 눈금 간격: 큰 눈금이 6개 이하가 되는 가장 작은 간격
const majorStep = (sec) => [1, 2, 5, 10, 15, 30, 60, 120, 300, 600].find(s => sec / s <= 6) || 1200;

// 프레임 i가 영상의 [i/N, (i+1)/N] 구간을 대표한다고 보고 같은 판정끼리 묶는다
const buildRuns = (frames, duration) => {
  const n = frames.length;
  const runs = [];
  frames.forEach((f, i) => {
    const fake = f.fake_score > FAKE_THRESHOLD;
    const last = runs[runs.length - 1];
    if (last && last.fake === fake) last.end = ((i + 1) / n) * duration;
    else runs.push({ fake, start: (i / n) * duration, end: ((i + 1) / n) * duration });
  });
  return runs;
};

const Icon = ({ d, fill }) => (
  <svg viewBox="0 0 24 24" aria-hidden="true" className={fill ? 'is-fill' : ''}><path d={d} /></svg>
);
const ICONS = {
  play: 'M8 5.5v13l10.5-6.5z',
  pause: 'M7 5h3.5v14H7zM13.5 5H17v14h-3.5z',
  prev: 'M6 5v14M18 5.5v13L8.5 12z',
  next: 'M18 5v14M6 5.5v13L15.5 12z',
  sound: 'M4 9.5h3.5L12 5.5v13l-4.5-4H4zM15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11',
  mute: 'M4 9.5h3.5L12 5.5v13l-4.5-4H4zM16 9.5l5 5M21 9.5l-5 5',
};

/**
 * 영상 아래 재생 컨트롤 + Deepfake / Real 구간 트랙.
 * 트랙(눈금 포함)을 클릭하거나 드래그하면 해당 초로 이동한다.
 */
const VideoSegmentTrack = ({ videoRef, duration, frames }) => {
  const areaRef = useRef(null);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [dragging, setDragging] = useState(false);

  // 영상 상태 동기화 (재생 중에는 rAF로 부드럽게)
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return undefined;
    let raf;
    const loop = () => { setTime(v.currentTime); raf = requestAnimationFrame(loop); };
    const onPlay = () => { setPlaying(true); cancelAnimationFrame(raf); raf = requestAnimationFrame(loop); };
    const onPause = () => { setPlaying(false); cancelAnimationFrame(raf); setTime(v.currentTime); };
    const onSync = () => setTime(v.currentTime);
    const onVolume = () => setMuted(v.muted);
    v.addEventListener('play', onPlay);
    v.addEventListener('pause', onPause);
    v.addEventListener('ended', onPause);
    v.addEventListener('seeked', onSync);
    v.addEventListener('timeupdate', onSync);
    v.addEventListener('volumechange', onVolume);
    return () => {
      cancelAnimationFrame(raf);
      v.removeEventListener('play', onPlay);
      v.removeEventListener('pause', onPause);
      v.removeEventListener('ended', onPause);
      v.removeEventListener('seeked', onSync);
      v.removeEventListener('timeupdate', onSync);
      v.removeEventListener('volumechange', onVolume);
    };
  }, [videoRef]);

  const seek = useCallback((sec) => {
    const v = videoRef.current;
    if (!v || !duration) return;
    const t = Math.min(Math.max(sec, 0), duration);
    v.currentTime = t;
    setTime(t);
  }, [videoRef, duration]);

  const seekFromPointer = (clientX) => {
    const rect = areaRef.current.getBoundingClientRect();
    seek(((clientX - rect.left) / rect.width) * duration);
  };

  const onPointerDown = (e) => {
    if (!duration) return;
    try { e.currentTarget.setPointerCapture(e.pointerId); } catch { /* 캡처 실패해도 클릭 이동은 동작 */ }
    setDragging(true);
    seekFromPointer(e.clientX);
  };
  const onPointerMove = (e) => { if (dragging) seekFromPointer(e.clientX); };
  const onPointerUp = () => setDragging(false);

  const togglePlay = () => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) v.play().catch(() => {}); else v.pause();
  };
  const toggleMute = () => { const v = videoRef.current; if (v) v.muted = !v.muted; };
  const cycleSpeed = () => {
    const next = SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length];
    if (videoRef.current) videoRef.current.playbackRate = next;
    setSpeed(next);
  };

  const runs = duration && frames.length ? buildRuns(frames, duration) : [];
  // 이전/다음 버튼: 판정이 바뀌는 구간 경계로 이동
  const bounds = runs.map(r => r.start);
  const goPrev = () => seek([...bounds].reverse().find(b => b < time - 0.25) ?? 0);
  const goNext = () => { const b = bounds.find(x => x > time + 0.05); if (b != null) seek(b); };

  const pct = (sec) => `${duration ? (sec / duration) * 100 : 0}%`;
  const major = majorStep(duration || 1);
  const minor = major / 4;
  const ticks = [];
  if (duration) for (let s = 0; s <= duration + 1e-6; s += minor) ticks.push({ s, major: Math.abs(s / major - Math.round(s / major)) < 1e-6 });

  const onKeyDown = (e) => {
    if (e.key === 'ArrowRight') { e.preventDefault(); seek(time + 1); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); seek(time - 1); }
    else if (e.key === 'Home') { e.preventDefault(); seek(0); }
    else if (e.key === 'End') { e.preventDefault(); seek(duration); }
  };

  const ROWS = [
    { key: 'fake', label: 'Deepfake', fake: true },
    { key: 'real', label: 'Real', fake: false },
  ];

  return (
    <div className="dg-vtrack">
      <div className="dg-vtrack__controls">
        <span className="dg-vtrack__clock">{clock(time)} <i>/</i> {clock(duration)}</span>
        <button type="button" className="dg-vtrack__btn" onClick={toggleMute} aria-label={muted ? '소리 켜기' : '소리 끄기'}>
          <Icon d={muted ? ICONS.mute : ICONS.sound} />
        </button>
        <button type="button" className="dg-vtrack__btn dg-vtrack__speed" onClick={cycleSpeed} aria-label={`재생 속도 ${speed}배`}>
          {speed}x
        </button>
        <span className="dg-vtrack__spacer" />
        <button type="button" className="dg-vtrack__btn" onClick={goPrev} aria-label="이전 구간">
          <Icon d={ICONS.prev} fill />
        </button>
        <button type="button" className="dg-vtrack__btn dg-vtrack__play" onClick={togglePlay} aria-label={playing ? '일시정지' : '재생'}>
          <Icon d={playing ? ICONS.pause : ICONS.play} fill />
        </button>
        <button type="button" className="dg-vtrack__btn" onClick={goNext} aria-label="다음 구간">
          <Icon d={ICONS.next} fill />
        </button>
      </div>

      <div className="dg-vtrack__body">
        <div className="dg-vtrack__labels">
          <span className="dg-vtrack__label-ruler" />
          {ROWS.map(r => (
            <span key={r.key} className={`dg-vtrack__label dg-vtrack__label--${r.key}`}>{r.label}</span>
          ))}
        </div>

        <div ref={areaRef} className={`dg-vtrack__area${dragging ? ' is-dragging' : ''}`}
          onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp}>
          <div className="dg-vtrack__ruler" aria-hidden="true">
            {ticks.map(t => (
              <span key={t.s} className={`dg-vtrack__tick${t.major ? ' is-major' : ''}`} style={{ left: pct(t.s) }}>
                {t.major && <em>{clock(t.s)}</em>}
              </span>
            ))}
          </div>
          {ROWS.map(r => (
            <div key={r.key} className={`dg-vtrack__row dg-vtrack__row--${r.key}`}>
              {runs.filter(run => run.fake === r.fake).map(run => (
                <span key={run.start} className="dg-vtrack__seg"
                  style={{ left: pct(run.start), width: pct(run.end - run.start) }}
                  title={`${clock(run.start)} – ${clock(run.end)}`} />
              ))}
            </div>
          ))}

          <div className="dg-vtrack__head" style={{ left: pct(time) }}>
            <span className="dg-vtrack__handle" role="slider" tabIndex={0}
              aria-label="재생 위치" aria-valuemin={0} aria-valuemax={Math.round(duration || 0)}
              aria-valuenow={Math.round(time)} aria-valuetext={`${clock(time)} / ${clock(duration)}`}
              onKeyDown={onKeyDown} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoSegmentTrack;

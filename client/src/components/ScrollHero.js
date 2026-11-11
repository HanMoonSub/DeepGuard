import React, { useEffect, useRef } from 'react';
import Button from './Button';
import './ScrollHero.css';

/*
 * Scroll-driven hero. The scene changes with the scroll position (p: 0~1).
 *   K1 intro photo → zoom-in video (clip) → K2 eye close-up photo → video in reverse → K3 landmark photo → K4 heatmap photo
 * K1, K3 and K4 share the same framing so they crossfade; only the zoom-in segment uses video frames.
 * The video starts from a different framing than the photos, so its first frame is positioned and scaled so its face frame
 * lines up with K1's face frame, then expands to full screen as it plays. Files are in public/hero/ (sources and how they were made: client/design/higgsfield-hero.md)
 */

const clamp = (v, a = 0, b = 1) => Math.min(b, Math.max(a, v));
const lerp = (a, b, t) => a + (b - a) * t;
const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
// 0~1 value that fades in over [a,b] and fades out over [c,d]
const band = (p, a, b, c, d) => {
  if (p < a || p > d) return 0;
  if (p < b) return (p - a) / (b - a);
  if (p <= c) return 1;
  return (d - p) / (d - c);
};

const BASE = `${process.env.PUBLIC_URL}/hero/`;

// Photo (K1~K4) size and anchor points (original pixel coordinates)
const STILL_W = 1672;
const STILL_H = 941;
const FACE = { x: 1283, y: 392, w: 366 }; // center and width of the face frame in K1, K3 and K4
const K2_EYE = { x: 844, y: 360 }; // K2 pupil center

// Zoom-in video frames (size and anchor points after cropping out the watermark)
const CLIP_N = 95;
const CLIP_PX = 2; // frame files are 2x the resolution of the coordinates below (cropped from a 1728×992 video)
const CLIP_W = 661;
const CLIP_H = 372;
const CLIP_FACE = { x: 293, y: 139, w: 167 }; // face frame in the first frame
const CLIP_EYE = { x: 343, y: 195 }; // pupil in the last frame

const STEPS = [
  { key: 'intro', range: [-1, 0, 0.1, 0.14] },
  { key: 'low', range: [0.36, 0.4, 0.48, 0.52] },
  { key: 'xai', range: [0.74, 0.79, 0.86, 0.9] },
  { key: 'verdict', range: [0.9, 0.94, 2, 3] },
];

const FAKE_SCORE = 99.3;

const loadImage = (src) => {
  const im = new Image();
  im.decoding = 'async';
  im.src = src;
  return im;
};
const ready = (im) => im && im.complete && im.naturalWidth > 0;

const ScrollHero = ({ onStart }) => {
  const secRef = useRef(null);
  const canvasRef = useRef(null);
  const stepRefs = useRef([]);
  const railRefs = useRef([]);
  const pctRef = useRef(null);
  const hintRef = useRef(null);

  useEffect(() => {
    const sec = secRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    const k = [1, 2, 3, 4].map((i) => loadImage(`${BASE}k${i}.webp`));
    const frames = Array.from({ length: CLIP_N }, (_, i) =>
      loadImage(`${BASE}clip/f_${String(i + 1).padStart(3, '0')}.webp`),
    );

    // Mask that softly fades out the video frame edges (so it doesn't look framed)
    const MW = CLIP_W * CLIP_PX;
    const MH = CLIP_H * CLIP_PX;
    const mask = document.createElement('canvas');
    mask.width = MW;
    mask.height = MH;
    const mctx = mask.getContext('2d');
    mctx.fillStyle = '#fff';
    mctx.fillRect(0, 0, MW, MH);
    mctx.globalCompositeOperation = 'destination-in';
    const gx = mctx.createLinearGradient(0, 0, MW, 0);
    gx.addColorStop(0, 'rgba(0,0,0,0)');
    gx.addColorStop(0.14, '#000');
    gx.addColorStop(0.86, '#000');
    gx.addColorStop(1, 'rgba(0,0,0,0)');
    mctx.fillStyle = gx;
    mctx.fillRect(0, 0, MW, MH);
    const gy = mctx.createLinearGradient(0, 0, 0, MH);
    gy.addColorStop(0, 'rgba(0,0,0,0)');
    gy.addColorStop(0.12, '#000');
    gy.addColorStop(0.88, '#000');
    gy.addColorStop(1, 'rgba(0,0,0,0)');
    mctx.fillStyle = gy;
    mctx.fillRect(0, 0, MW, MH);
    const off = document.createElement('canvas');
    off.width = MW;
    off.height = MH;
    const octx = off.getContext('2d');

    // Film grain that hides the smearing in the low-resolution video segment
    const grain = document.createElement('canvas');
    grain.width = 160;
    grain.height = 160;
    const gctx = grain.getContext('2d');
    const noise = gctx.createImageData(160, 160);
    for (let i = 0; i < noise.data.length; i += 4) {
      const v = Math.random() * 255;
      noise.data[i] = v;
      noise.data[i + 1] = v;
      noise.data[i + 2] = v;
      noise.data[i + 3] = 255;
    }
    gctx.putImageData(noise, 0, 0);
    let grainPattern = null;

    let w = 0;
    let h = 0;
    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      w = canvas.clientWidth;
      h = canvas.clientHeight;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.imageSmoothingQuality = 'high';
      grainPattern = ctx.createPattern(grain, 'repeat');
    };
    resize();
    window.addEventListener('resize', resize);

    // Fill the screen while keeping the anchor point (fx, fy) as close to the center as possible
    const cover = (iw, ih, fx, fy) => {
      const s = Math.max(w / iw, h / ih);
      return {
        s,
        x: clamp(w / 2 - fx * s, w - iw * s, 0),
        y: clamp(h / 2 - fy * s, h - ih * s, 0),
      };
    };

    const drawAt = (im, r, iw, ih, alpha) => {
      if (alpha <= 0 || !ready(im)) return;
      ctx.globalAlpha = alpha;
      ctx.drawImage(im, r.x, r.y, iw * r.s, ih * r.s);
      ctx.globalAlpha = 1;
    };

    const frameAt = (u) => {
      const idx = Math.round(clamp(u) * (CLIP_N - 1));
      for (let d = 0; d < CLIP_N; d += 1) {
        if (ready(frames[idx - d])) return frames[idx - d];
        if (ready(frames[idx + d])) return frames[idx + d];
      }
      return null;
    };

    const draw = (p) => {
      const still = cover(STILL_W, STILL_H, FACE.x, FACE.y);
      // Video start position: overlay the first frame's face frame on the face frame in the photo
      const k0 = (FACE.w * still.s) / CLIP_FACE.w;
      const r0 = {
        s: k0,
        x: still.x + FACE.x * still.s - CLIP_FACE.x * k0,
        y: still.y + FACE.y * still.s - CLIP_FACE.y * k0,
      };
      // Video end position: fill the screen around the pupil
      const r1 = cover(CLIP_W, CLIP_H, CLIP_EYE.x, CLIP_EYE.y);

      // Zoom in (0.12~0.36) → hold (~0.54) → reverse (~0.66)
      let u = 0;
      if (p < 0.36) u = (p - 0.12) / 0.24;
      else if (p < 0.54) u = 1;
      else u = 1 - (p - 0.54) / 0.12;
      // The video's zoom motion is concentrated in the first half, so give more scroll to the early frames
      u = clamp(u) ** 2;

      drawAt(k[0], still, STILL_W, STILL_H, p < 0.12 ? 1 : 1 - clamp((p - 0.12) / 0.03));

      if (p > 0.12 && p < 0.71) {
        const fr = frameAt(u);
        if (fr) {
          const r = { s: lerp(r0.s, r1.s, u), x: lerp(r0.x, r1.x, u), y: lerp(r0.y, r1.y, u) };
          octx.globalCompositeOperation = 'source-over';
          octx.clearRect(0, 0, MW, MH);
          octx.drawImage(fr, 0, 0, MW, MH);
          octx.globalCompositeOperation = 'destination-in';
          octx.drawImage(mask, 0, 0);
          ctx.globalAlpha = clamp((p - 0.12) / 0.015);
          ctx.drawImage(off, r.x, r.y, CLIP_W * r.s, CLIP_H * r.s);
          // A bit more grain the more of the screen the video covers (none during the photo segments)
          const gA = 0.07 * clamp((p - 0.13) / 0.05) * (1 - clamp((p - 0.64) / 0.06));
          if (gA > 0 && grainPattern) {
            ctx.globalAlpha = gA;
            ctx.globalCompositeOperation = 'overlay';
            ctx.fillStyle = grainPattern;
            // Shake the position every frame so it doesn't look like a smudge stuck to the screen
            const ox = reduce ? 0 : Math.floor(Math.random() * 160);
            const oy = reduce ? 0 : Math.floor(Math.random() * 160);
            ctx.translate(-ox, -oy);
            ctx.fillRect(0, 0, w + 160, h + 160);
            ctx.translate(ox, oy);
            ctx.globalCompositeOperation = 'source-over';
          }
          ctx.globalAlpha = 1;
        }
      }

      // K2: place it as close as possible to the pupil position of the last video frame (without zooming in too much) and approach slowly
      const k2A = band(p, 0.36, 0.4, 0.5, 0.54);
      if (k2A > 0) {
        const ex = r1.x + CLIP_EYE.x * r1.s;
        const ey = r1.y + CLIP_EYE.y * r1.s;
        const s = Math.max(w / STILL_W, h / STILL_H) * (1.08 + 0.06 * clamp((p - 0.38) / 0.14));
        const r = {
          s,
          x: clamp(ex - K2_EYE.x * s, w - STILL_W * s, 0),
          y: clamp(ey - K2_EYE.y * s, h - STILL_H * s, 0),
        };
        drawAt(k[1], r, STILL_W, STILL_H, k2A);
      }

      drawAt(k[2], still, STILL_W, STILL_H, clamp((p - 0.65) / 0.05));
      // K3 passes briefly without text and crossfades at length into the heatmap (K4)
      drawAt(k[3], still, STILL_W, STILL_H, ease(clamp((p - 0.68) / 0.1)));

      // Darkening for text readability (left on desktop / bottom on mobile)
      const g = w >= 900 ? ctx.createLinearGradient(0, 0, w * 0.6, 0) : ctx.createLinearGradient(0, h, 0, h * 0.35);
      g.addColorStop(0, 'rgba(5, 9, 18, 0.9)');
      g.addColorStop(1, 'rgba(5, 9, 18, 0)');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, w, h);
    };

    const updateDom = (p) => {
      let active = 0;
      STEPS.forEach((s, i) => {
        if (p >= s.range[0]) active = i;
        const el = stepRefs.current[i];
        if (!el) return;
        const a = band(p, ...s.range);
        el.style.opacity = a.toFixed(3);
        el.style.setProperty('--dg-sh-shift', `${((1 - a) * 24).toFixed(1)}px`);
        el.style.visibility = a > 0.01 ? 'visible' : 'hidden';
        el.style.pointerEvents = a > 0.6 ? 'auto' : 'none';
      });
      railRefs.current.forEach((el, i) => el && el.classList.toggle('is-on', i === active));
      if (pctRef.current) {
        const v = FAKE_SCORE * ease(clamp((p - 0.9) / 0.06));
        pctRef.current.textContent = `${v.toFixed(1)}%`;
      }
      if (hintRef.current) hintRef.current.style.opacity = String(1 - clamp(p / 0.05));
    };

    const target = () => {
      const r = sec.getBoundingClientRect();
      const total = sec.offsetHeight - window.innerHeight;
      return total > 0 ? clamp(-r.top / total) : 0;
    };

    let alive = true;
    let p = target();
    let raf = null;
    let visible = true;
    const tick = () => {
      raf = null;
      if (!alive || !visible) return;
      const tp = target();
      p = reduce ? tp : p + (tp - p) * 0.14;
      if (Math.abs(tp - p) < 1e-4) p = tp;
      ctx.clearRect(0, 0, w, h);
      draw(p);
      updateDom(p);
      raf = requestAnimationFrame(tick);
    };
    const io = new IntersectionObserver(([e]) => {
      visible = e.isIntersecting;
      if (visible && !raf) raf = requestAnimationFrame(tick);
    });
    io.observe(sec);
    raf = requestAnimationFrame(tick);

    return () => {
      alive = false;
      io.disconnect();
      window.removeEventListener('resize', resize);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);

  const setStep = (i) => (el) => {
    stepRefs.current[i] = el;
  };

  return (
    <section className="dg-sh" ref={secRef} aria-label="DeepGuard 탐지 과정 소개">
      <div className="dg-sh__stage">
        <canvas ref={canvasRef} className="dg-sh__canvas" aria-hidden="true" />

        <div className="dg-sh__copy">
          <div className="dg-sh__step" ref={setStep(0)}>
            <h1 className="dg-sh__title">
              이 얼굴,<br /><em>진짜일까요?</em>
            </h1>
            <p className="dg-sh__lead">
              스크롤을 내려 DeepGuard가 위조 흔적을 찾아내는 과정을 따라가 보세요.
            </p>
          </div>

          <div className="dg-sh__step" ref={setStep(1)}>
            <h2 className="dg-sh__heading">픽셀 하나까지<br />들여다봅니다</h2>
            <p className="dg-sh__lead">
              텍스처, 합성 경계, 압축 흔적처럼 눈으로는 보이지 않는 미세한 위조 흔적을 찾습니다.
            </p>
          </div>

          <div className="dg-sh__step" ref={setStep(2)}>
            <h2 className="dg-sh__heading">딥페이크,<br /><em>눈으로 확인</em>하세요</h2>
            <p className="dg-sh__lead">
              어디가, 왜 가짜인지 보여드립니다. 히트맵으로 시각화해 위조 흔적을 한눈에 볼 수 있습니다.
            </p>
          </div>

          <div className="dg-sh__step dg-sh__verdict" ref={setStep(3)}>
            <div className="dg-sh__score">
              <b>FAKE</b>
              <span ref={pctRef}>0.0%</span>
            </div>
            <p className="dg-sh__lead">이미지와 영상 속 위조 흔적을 탐지하고, 어디가 왜 가짜인지 시각적 근거까지 제공합니다.</p>
            <div className="dg-hero__cta">
              <Button variant="primary" href="/analysis" onClick={onStart}>지금 분석해보기</Button>
            </div>
          </div>
        </div>

        <ol className="dg-sh__rail" aria-hidden="true">
          {STEPS.map((s, i) => (
            <li
              key={s.key}
              ref={(el) => {
                railRefs.current[i] = el;
              }}
            />
          ))}
        </ol>

        <div className="dg-sh__hint" ref={hintRef} aria-hidden="true">
          <span className="dg-sh__mouse" />
          SCROLL
        </div>
      </div>
    </section>
  );
};

export default ScrollHero;

import React, { useState, useEffect, useRef, useCallback } from 'react';
import './HeatmapZoomImage.css';

// The server (explainability/utils/image.py show_bbox_on_image) draws boxes as 1px lines in RGB (r, g, 30).
// One of r and g is always 255. The response has no coordinates, so we find rectangle borders of this color in the PNG pixels.
// To find boxes even when they overlap, are clipped at the image edge, or are partly hidden by another box's label,
// rectangles are rebuilt from the corners where horizontal and vertical segments of the same color meet.
const GAP = 2;  // treat gaps as small as another line crossing through as continuous

// Box color key per pixel (r * 256 + g + 1), or 0 if the pixel is not a box color
const colorKeys = (d, w, h) => {
  const keys = new Int32Array(w * h);
  for (let p = 0; p < w * h; p += 1) {
    const i = p * 4;
    if (d[i + 2] === 30 && d[i + 3] === 255 && (d[i] === 255 || d[i + 1] === 255)) keys[p] = d[i] * 256 + d[i + 1] + 1;
  }
  return keys;
};

// List of same-color segments [a, b] in one line (horizontal or vertical). at(i) is the color key of the i-th pixel in that line
const runsOf = (len, at) => {
  const runs = [];
  const open = new Map(); // color key → segment in progress
  for (let i = 0; i < len; i += 1) {
    const k = at(i);
    if (k) {
      const r = open.get(k);
      if (r && i - r.b - 1 <= GAP) r.b = i;
      else { if (r && r.b > r.a) runs.push(r); open.set(k, { k, a: i, b: i }); }
    }
  }
  open.forEach(r => { if (r.b > r.a) runs.push(r); });
  return runs;
};

const findBoxes = (d, w, h) => {
  const keys = colorKeys(d, w, h);
  const hs = [];
  for (let y = 0; y < h; y += 1) runsOf(w, x => keys[y * w + x]).forEach(r => hs.push({ ...r, y }));
  // Index of vertical segment start/end points: 'color:x:y' → segment
  const vStart = new Map();
  const vEnd = new Map();
  for (let x = 0; x < w; x += 1) {
    runsOf(h, y => keys[y * w + x]).forEach(r => {
      vStart.set(`${r.k}:${x}:${r.a}`, r);
      vEnd.set(`${r.k}:${x}:${r.b}`, r);
    });
  }
  const near = (map, k, x, y) => {
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        const r = map.get(`${k}:${x + dx}:${y + dy}`);
        if (r) return r;
      }
    }
    return null;
  };

  // Build a candidate rectangle if at least one of the four corners remains
  const cands = [];
  hs.forEach(({ k, a, b, y }) => {
    [a, b].forEach(x => {
      const down = near(vStart, k, x, y); // top corner
      if (down) cands.push({ k, x0: a, x1: b, y0: y, y1: down.b });
      const up = near(vEnd, k, x, y);     // bottom corner
      if (up) cands.push({ k, x0: a, x1: b, y0: up.a, y1: y });
    });
  });

  // Merge candidates from the same box (including ones shortened by being hidden)
  const same = (p, q) => {
    if (p.k !== q.k) return false;
    const close = (u, v) => Math.abs(u - v) <= 1;
    if (close(p.y0, q.y0) && close(p.y1, q.y1)) return true; // same top and bottom edges
    if (close(p.x0, q.x0) && close(p.x1, q.x1)) return true; // same left and right edges
    const ix = Math.min(p.x1, q.x1) - Math.max(p.x0, q.x0) + 1;
    const iy = Math.min(p.y1, q.y1) - Math.max(p.y0, q.y0) + 1;
    if (ix <= 0 || iy <= 0) return false;
    const area = r => (r.x1 - r.x0 + 1) * (r.y1 - r.y0 + 1);
    return (ix * iy) / Math.min(area(p), area(q)) >= 0.7;   // one almost fits inside the other
  };
  const boxes = [];
  cands.forEach(c => {
    let cur = { ...c };
    for (let i = boxes.length - 1; i >= 0; i -= 1) {
      if (same(boxes[i], cur)) {
        const o = boxes.splice(i, 1)[0];
        cur = { k: cur.k, x0: Math.min(o.x0, cur.x0), x1: Math.max(o.x1, cur.x1), y0: Math.min(o.y0, cur.y0), y1: Math.max(o.y1, cur.y1) };
        i = boxes.length; // compare again with the enlarged rectangle
      }
    }
    boxes.push(cur);
  });

  // Top to bottom, left to right
  return boxes
    .map(({ k, x0, x1, y0, y1 }) => ({
      x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1,
      color: `rgb(${Math.floor((k - 1) / 256)}, ${(k - 1) % 256}, 30)`,
    }))
    .sort((p, q) => (p.y - q.y) || (p.x - q.x));
};

const detectBoxes = (img) => {
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  return findBoxes(ctx.getImageData(0, 0, w, h).data, w, h);
};

// Zoom area including padding around the box (kept inside the image)
const cropOf = (box, nw, nh) => {
  const pad = Math.max(Math.max(box.w, box.h) * 0.35, 12);
  const x = Math.max(box.x - pad, 0);
  const y = Math.max(box.y - pad, 0);
  return { x, y, w: Math.min(box.x + box.w + pad, nw) - x, h: Math.min(box.y + box.h + pad, nh) - y };
};

/**
 * Forgery result image. If the result has bboxes drawn, finds the boxes and shows a highlight animation at their position;
 * if zoomable, a box can be clicked to view that part zoomed in.
 * @param {boolean} hasBoxes - true for the bbox / heatmap_bbox display
 * @param {boolean} zoomable - whether click-to-zoom on boxes is enabled (local structure only, since its artifacts are small)
 */
const HeatmapZoomImage = ({ src, alt, hasBoxes, zoomable, onError }) => {
  const wrapRef = useRef(null);
  // Store which src the load result belongs to.
  // (For cached images onLoad can fire before the src-change effect, so resetting in an effect would wipe the detection result)
  const [loaded, setLoaded] = useState({ src: null, natural: null, boxes: [] });
  const [frame, setFrame] = useState(null);       // container size { w, h }
  const [active, setActive] = useState(null);     // index of the box being zoomed

  const isCurrent = loaded.src === src;
  const natural = isCurrent ? loaded.natural : null;          // { w, h }
  const boxes = isCurrent && hasBoxes ? loaded.boxes : [];
  const canZoom = zoomable && boxes.length > 0;

  // Close the zoom view when the image or display mode changes
  useEffect(() => { setActive(null); }, [src, hasBoxes, zoomable]);

  const handleLoad = (e) => {
    const img = e.currentTarget;
    let found = [];
    try {
      found = detectBoxes(img);
    } catch {
      found = []; // if pixels cannot be read (e.g. cross-origin image), only disable the animation and zoom
    }
    setLoaded({ src: img.getAttribute('src'), natural: { w: img.naturalWidth, h: img.naturalHeight }, boxes: found });
  };

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return undefined;
    const measure = () => setFrame({ w: el.clientWidth, h: el.clientHeight });
    measure(); // the first ResizeObserver callback can be delayed until paint, so measure once right away
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const close = useCallback(() => setActive(null), []);
  const step = useCallback((dir) => setActive(i => (i + dir + boxes.length) % boxes.length), [boxes.length]);

  useEffect(() => {
    if (active === null) return undefined;
    const onKey = (e) => {
      if (e.key === 'Escape') close();
      else if (e.key === 'ArrowRight' && boxes.length > 1) step(1);
      else if (e.key === 'ArrowLeft' && boxes.length > 1) step(-1);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [active, boxes.length, close, step]);

  // Actual image area drawn with object-fit: contain
  const fit = natural && frame ? (() => {
    const s = Math.min(frame.w / natural.w, frame.h / natural.h);
    return { s, left: (frame.w - natural.w * s) / 2, top: (frame.h - natural.h * s) / 2 };
  })() : null;

  // Zoom panel: fit the area to the panel (86% x 72% of the container) and show it cropped as a background image
  let zoom = null;
  if (canZoom && active !== null && natural && frame && boxes[active]) {
    const c = cropOf(boxes[active], natural.w, natural.h);
    const s = Math.min((frame.w * 0.86) / c.w, (frame.h * 0.72) / c.h);
    zoom = {
      scale: s / (fit ? fit.s : 1),
      style: {
        width: c.w * s,
        height: c.h * s,
        backgroundImage: `url("${src}")`,
        backgroundSize: `${natural.w * s}px ${natural.h * s}px`,
        backgroundPosition: `${-c.x * s}px ${-c.y * s}px`,
      },
    };
  }

  return (
    <div ref={wrapRef} className="dg-zoomimg">
      <img src={src} alt={alt} onLoad={handleLoad} onError={onError} />

      {fit && canZoom && (
        <p className="dg-zoomimg__hint">
          <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7" /><path d="M16.5 16.5 21 21M11 8v6M8 11h6" /></svg>
          박스를 클릭하면 확대해서 볼 수 있어요
        </p>
      )}

      {fit && boxes.map((b, i) => {
        const style = {
          left: fit.left + b.x * fit.s,
          top: fit.top + b.y * fit.s,
          width: b.w * fit.s,
          height: b.h * fit.s,
          '--box-color': b.color,
        };
        return canZoom ? (
          <button key={`${src}-${b.x}-${b.y}`} type="button" className="dg-zoomimg__spot is-zoomable" style={style}
            onClick={() => setActive(i)} aria-label={`위조 의심 영역 ${i + 1} 확대해서 보기`} />
        ) : (
          <span key={`${src}-${b.x}-${b.y}`} className="dg-zoomimg__spot" style={style} aria-hidden="true" />
        );
      })}

      {zoom && (
        <div className="dg-zoomimg__panel" role="dialog" aria-modal="true"
          aria-label={`위조 의심 영역 ${active + 1} 확대`} onClick={close}>
          <div className="dg-zoomimg__panel-head" onClick={e => e.stopPropagation()}>
            <span className="dg-zoomimg__panel-title">
              위조 의심 영역 <b>{active + 1}</b>{boxes.length > 1 && <> / {boxes.length}</>}
              <em>×{zoom.scale.toFixed(1)}</em>
            </span>
            <button type="button" className="dg-zoomimg__close" onClick={close} aria-label="확대 닫기">✕</button>
          </div>
          <div key={active} className="dg-zoomimg__view" style={zoom.style} onClick={e => e.stopPropagation()} />
          <div className="dg-zoomimg__panel-foot" onClick={e => e.stopPropagation()}>
            {boxes.length > 1 && (
              <button type="button" className="dg-zoomimg__nav" onClick={() => step(-1)} aria-label="이전 영역">‹ 이전</button>
            )}
            <span>바깥을 누르거나 Esc로 닫을 수 있어요</span>
            {boxes.length > 1 && (
              <button type="button" className="dg-zoomimg__nav" onClick={() => step(1)} aria-label="다음 영역">다음 ›</button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default HeatmapZoomImage;

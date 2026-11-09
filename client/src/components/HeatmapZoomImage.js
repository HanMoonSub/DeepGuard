import React, { useState, useEffect, useRef, useCallback } from 'react';
import './HeatmapZoomImage.css';

// 서버(explainability/utils/image.py show_bbox_on_image)는 박스를 RGB (r, g, 30) 1px 선으로 그린다.
// r, g 중 하나는 항상 255. 응답에 좌표가 없으므로 PNG 픽셀에서 이 색의 사각형 테두리를 찾는다.
const isBoxPixel = (d, i) => d[i + 2] === 30 && d[i + 3] === 255 && (d[i] === 255 || d[i + 1] === 255);

const MIN_SIDE = 6;      // 이보다 작은 덩어리는 노이즈로 본다
const EDGE_COVER = 0.6;  // 왼쪽/오른쪽/아래 테두리가 이 비율 이상 이어져 있어야 사각형으로 인정 (위쪽은 확률 라벨에 가려질 수 있음)

const detectBoxes = (img) => {
  const w = img.naturalWidth;
  const h = img.naturalHeight;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const d = ctx.getImageData(0, 0, w, h).data;

  const mask = new Uint8Array(w * h);
  for (let p = 0; p < w * h; p += 1) if (isBoxPixel(d, p * 4)) mask[p] = 1;

  const seen = new Uint8Array(w * h);
  const boxes = [];
  const stack = [];
  for (let start = 0; start < w * h; start += 1) {
    if (!mask[start] || seen[start]) continue;
    // 8방향 연결 덩어리의 경계 상자
    let x0 = w, y0 = h, x1 = -1, y1 = -1;
    seen[start] = 1;
    stack.push(start);
    while (stack.length) {
      const p = stack.pop();
      const x = p % w, y = (p - x) / w;
      if (x < x0) x0 = x; if (x > x1) x1 = x;
      if (y < y0) y0 = y; if (y > y1) y1 = y;
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dx = -1; dx <= 1; dx += 1) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          const q = ny * w + nx;
          if (mask[q] && !seen[q]) { seen[q] = 1; stack.push(q); }
        }
      }
    }
    const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
    if (bw < MIN_SIDE || bh < MIN_SIDE) continue;
    let left = 0, right = 0, bottom = 0;
    for (let y = y0; y <= y1; y += 1) { left += mask[y * w + x0]; right += mask[y * w + x1]; }
    for (let x = x0; x <= x1; x += 1) bottom += mask[y1 * w + x];
    if (left / bh < EDGE_COVER || right / bh < EDGE_COVER || bottom / bw < EDGE_COVER) continue;
    boxes.push({ x: x0, y: y0, w: bw, h: bh });
  }
  // 위→아래, 왼→오른 순서로 번호를 매긴다
  return boxes.sort((a, b) => (a.y - b.y) || (a.x - b.x));
};

// 박스 주변 여백을 포함한 확대 영역 (이미지 밖으로 나가지 않게)
const cropOf = (box, nw, nh) => {
  const pad = Math.max(Math.max(box.w, box.h) * 0.35, 12);
  const x = Math.max(box.x - pad, 0);
  const y = Math.max(box.y - pad, 0);
  return { x, y, w: Math.min(box.x + box.w + pad, nw) - x, h: Math.min(box.y + box.h + pad, nh) - y };
};

/**
 * 위조 흔적 결과 이미지. bbox가 그려진 결과면 박스를 찾아 클릭 시 그 부분을 확대해서 보여준다.
 * @param {boolean} zoomable - bbox / heatmap_bbox 표시일 때 true
 */
const HeatmapZoomImage = ({ src, alt, zoomable, onError }) => {
  const wrapRef = useRef(null);
  // 로드 결과는 어떤 src의 것인지 함께 저장한다.
  // (캐시된 이미지는 onLoad가 src 변경 effect보다 먼저 올 수 있어, effect로 초기화하면 검출 결과가 지워짐)
  const [loaded, setLoaded] = useState({ src: null, natural: null, boxes: [] });
  const [frame, setFrame] = useState(null);       // 컨테이너 크기 { w, h }
  const [active, setActive] = useState(null);     // 확대 중인 박스 index

  const isCurrent = loaded.src === src;
  const natural = isCurrent ? loaded.natural : null;          // { w, h }
  const boxes = isCurrent && zoomable ? loaded.boxes : [];

  // 이미지나 표시 방식이 바뀌면 확대 창을 닫는다
  useEffect(() => { setActive(null); }, [src, zoomable]);

  const handleLoad = (e) => {
    const img = e.currentTarget;
    let found = [];
    try {
      found = detectBoxes(img);
    } catch {
      found = []; // 다른 출처 이미지 등으로 픽셀을 읽을 수 없으면 확대 기능만 끈다
    }
    setLoaded({ src: img.getAttribute('src'), natural: { w: img.naturalWidth, h: img.naturalHeight }, boxes: found });
  };

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return undefined;
    const measure = () => setFrame({ w: el.clientWidth, h: el.clientHeight });
    measure(); // ResizeObserver 첫 콜백은 화면이 그려질 때까지 늦어질 수 있어 바로 한 번 잰다
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

  // object-fit: contain 으로 그려진 실제 이미지 영역
  const fit = natural && frame ? (() => {
    const s = Math.min(frame.w / natural.w, frame.h / natural.h);
    return { s, left: (frame.w - natural.w * s) / 2, top: (frame.h - natural.h * s) / 2 };
  })() : null;

  const showBoxes = zoomable && fit && boxes.length > 0;

  // 확대 패널: 영역을 패널(컨테이너의 86% x 72%)에 맞춰 배경 이미지로 잘라 보여준다
  let zoom = null;
  if (active !== null && natural && frame && boxes[active]) {
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

      {showBoxes && (
        <>
          <p className="dg-zoomimg__hint">
            <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7" /><path d="M16.5 16.5 21 21M11 8v6M8 11h6" /></svg>
            박스를 클릭하면 확대해서 볼 수 있어요
          </p>
          {boxes.map((b, i) => (
            <button key={`${b.x}-${b.y}`} type="button" className="dg-zoomimg__spot"
              style={{
                left: fit.left + b.x * fit.s,
                top: fit.top + b.y * fit.s,
                width: b.w * fit.s,
                height: b.h * fit.s,
              }}
              onClick={() => setActive(i)}
              aria-label={`위조 의심 영역 ${i + 1} 확대해서 보기`} />
          ))}
        </>
      )}

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

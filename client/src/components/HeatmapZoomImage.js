import React, { useState, useEffect, useRef, useCallback } from 'react';
import './HeatmapZoomImage.css';

// 서버(explainability/utils/image.py show_bbox_on_image)는 박스를 RGB (r, g, 30) 1px 선으로 그린다.
// r, g 중 하나는 항상 255. 응답에 좌표가 없으므로 PNG 픽셀에서 이 색의 사각형 테두리를 찾는다.
// 박스가 겹치거나, 이미지 끝에서 잘리거나, 다른 박스의 라벨에 일부 가려져도 찾을 수 있도록
// 같은 색의 가로·세로 선분이 만나는 모서리를 기준으로 사각형을 복원한다.
const GAP = 2;  // 다른 선이 지나가며 끊긴 정도의 틈은 이어서 본다

// 픽셀마다 박스 색 키(r * 256 + g + 1), 박스 색이 아니면 0
const colorKeys = (d, w, h) => {
  const keys = new Int32Array(w * h);
  for (let p = 0; p < w * h; p += 1) {
    const i = p * 4;
    if (d[i + 2] === 30 && d[i + 3] === 255 && (d[i] === 255 || d[i + 1] === 255)) keys[p] = d[i] * 256 + d[i + 1] + 1;
  }
  return keys;
};

// 한 줄(가로 또는 세로)에서 같은 색 선분 [a, b] 목록. at(i)는 그 줄 i번째 픽셀의 색 키
const runsOf = (len, at) => {
  const runs = [];
  const open = new Map(); // 색 키 → 진행 중인 선분
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
  // 세로 선분의 시작/끝 점 색인: '색:x:y' → 선분
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

  // 네 모서리 중 하나라도 남아 있으면 후보 사각형을 만든다
  const cands = [];
  hs.forEach(({ k, a, b, y }) => {
    [a, b].forEach(x => {
      const down = near(vStart, k, x, y); // 위쪽 모서리
      if (down) cands.push({ k, x0: a, x1: b, y0: y, y1: down.b });
      const up = near(vEnd, k, x, y);     // 아래쪽 모서리
      if (up) cands.push({ k, x0: a, x1: b, y0: up.a, y1: y });
    });
  });

  // 같은 박스에서 나온 후보(가려져 짧아진 것 포함)를 합친다
  const same = (p, q) => {
    if (p.k !== q.k) return false;
    const close = (u, v) => Math.abs(u - v) <= 1;
    if (close(p.y0, q.y0) && close(p.y1, q.y1)) return true; // 위·아래 변이 같다
    if (close(p.x0, q.x0) && close(p.x1, q.x1)) return true; // 왼·오른 변이 같다
    const ix = Math.min(p.x1, q.x1) - Math.max(p.x0, q.x0) + 1;
    const iy = Math.min(p.y1, q.y1) - Math.max(p.y0, q.y0) + 1;
    if (ix <= 0 || iy <= 0) return false;
    const area = r => (r.x1 - r.x0 + 1) * (r.y1 - r.y0 + 1);
    return (ix * iy) / Math.min(area(p), area(q)) >= 0.7;   // 한쪽이 다른 쪽에 거의 들어간다
  };
  const boxes = [];
  cands.forEach(c => {
    let cur = { ...c };
    for (let i = boxes.length - 1; i >= 0; i -= 1) {
      if (same(boxes[i], cur)) {
        const o = boxes.splice(i, 1)[0];
        cur = { k: cur.k, x0: Math.min(o.x0, cur.x0), x1: Math.max(o.x1, cur.x1), y0: Math.min(o.y0, cur.y0), y1: Math.max(o.y1, cur.y1) };
        i = boxes.length; // 커진 사각형으로 다시 비교
      }
    }
    boxes.push(cur);
  });

  // 위→아래, 왼→오른 순서
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

// 박스 주변 여백을 포함한 확대 영역 (이미지 밖으로 나가지 않게)
const cropOf = (box, nw, nh) => {
  const pad = Math.max(Math.max(box.w, box.h) * 0.35, 12);
  const x = Math.max(box.x - pad, 0);
  const y = Math.max(box.y - pad, 0);
  return { x, y, w: Math.min(box.x + box.w + pad, nw) - x, h: Math.min(box.y + box.h + pad, nh) - y };
};

/**
 * 위조 흔적 결과 이미지. bbox가 그려진 결과면 박스를 찾아 그 위치에 강조 애니메이션을 보여주고,
 * zoomable이면 박스를 클릭해 그 부분을 확대해서 볼 수 있다.
 * @param {boolean} hasBoxes - bbox / heatmap_bbox 표시일 때 true
 * @param {boolean} zoomable - 박스 클릭 확대 사용 여부 (국소 구조일 때만: 위조 흔적이 작아서)
 */
const HeatmapZoomImage = ({ src, alt, hasBoxes, zoomable, onError }) => {
  const wrapRef = useRef(null);
  // 로드 결과는 어떤 src의 것인지 함께 저장한다.
  // (캐시된 이미지는 onLoad가 src 변경 effect보다 먼저 올 수 있어, effect로 초기화하면 검출 결과가 지워짐)
  const [loaded, setLoaded] = useState({ src: null, natural: null, boxes: [] });
  const [frame, setFrame] = useState(null);       // 컨테이너 크기 { w, h }
  const [active, setActive] = useState(null);     // 확대 중인 박스 index

  const isCurrent = loaded.src === src;
  const natural = isCurrent ? loaded.natural : null;          // { w, h }
  const boxes = isCurrent && hasBoxes ? loaded.boxes : [];
  const canZoom = zoomable && boxes.length > 0;

  // 이미지나 표시 방식이 바뀌면 확대 창을 닫는다
  useEffect(() => { setActive(null); }, [src, hasBoxes, zoomable]);

  const handleLoad = (e) => {
    const img = e.currentTarget;
    let found = [];
    try {
      found = detectBoxes(img);
    } catch {
      found = []; // 다른 출처 이미지 등으로 픽셀을 읽을 수 없으면 애니메이션·확대만 끈다
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

  // 확대 패널: 영역을 패널(컨테이너의 86% x 72%)에 맞춰 배경 이미지로 잘라 보여준다
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

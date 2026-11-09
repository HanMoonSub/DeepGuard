import React from 'react';
import Button from './Button';

// 이미지/프레임 위조 흔적 화면이 함께 쓰는 '시각화 설정' 패널 (스타일: pages/ImageHeatmapPage.css의 dg-ihm__*)
// 값은 draft에만 반영하고 '적용하기'를 눌러야 다시 그린다

const BRANCH_HINT = { low: '국소 흔적', high: '전역 구조', both: '국소 + 전역' };
const BRANCH_NAME = { low: '국소적', high: '전역적', both: '통합' };
const BRANCH_SUB = { low: '피부 질감', high: '얼굴 기하 구조', both: '질감 + 구조' };
const DISPLAY_OPTIONS = [
  { value: 'heatmap',      label: 'Heatmap',        desc: '위조 의심 영역을 색으로' },
  { value: 'bbox',         label: 'BBox',           desc: '위조 의심 영역을 박스로' },
  { value: 'heatmap_bbox', label: 'Heatmap + BBox', desc: '색과 박스를 함께' },
];
// 위조 흔적 임계값은 서버(explainability)에서 branch별 고정값을 쓴다 (전역 0.8 / 국소 0.95)
export const DEFAULT_DISPLAY = { display_type: 'heatmap_bbox', overlay_ratio: 0.8 };

// 0.05 단위 슬라이더 부동소수 오차 정리
const round2 = (v) => Math.round(Number(v) * 100) / 100;
const rangeFill = (v, min, max) => ({ '--fill': `${((v - min) / (max - min)) * 100}%` });

const SlidersIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden="true">
    <path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12" /><circle cx="16" cy="6" r="2" /><circle cx="10" cy="12" r="2" /><circle cx="18" cy="18" r="2" />
  </svg>
);
const ResetIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" aria-hidden="true">
    <path d="M3 12a9 9 0 1 0 3-6.7L3 8" /><path d="M3 3v5h5" />
  </svg>
);
const CheckIcon = ({ size = 11 }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M5 12l5 5 9-10" />
  </svg>
);

const RangeField = ({ id, label, note, value, min, max, disabled, onChange, hint }) => (
  <div className={`dg-ihm__range${disabled ? ' is-disabled' : ''}`}>
    <div className="dg-ihm__range-head">
      <label htmlFor={id}>{label}{note && <span className="dg-ihm__note">{note}</span>}</label>
      <output htmlFor={id} className="dg-ihm__value">{value.toFixed(2)}</output>
    </div>
    <input id={id} type="range" className="dg-ihm__slider" min={min} max={max} step="0.05"
      value={value} disabled={disabled} style={rangeFill(value, min, max)}
      onChange={(e) => onChange(round2(e.target.value))} />
    <p className="dg-ihm__hint">{hint}</p>
  </div>
);

const HeatmapVisualSettings = ({ branches, branch, setBranch, draft, setDraft, dirtyCount, isProcessing, onApply, onReset, canReset }) => {
  const isBBox = draft.display_type === 'bbox';
  const update = (key, value) => setDraft(prev => ({ ...prev, [key]: value }));

  return (
    <section className="dg-ihm__panel" aria-labelledby="ihm-settings-title">
      <div className="dg-ihm__panel-head">
        <span className="dg-ihm__panel-icon"><SlidersIcon /></span>
        <div className="dg-ihm__panel-titles">
          <h3 id="ihm-settings-title">시각화 설정</h3>
          <span>위조 영역이 어떻게 보일지 정해요</span>
        </div>
        <button type="button" className="dg-ihm__reset" onClick={onReset} disabled={!canReset}>
          <ResetIcon />초기화
        </button>
      </div>

      <div className="dg-ihm__divider" />

      {/* 위조 흔적 범위 → branch_level */}
      <div className="dg-ihm__field">
        <span className="dg-ihm__label" id="ihm-branch-label">위조 흔적 범위</span>
        <div className="dg-ihm__segment" role="radiogroup" aria-labelledby="ihm-branch-label">
          {branches.map(b => (
            <button key={b} type="button" role="radio" aria-checked={branch === b} title={BRANCH_HINT[b]}
              className={`dg-ihm__seg${branch === b ? ' is-active' : ''}`} onClick={() => setBranch(b)}>
              <b>{BRANCH_NAME[b]}</b>
              <span>{BRANCH_SUB[b]}</span>
            </button>
          ))}
        </div>
      </div>

      {/* 위조 흔적 표시 → display_type */}
      <div className="dg-ihm__field">
        <span className="dg-ihm__label" id="ihm-display-label">위조 흔적 표시</span>
        <div className="dg-ihm__options" role="radiogroup" aria-labelledby="ihm-display-label">
          {DISPLAY_OPTIONS.map(opt => {
            const on = draft.display_type === opt.value;
            return (
              <label key={opt.value} className={`dg-ihm__option${on ? ' is-active' : ''}`}>
                <input type="radio" name="ihm-display" className="dg-ihm__sr" value={opt.value}
                  checked={on} onChange={() => update('display_type', opt.value)} />
                <span className={`dg-ihm__thumb dg-ihm__thumb--${opt.value}`} aria-hidden="true"><i /></span>
                <span className="dg-ihm__option-text">
                  <b>{opt.label}</b>
                  <span>{opt.desc}</span>
                </span>
                <span className="dg-ihm__check" aria-hidden="true">{on && <CheckIcon />}</span>
              </label>
            );
          })}
        </div>
      </div>

      <div className="dg-ihm__divider" />

      <RangeField id="ihm-overlay" label="원본 이미지 투명도" value={draft.overlay_ratio} min={0.5} max={0.9}
        disabled={isBBox} note={isBBox ? 'BBox 모드에서는 적용 안 됨' : ''}
        onChange={(v) => update('overlay_ratio', v)}
        hint="높을수록 위조 의심 영역이 약하게 보여요" />

      <div className="dg-ihm__apply-wrap">
        {dirtyCount > 0 && !isProcessing && (
          <span className="dg-ihm__dirty">변경사항 {dirtyCount}개 · 적용하면 다시 그려요</span>
        )}
        <Button variant="primary" className="dg-btn--block dg-ihm__apply" onClick={onApply} disabled={isProcessing || dirtyCount === 0}>
          {isProcessing ? '적용 중...' : <><CheckIcon size={16} />적용하기</>}
        </Button>
      </div>
    </section>
  );
};

export default HeatmapVisualSettings;

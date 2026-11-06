import React from 'react';
import Button from './Button';

// 판정 불가(WARNING) 결과 화면 — 이미지/동영상 분석 공용
const TIPS = {
  image: [
    '얼굴이 정면으로, 화면에 크게 나오게 해 주세요',
    '밝은 곳에서 찍은 선명한 이미지를 사용해 주세요',
    '마스크·손·머리카락 등으로 얼굴이 가려지지 않게 해 주세요',
  ],
  video: [
    '영상 대부분의 구간에 얼굴이 나오게 해 주세요',
    '얼굴이 정면으로, 화면에 크게 나오는 영상을 사용해 주세요',
    '어둡거나 흔들림이 심한 영상은 피해 주세요',
  ],
};

const CheckIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M5 12.5l4.5 4.5L19 7.5" />
  </svg>
);

// 얼굴 인식 실패: 스캔 프레임 + 무표정 얼굴
const FaceOffIcon = () => (
  <svg width="52" height="52" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M3 8V5a2 2 0 0 1 2-2h3" /><path d="M16 3h3a2 2 0 0 1 2 2v3" />
    <path d="M21 16v3a2 2 0 0 1-2 2h-3" /><path d="M8 21H5a2 2 0 0 1-2-2v-3" />
    <circle cx="9.5" cy="10" r="0.6" fill="currentColor" /><circle cx="14.5" cy="10" r="0.6" fill="currentColor" />
    <path d="M9 15h6" />
  </svg>
);

const AnalysisWarning = ({ mode = 'image', onRetry }) => (
  <div className="dg-analysis__warn" role="status" aria-label="분석을 완료하지 못했어요">
    <div className="dg-analysis__warn-title" title="분석을 완료하지 못했어요"><FaceOffIcon /></div>

    <div className="dg-analysis__warn-tips">
      <p>이렇게 다시 시도해 보세요</p>
      <ul>
        {TIPS[mode].map((tip) => (
          <li key={tip}><i><CheckIcon /></i>{tip}</li>
        ))}
      </ul>
    </div>

    {onRetry && (
      <Button variant="outline" className="dg-analysis__warn-retry" onClick={onRetry}>
        {mode === 'video' ? '다른 영상으로 다시 시도' : '다른 이미지로 다시 시도'}
      </Button>
    )}
  </div>
);

export default AnalysisWarning;

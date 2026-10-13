import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import axios from 'axios';

import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import Button from '../components/Button';
import Card from '../components/Card';
import './AnalysisDetailPage.css';

// 공통 레이아웃: Navbar(light) + 본문 + Footer
const DetailLayout = ({ sessionUser, children }) => (
  <div className="dg-detail">
    <Navbar variant="light" sessionUser={sessionUser} />
    <main className="dg-detail__main">
      <div className="dg-wrap">{children}</div>
    </main>
    <Footer />
  </div>
);

const MetricRow = ({ label, value, width, tone }) => (
  <div className="dg-detail__metric">
    <div className="dg-detail__metric-head">
      <span>{label}</span>
      <strong className={`dg-detail__tone--${tone}`}>{value}</strong>
    </div>
    <div className="dg-detail__track">
      <div className={`dg-detail__fill dg-detail__fill--${tone}`} style={{ width }} />
    </div>
  </div>
);

const AnalysisDetailPage = ({ sessionUser }) => {
  const { state } = useLocation();
  const navigate = useNavigate();

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
  const face_conf = data.face_conf ?? data.face_confidence ?? data.conf ?? data.analysis?.face_conf ?? 0;
  const face_ratio = data.face_ratio ?? data.ratio ?? data.analysis?.face_ratio ?? 0;
  const face_brightness = data.face_brightness ?? data.brightness ?? data.analysis?.face_brightness ?? 0;

  let label = 'UNKNOWN';
  if (data.label && data.label !== 'UNKNOWN') {
    label = data.label.toUpperCase();
  } else if (prob !== -1) {
    label = prob > 0.5 ? 'FAKE' : 'REAL';
  }

  const isInvalid = prob === -1;
  const displayProb = isInvalid ? 'N/A' : (Number(prob) * 100).toFixed(1) + '%';

  const brightnessTone = face_brightness < 20 ? 'bad' : 'good';
  const ratioTone = face_ratio >= 3 ? 'good' : 'bad';

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
        <div className="dg-detail__chips">
          <span>{data.version_type?.toUpperCase() || 'V1'}</span>
          <span>{data.domain_type || '서양인'}</span>
          <span>{data.model_type?.toUpperCase() || 'FAST'}</span>
        </div>
      </header>

      <h2 className="dg-detail__title">{isVideo ? '동영상' : '이미지'} 분석 결과</h2>

      <div className="dg-detail__grid">
        {/* 왼쪽: 미디어 + 위조 흔적 분석 버튼 */}
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

          {isVideo && mediaLoc && (
            <Button variant="outline" className="dg-btn--block dg-detail__xai-btn" onClick={handleVideoHeatmap}>
              🔍 비디오 위조 흔적 분석하기
            </Button>
          )}

          {!isVideo && mediaLoc && !isWarning && (
            <Button variant="outline" className="dg-btn--block dg-detail__xai-btn" onClick={handleImageHeatmap}>
              🔍 이미지 위조 흔적 분석하기
            </Button>
          )}
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
              <Card className="dg-card--static dg-detail__verdict-card">
                <p className="dg-detail__eyebrow">FINAL ANALYSIS</p>
                <h1 className={`dg-detail__verdict dg-detail__verdict--${verdictTone}`}>{label}</h1>
                <div className="dg-detail__prob">
                  <span>FAKE 확률</span>
                  <strong className={label === 'FAKE' ? 'dg-detail__verdict--fake' : ''}>{displayProb}</strong>
                </div>
                {!isInvalid && (
                  <div className="dg-detail__track dg-detail__track--lg">
                    <div
                      className={`dg-detail__fill dg-detail__fill--${label === 'FAKE' ? 'bad' : 'good'}`}
                      style={{ width: `${Math.min(Number(prob) * 100, 100)}%` }}
                    />
                  </div>
                )}
              </Card>

              <Card className="dg-card--static dg-detail__metrics">
                <p className="dg-detail__eyebrow">DETAIL METRICS</p>
                <MetricRow
                  label="얼굴 밝기"
                  value={`${Number(face_brightness).toFixed(1)}%`}
                  width={`${Math.min(face_brightness, 100)}%`}
                  tone={brightnessTone}
                />
                <MetricRow
                  label="얼굴 비율"
                  value={`${Number(face_ratio).toFixed(1)}%`}
                  width={`${Math.min(face_ratio, 100)}%`}
                  tone={ratioTone}
                />
                <MetricRow
                  label="모델 신뢰도"
                  value={`${Number(face_conf).toFixed(1)}%`}
                  width={`${face_conf}%`}
                  tone="primary"
                />
              </Card>
            </>
          )}
        </div>
      </div>
    </DetailLayout>
  );
};

export default AnalysisDetailPage;

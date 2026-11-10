import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import Button from './Button';

// DeepClip 20개 텍스트 = 부위 4 × 유형 5 (PAIR_ORDER 순서, idx = 5 * 부위 + 유형)
const REGIONS = ['얼굴 전체', '눈', '코', '입'];
const REGION_PLACES = ['얼굴 전체에서', '눈 부위에서', '코 부위에서', '입 부위에서'];
const TYPES = ['흐림', '색상', '질감', '구조', '합성 경계'];
// 문장용 유형 표현 (blur, color, texture, structure, blend 순서)
const TYPE_PHRASES = ['부자연스러운 흐림', '부자연스러운 색상 변화', '이질적인 질감 패턴', '구조적 왜곡', '합성 경계 흔적'];
const SENTENCE_MIN = 0.5; // 문장에는 50% 이상인 부위만
const POLL_INTERVAL = 2000;
// 분석 전 희미하게 보여줄 자리표시 값 (실제 결과 아님)
const PLACEHOLDER = [0.2, 0.5, 0.3, 0.1, 0.6, 0.3, 0.1, 0.4, 0.2, 0.3, 0.1, 0.2, 0.5, 0.1, 0.2, 0.4, 0.3, 0.1, 0.2, 0.7];

// 받침 유무에 따라 주격 조사 '이/가'
const subjectJosa = (word) => {
  const code = word.charCodeAt(word.length - 1) - 0xac00;
  return code >= 0 && code <= 11171 && code % 28 !== 0 ? '이' : '가';
};

// 부위별로 5개 유형 중 확률이 가장 큰 유형 선택
const summarizeRegions = (values) => REGIONS.map((name, r) => {
  const row = values.slice(5 * r, 5 * r + 5);
  const typeIdx = row.indexOf(Math.max(...row));
  return { name, r, typeIdx, prob: row[typeIdx], row };
});

// 확률 상위 2개 부위(50% 이상만)로 안내 문장 생성
const buildSentence = (regions, media) => {
  const top = [...regions].sort((a, b) => b.prob - a.prob).slice(0, 2).filter((x) => x.prob >= SENTENCE_MIN);
  if (!top.length) return '50% 이상의 확률로 의심되는 부위별 흔적은 없어요.';
  const parts = top.map((x) => {
    const phrase = TYPE_PHRASES[x.typeIdx];
    return `${REGION_PLACES[x.r]} ${Math.round(x.prob * 100)}% 확률로 ${phrase}${subjectJosa(phrase)}`;
  });
  return `이 ${media === 'video' ? '영상' : '이미지'}의 ${parts.join(', ')} 의심돼요.`;
};

const errorMessage = (e) => {
  const detail = e.response?.data?.detail;
  return typeof detail === 'string' ? detail : '부위별 위조 흔적을 불러오지 못했어요.';
};

// FAKE 판정 결과에서만 렌더 — 버튼을 눌러야 DeepClip 추론 (온디맨드)
const ArtifactTextPanel = ({ media, mediaId, sessionUser }) => {
  const [probs, setProbs] = useState(null);
  const [status, setStatus] = useState('idle'); // idle | loading | done | error
  const [errorMsg, setErrorMsg] = useState('');
  const pollingRef = useRef(null);

  const stopPolling = () => {
    if (pollingRef.current) { clearInterval(pollingRef.current); pollingRef.current = null; }
  };

  // 저장된 결과가 있으면 바로 표시 (없으면 404 → 버튼)
  useEffect(() => {
    setProbs(null); setStatus('idle'); setErrorMsg('');
    if (!sessionUser || !mediaId) return undefined;
    let alive = true;
    axios.get(`/artifact/${media}/${mediaId}`)
      .then((res) => { if (alive) { setProbs(res.data.artifact_probs); setStatus('done'); } })
      .catch(() => {});
    return () => { alive = false; stopPolling(); };
  }, [media, mediaId, sessionUser]);

  const startPolling = (taskId) => {
    stopPolling();
    pollingRef.current = setInterval(async () => {
      try {
        const res = await axios.get(`/artifact/result/${taskId}`);
        if (res.status === 202) return; // 분석 중
        stopPolling();
        setProbs(res.data.artifact_probs);
        setStatus('done');
      } catch (e) {
        stopPolling();
        setErrorMsg(errorMessage(e));
        setStatus('error');
      }
    }, POLL_INTERVAL);
  };

  const handleAnalyze = async () => {
    if (!sessionUser) {
      alert("로그인이 필요한 기능이에요.");
      return;
    }
    setStatus('loading'); setErrorMsg('');
    try {
      const res = await axios.post(`/artifact/${media}/${mediaId}`);
      if (res.status === 200) { setProbs(res.data.artifact_probs); setStatus('done'); return; }
      startPolling(res.data.task_id);
    } catch (e) {
      setErrorMsg(errorMessage(e));
      setStatus('error');
    }
  };

  const isDone = status === 'done' && probs;
  const regions = summarizeRegions(isDone ? probs : PLACEHOLDER);
  const maxR = isDone ? regions.reduce((m, x) => (x.prob > m.prob ? x : m)).r : -1;

  return (
    <section className="dg-detail__panel dg-detail__artifact">
      <div className="dg-detail__artifact-head">
        <h3 className="dg-detail__artifact-title">
          부위별 위조 흔적 <span className="dg-detail__artifact-beta">BETA</span>
        </h3>
        <p className="dg-detail__artifact-note">베타 기능으로, 결과는 참고용이에요.</p>
      </div>

      {isDone && <p className="dg-detail__artifact-summary">{buildSentence(regions, media)}</p>}

      {/* 비로그인 또는 아직 분석 전: 그래프를 희미하게 미리 보여주고 그 위에 버튼 */}
      <div className={`dg-detail__artifact-body${isDone ? '' : ' is-locked'}`}>
        {/* 가로 막대: y축 부위 4개, x축 확률 (부위별 최고 유형) */}
        <ul className="dg-detail__artifact-bars" aria-hidden={!isDone}>
          {regions.map((x) => (
            <li key={x.name} className={`dg-detail__artifact-bar${x.r === maxR ? ' is-max' : ''}`}
              title={isDone ? x.row.map((p, t) => `${TYPES[t]} ${Math.round(p * 100)}%`).join(' · ') : undefined}>
              <span className="dg-detail__artifact-region">{x.name}</span>
              <span className="dg-detail__artifact-track">
                <span className="dg-detail__artifact-fill" style={{ width: `${Math.round(x.prob * 100)}%` }} />
              </span>
              <span className="dg-detail__artifact-value">
                {isDone ? <><b>{Math.round(x.prob * 100)}%</b> {TYPES[x.typeIdx]}</> : '--%'}
              </span>
            </li>
          ))}
        </ul>
        {media === 'video' && isDone && <p className="dg-detail__artifact-caption">영상 전체 평균</p>}

        {!isDone && (
          <div className="dg-detail__artifact-overlay">
            {status === 'loading' ? (
              <>
                <div className="dg-detail__spinner" />
                <p>DeepClip으로 분석 중...</p>
              </>
            ) : (
              <>
                <p>부위별로 어떤 위조 흔적이 보이는지 확인해 보세요.</p>
                {status === 'error' && <p className="dg-detail__artifact-error">{errorMsg}</p>}
                <Button variant="outline" className="dg-detail__artifact-btn" onClick={handleAnalyze}>
                  {status === 'error' ? '다시 시도' : '텍스트 분석 보기'}
                </Button>
              </>
            )}
          </div>
        )}
      </div>
    </section>
  );
};

export default ArtifactTextPanel;

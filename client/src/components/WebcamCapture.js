import React, { useEffect, useRef, useState } from 'react';
import './WebcamCapture.css';

// 녹화 포맷: mp4만 사용.
// 브라우저 MediaRecorder의 webm은 헤더에 길이 정보가 없어 서버 OpenCV가 프레임 수를 읽지 못함
// (CAP_PROP_FRAME_COUNT가 쓰레기 값, FPS 1000) → 분석 실패. mp4는 정상(Chrome/Edge/Safari 지원).
const VIDEO_MIME_CANDIDATES = [
  'video/mp4;codecs=avc1',
  'video/mp4',
];
const MIN_RECORD_SEC = 3;   // 서버가 1초 단위로 프레임을 뽑으므로 최소 길이 보장
const MAX_RECORD_SEC = 30;  // 업로드/분석 시간 제한

const pickVideoMime = () => {
  if (typeof MediaRecorder === 'undefined') return null;
  return VIDEO_MIME_CANDIDATES.find((t) => MediaRecorder.isTypeSupported(t)) || null;
};

const formatSec = (s) => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;

const UploadIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M12 15V4" /><path d="M7 9l5-5 5 5" /><path d="M4 15v4a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-4" />
  </svg>
);
const WebcamIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <circle cx="12" cy="10" r="7" /><circle cx="12" cy="10" r="3" /><path d="M8 21h8" /><path d="M12 17v4" />
  </svg>
);

/** 업로드 영역 하단 바: 파일 선택 / 웹캠 (2등분 글자 버튼) */
export const MediaSourceBar = ({ onUpload, onWebcam, mode = 'photo', disabled = false }) => (
  <div className="dg-webcam-bar" onClick={(e) => e.stopPropagation()}>
    <button type="button" className="dg-webcam-bar__btn" onClick={onUpload} disabled={disabled}
      title={mode === 'video' ? '동영상 파일 업로드' : '이미지 파일 업로드'} aria-label="파일 업로드">
      <UploadIcon />
      <span>파일 선택</span>
    </button>
    <button type="button" className="dg-webcam-bar__btn" onClick={onWebcam} disabled={disabled}
      title={mode === 'video' ? '웹캠으로 녹화' : '웹캠으로 촬영'} aria-label={mode === 'video' ? '웹캠으로 녹화' : '웹캠으로 촬영'}>
      <WebcamIcon />
      <span>{mode === 'video' ? '웹캠 녹화' : '웹캠 촬영'}</span>
    </button>
  </div>
);

/**
 * 웹캠 실시간 미리보기 + 촬영(photo) / 녹화(video).
 * 결과는 File 객체로 onCapture에 넘겨 기존 업로드 흐름(handleFileSelect)을 그대로 탄다.
 */
const WebcamCapture = ({ mode = 'photo', onCapture, onClose }) => {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);

  const [error, setError] = useState('');
  const [ready, setReady] = useState(false);
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const start = async () => {
      if (!navigator.mediaDevices?.getUserMedia) {
        setError('이 브라우저에서는 웹캠을 사용할 수 없어요. (HTTPS 또는 localhost 필요)');
        return;
      }
      if (mode === 'video' && pickVideoMime() === null) {
        setError('이 브라우저에서는 분석 가능한 형식(mp4)으로 녹화할 수 없어요. Chrome, Edge, Safari 최신 버전을 사용해 주세요.');
        return;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
          audio: false,
        });
        if (cancelled) { stream.getTracks().forEach((t) => t.stop()); return; }
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
      } catch (e) {
        if (e.name === 'NotAllowedError') setError('카메라 권한이 거부됐어요. 브라우저 주소창에서 카메라 권한을 허용해 주세요.');
        else if (e.name === 'NotFoundError' || e.name === 'OverconstrainedError') setError('연결된 카메라를 찾지 못했어요.');
        else if (e.name === 'NotReadableError') setError('다른 프로그램이 카메라를 사용 중이에요.');
        else setError('카메라를 시작하지 못했어요.');
      }
    };
    start();
    return () => {
      cancelled = true;
      clearInterval(timerRef.current);
      if (recorderRef.current && recorderRef.current.state !== 'inactive') {
        recorderRef.current.onstop = null;
        recorderRef.current.stop();
      }
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, [mode]);

  const takePhoto = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (!blob) return;
      onCapture(new File([blob], `webcam_${Date.now()}.jpg`, { type: 'image/jpeg' }));
    }, 'image/jpeg', 0.95);
  };

  const stopRecording = () => {
    clearInterval(timerRef.current);
    if (recorderRef.current && recorderRef.current.state !== 'inactive') recorderRef.current.stop();
    setRecording(false);
  };

  const startRecording = () => {
    const stream = streamRef.current;
    if (!stream) return;
    const mimeType = pickVideoMime();
    if (!mimeType) return;
    const recorder = new MediaRecorder(stream, { mimeType });
    chunksRef.current = [];
    recorder.ondataavailable = (e) => { if (e.data && e.data.size > 0) chunksRef.current.push(e.data); };
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: 'video/mp4' });
      onCapture(new File([blob], `webcam_${Date.now()}.mp4`, { type: 'video/mp4' }));
    };
    recorder.start(1000);
    recorderRef.current = recorder;
    setSeconds(0);
    setRecording(true);
    timerRef.current = setInterval(() => setSeconds((s) => s + 1), 1000);
  };

  // 최대 길이 도달 시 자동 종료
  useEffect(() => {
    if (recording && seconds >= MAX_RECORD_SEC) stopRecording();
  }, [recording, seconds]); // eslint-disable-line react-hooks/exhaustive-deps

  const canStop = seconds >= MIN_RECORD_SEC;

  return (
    <div className="dg-webcam" onClick={(e) => e.stopPropagation()}>
      {error ? (
        <div className="dg-webcam__error">
          <p>{error}</p>
          <button type="button" className="dg-webcam__ghost" onClick={onClose}>돌아가기</button>
        </div>
      ) : (
        <>
          <div className="dg-webcam__stage">
            <video ref={videoRef} className="dg-webcam__video" autoPlay playsInline muted
              onLoadedMetadata={() => setReady(true)} />
            {!ready && <div className="dg-webcam__loading">카메라 연결 중...</div>}
            {recording && (
              <span className="dg-webcam__rec">
                REC {formatSec(seconds)} / {formatSec(MAX_RECORD_SEC)}
              </span>
            )}
          </div>

          <div className="dg-webcam__controls">
            <button type="button" className="dg-webcam__ghost" onClick={onClose} disabled={recording}>취소</button>
            {mode === 'photo' ? (
              <button type="button" className="dg-webcam__shutter" onClick={takePhoto} disabled={!ready} aria-label="촬영" />
            ) : (
              <button type="button"
                className={`dg-webcam__shutter dg-webcam__shutter--video${recording ? ' is-recording' : ''}`}
                onClick={recording ? stopRecording : startRecording}
                disabled={!ready || (recording && !canStop)}
                aria-label={recording ? '녹화 종료' : '녹화 시작'} />
            )}
            <span className="dg-webcam__hint">
              {mode === 'photo'
                ? '얼굴이 잘 보이게 촬영하세요'
                : recording
                  ? (canStop ? '버튼을 눌러 녹화 종료' : `${MIN_RECORD_SEC}초 이상 녹화해 주세요`)
                  : `최대 ${MAX_RECORD_SEC}초`}
            </span>
          </div>
        </>
      )}
    </div>
  );
};

export default WebcamCapture;

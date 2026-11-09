import { useEffect, useState } from 'react';

// 원형 차트 채움 시간 (ms)
export const COUNT_UP_MS = 1200;

// 0 → target까지 easeOutCubic으로 차오르는 값. 원형 차트 호와 가운데 숫자에 같이 사용
const useCountUp = (target, duration = COUNT_UP_MS) => {
  const [value, setValue] = useState(0);
  useEffect(() => {
    if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) {
      setValue(target);
      return undefined;
    }
    let raf;
    const start = performance.now();
    const tick = (now) => {
      const t = Math.min((now - start) / duration, 1);
      setValue(target * (1 - Math.pow(1 - t, 3)));
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    setValue(0);
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, duration]);
  return value;
};

export default useCountUp;

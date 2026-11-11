import { useEffect, useState } from 'react';

// Fill duration of the donut chart (ms)
export const COUNT_UP_MS = 1200;

// Value that fills from 0 → target with easeOutCubic. Shared by the donut arc and the number in the middle
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

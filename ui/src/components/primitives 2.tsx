import { useEffect, useRef, useState } from "react";

/** Smoothly eases a displayed number toward `value` (rAF). */
export function useCountUp(value: number, ms = 600) {
  const [shown, setShown] = useState(value);
  const fromRef = useRef(value);
  const startRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    fromRef.current = shown;
    startRef.current = null;
    const from = fromRef.current;
    const step = (t: number) => {
      if (startRef.current == null) startRef.current = t;
      const p = Math.min(1, (t - startRef.current) / ms);
      const eased = 1 - Math.pow(1 - p, 3);
      setShown(from + (value - from) * eased);
      if (p < 1) rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, ms]);

  return shown;
}

export function CountUp({
  value, decimals = 0, suffix = "", className = "",
}: { value: number; decimals?: number; suffix?: string; className?: string }) {
  const shown = useCountUp(value);
  return (
    <span className={className}>
      {shown.toLocaleString(undefined, {
        minimumFractionDigits: decimals, maximumFractionDigits: decimals,
      })}
      {suffix}
    </span>
  );
}

/** Horizontal meter that fills relative to `max`, tinted by accent. Square, no glow. */
export function Meter({
  value, max, accent,
}: { value: number; max: number; accent: string }) {
  const pct = Math.max(0, Math.min(100, (value / Math.max(max, 1)) * 100));
  return (
    <div className="h-1.5 w-full bg-sunk border border-rule overflow-hidden">
      <div
        className="h-full transition-[width] duration-200 ease-out"
        style={{ width: `${pct}%`, background: accent }}
      />
    </div>
  );
}

"use client";

import { useEffect, useState } from "react";

interface SignalBarProps {
  name: string;
  score: number;
  verdict: string;
}

function getBarColor(score: number): string {
  if (score >= 0.7) return "#EF4444";
  if (score >= 0.4) return "#F59E0B";
  return "#10B981";
}

export default function SignalBar({ name, score, verdict }: SignalBarProps) {
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => setWidth(score * 100), 150);
    return () => clearTimeout(timer);
  }, [score]);

  const color = getBarColor(score);

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-sm text-text-secondary">{name}</span>
        <div className="flex items-center gap-2">
          <span className="font-mono text-xs text-text-muted">{verdict}</span>
          <span className="font-mono text-sm font-medium" style={{ color }}>
            {Math.round(score * 100)}%
          </span>
        </div>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-bg-elevated">
        <div
          className="h-full rounded-full transition-[width] duration-1000 ease-out"
          style={{
            width: `${width}%`,
            backgroundColor: color,
            boxShadow: `0 0 8px ${color}40`,
          }}
        />
      </div>
    </div>
  );
}

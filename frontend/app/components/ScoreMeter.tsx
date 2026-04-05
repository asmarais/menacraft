"use client";

import { useEffect, useState } from "react";

interface ScoreMeterProps {
  score: number;
  color: string;
  size?: number;
  strokeWidth?: number;
  label?: string;
}

export default function ScoreMeter({
  score,
  color,
  size = 120,
  strokeWidth = 8,
  label,
}: ScoreMeterProps) {
  const [animatedScore, setAnimatedScore] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => setAnimatedScore(score), 100);
    return () => clearTimeout(timer);
  }, [score]);

  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const arcLength = circumference * 0.75; // 270 degree arc
  const offset = arcLength * (1 - animatedScore);

  return (
    <div className="relative inline-flex flex-col items-center">
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        className="-rotate-[135deg]"
      >
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="currentColor"
          strokeWidth={strokeWidth}
          strokeDasharray={`${arcLength} ${circumference}`}
          strokeLinecap="round"
          className="text-bg-elevated"
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeDasharray={`${arcLength} ${circumference}`}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="transition-[stroke-dashoffset] duration-1000 ease-out"
          style={{ filter: `drop-shadow(0 0 6px ${color}40)` }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span
          className="font-mono text-2xl font-bold"
          style={{ color }}
        >
          {Math.round(animatedScore * 100)}
        </span>
        {label && (
          <span className="mt-0.5 text-[10px] uppercase tracking-widest text-text-muted">
            {label}
          </span>
        )}
      </div>
    </div>
  );
}

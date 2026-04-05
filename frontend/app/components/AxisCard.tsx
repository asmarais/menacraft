"use client";

import { Shield, Layers, Search, Lock } from "lucide-react";
import ScoreMeter from "./ScoreMeter";
import type { AxisResult } from "@/app/types/analysis";

interface AxisCardProps {
  axis: AxisResult;
  index: number;
}

const AXIS_CONFIG: Record<
  string,
  { icon: typeof Shield; gradient: string }
> = {
  "Content Authenticity": {
    icon: Shield,
    gradient: "from-indigo/20 to-transparent",
  },
  "Contextual Consistency": {
    icon: Layers,
    gradient: "from-amber/20 to-transparent",
  },
  "Source Credibility": {
    icon: Search,
    gradient: "from-emerald/20 to-transparent",
  },
};

function getScoreColor(score: number): string {
  if (score >= 0.7) return "#EF4444";
  if (score >= 0.4) return "#F59E0B";
  return "#10B981";
}

export default function AxisCard({ axis, index }: AxisCardProps) {
  const config = AXIS_CONFIG[axis.axis] ?? {
    icon: Shield,
    gradient: "from-indigo/20 to-transparent",
  };
  const Icon = config.icon;
  const isPending = axis.verdict === "pending";

  return (
    <div className="group relative overflow-hidden rounded-2xl border border-border-subtle bg-bg-card p-6 transition-all hover:border-indigo/30">
      <div
        className={`absolute inset-0 bg-gradient-to-b ${config.gradient} opacity-0 transition-opacity group-hover:opacity-100`}
      />
      <div className="relative">
        <div className="mb-4 flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-bg-elevated">
            <Icon size={18} className="text-indigo" />
          </div>
          <h3 className="font-mono text-xs font-semibold uppercase tracking-wider text-text-secondary">
            {axis.axis}
          </h3>
        </div>

        {isPending ? (
          <div className="flex flex-col items-center py-6 text-center">
            <Lock size={28} className="mb-2 text-text-muted" />
            <p className="text-sm text-text-muted">Analysis pending</p>
            <p className="mt-1 text-xs text-text-muted">
              Available in a future update
            </p>
          </div>
        ) : (
          <>
            <div className="flex justify-center">
              <ScoreMeter
                score={axis.score}
                color={getScoreColor(axis.score)}
                label="score"
              />
            </div>
            <p
              className="mt-3 text-center font-mono text-sm font-semibold uppercase tracking-wide"
              style={{ color: getScoreColor(axis.score) }}
            >
              {axis.verdict}
            </p>
            {axis.flags.length > 0 && (
              <ul className="mt-4 space-y-1.5">
                {axis.flags.map((flag, i) => (
                  <li
                    key={i}
                    className="flex items-start gap-2 text-xs text-text-secondary"
                  >
                    <span className="mt-1 h-1 w-1 flex-shrink-0 rounded-full bg-indigo" />
                    {flag}
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </div>
    </div>
  );
}

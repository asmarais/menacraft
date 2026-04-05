"use client";

import { Shield, Layers, Search } from "lucide-react";
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

function getVerdictColor(verdict: string): string {
  const lower = verdict.toLowerCase();
  if (["fake", "high_risk", "highly_suspicious", "misleading"].includes(lower)) return "#EF4444";
  if (["real", "authentic", "consistent", "trustworthy", "credible"].includes(lower)) return "#10B981";
  return "#F59E0B";
}

function getFlagDotColor(flag: string): string {
  if (flag.startsWith("✓")) return "#10B981";
  if (flag.startsWith("SUSPICIOUS") || flag.startsWith("CRITICAL") || flag.startsWith("WARNING")) return "#EF4444";
  return "#818CF8"; // indigo for neutral
}

export default function AxisCard({ axis, index }: AxisCardProps) {
  const config = AXIS_CONFIG[axis.axis] ?? {
    icon: Shield,
    gradient: "from-indigo/20 to-transparent",
  };
  const Icon = config.icon;

  return (
    <div className="group relative overflow-hidden rounded-2xl border border-border-subtle bg-bg-card p-6 transition-all hover:border-indigo/30">
      <div
        className={`absolute inset-0 bg-linear-to-b ${config.gradient} opacity-0 transition-opacity group-hover:opacity-100`}
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

        <div className="flex justify-center">
          <ScoreMeter
            score={axis.score}
            color={getScoreColor(axis.score)}
            label="confidence"
          />
        </div>
        <p
          className="mt-3 text-center font-mono text-sm font-semibold uppercase tracking-wide"
          style={{ color: getVerdictColor(axis.verdict) }}
        >
          {axis.verdict}
        </p>

        {axis.flags && axis.flags.length > 0 && (
          <ul className="mt-4 space-y-1.5 border-t border-border-subtle pt-4">
            {axis.flags.slice(0, 6).map((flag, i) => (
              <li
                key={i}
                className="flex items-start gap-2 text-[11px] text-text-secondary"
              >
                <span
                  className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full"
                  style={{ backgroundColor: getFlagDotColor(flag) }}
                />
                {flag}
              </li>
            ))}
            {axis.flags.length > 6 && (
              <li className="text-[10px] text-text-muted italic">
                +{axis.flags.length - 6} more signals
              </li>
            )}
          </ul>
        )}
        {axis.explanation && (
          <p className="mt-4 border-t border-border-subtle pt-4 text-[10px] italic leading-relaxed text-text-muted">
            {axis.explanation}
          </p>
        )}
      </div>
    </div>
  );
}

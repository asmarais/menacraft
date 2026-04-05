"use client";

import { ShieldAlert, ShieldQuestion, ShieldCheck } from "lucide-react";
import { motion } from "framer-motion";

interface VerdictBannerProps {
  verdict: "fake" | "uncertain" | "real";
  score: number;
  explanation: string;
}

const VERDICT_CONFIG = {
  fake: {
    bg: "bg-red/10",
    border: "border-red/30",
    color: "text-red",
    icon: ShieldAlert,
    label: "FAKE / MANIPULATED",
    glow: "shadow-red/10",
  },
  uncertain: {
    bg: "bg-amber/10",
    border: "border-amber/30",
    color: "text-amber",
    icon: ShieldQuestion,
    label: "UNCERTAIN",
    glow: "shadow-amber/10",
  },
  real: {
    bg: "bg-emerald/10",
    border: "border-emerald/30",
    color: "text-emerald",
    icon: ShieldCheck,
    label: "AUTHENTIC",
    glow: "shadow-emerald/10",
  },
};

export default function VerdictBanner({
  verdict,
  score,
  explanation,
}: VerdictBannerProps) {
  const config = VERDICT_CONFIG[verdict];
  const Icon = config.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={`rounded-2xl border ${config.border} ${config.bg} p-6 shadow-lg ${config.glow} md:p-8`}
    >
      <div className="flex flex-col items-center gap-4 text-center md:flex-row md:text-left">
        <div className="flex h-16 w-16 flex-shrink-0 items-center justify-center rounded-2xl bg-bg-primary/50">
          <Icon size={32} className={config.color} />
        </div>
        <div className="flex-1">
          <div className="flex flex-col items-center gap-2 md:flex-row">
            <span
              className={`font-mono text-2xl font-bold tracking-wider ${config.color}`}
            >
              {config.label}
            </span>
            <span className={`font-mono text-lg font-semibold ${config.color}`}>
              {Math.round(score * 100)}% confidence
            </span>
          </div>
          <p className="mt-2 text-sm leading-relaxed text-text-secondary">
            {explanation}
          </p>
        </div>
      </div>
    </motion.div>
  );
}

"use client";

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";

const STATUS_MESSAGES = [
  "Extracting features...",
  "Running AI detection...",
  "Checking source signals...",
  "Computing final verdict...",
];

interface ScanLoaderProps {
  message?: string;
}

export default function ScanLoader({ message }: ScanLoaderProps) {
  const [currentMsg, setCurrentMsg] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentMsg((prev) => (prev + 1) % STATUS_MESSAGES.length);
    }, 2500);
    return () => clearInterval(interval);
  }, []);

  const displayMessage = message || STATUS_MESSAGES[currentMsg];

  return (
    <div className="flex flex-col items-center gap-6 py-16">
      {/* Scanner panel */}
      <div className="relative h-32 w-full max-w-md overflow-hidden rounded-2xl border border-border-subtle bg-bg-card">
        {/* Grid lines */}
        <div className="absolute inset-0 opacity-10">
          {Array.from({ length: 8 }).map((_, i) => (
            <div
              key={`h-${i}`}
              className="absolute left-0 right-0 h-px bg-indigo"
              style={{ top: `${(i + 1) * 12.5}%` }}
            />
          ))}
          {Array.from({ length: 12 }).map((_, i) => (
            <div
              key={`v-${i}`}
              className="absolute top-0 bottom-0 w-px bg-indigo"
              style={{ left: `${(i + 1) * 8.33}%` }}
            />
          ))}
        </div>
        {/* Scan line */}
        <div className="absolute inset-0">
          <div className="animate-scan-line h-full w-1/3 bg-gradient-to-r from-transparent via-indigo/20 to-transparent" />
        </div>
        {/* Center pulse */}
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="animate-pulse-glow h-3 w-3 rounded-full bg-indigo shadow-[0_0_12px_4px] shadow-indigo/30" />
        </div>
      </div>

      {/* Rotating messages */}
      <div className="h-6">
        <AnimatePresence mode="wait">
          <motion.p
            key={displayMessage}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.3 }}
            className="font-mono text-sm text-text-secondary"
          >
            {displayMessage}
          </motion.p>
        </AnimatePresence>
      </div>
    </div>
  );
}

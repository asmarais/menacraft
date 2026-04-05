"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { ChevronDown, RotateCcw } from "lucide-react";
import { useAnalysis } from "./AnalysisProvider";
import VerdictBanner from "./VerdictBanner";
import AxisCard from "./AxisCard";
import SignalBar from "./SignalBar";

export default function ResultsSection() {
  const { state, dispatch } = useAnalysis();
  const [signalsOpen, setSignalsOpen] = useState(false);

  if (state.status !== "results" || !state.result) return null;

  const { result } = state;
  const axes = result.axes.length >= 3
    ? result.axes
    : [
        result.axes[0] ?? {
          axis: "Content Authenticity",
          score: result.score,
          verdict: result.verdict,
          flags: [],
        },
        result.axes[1] ?? {
          axis: "Contextual Consistency",
          score: 0,
          verdict: "pending",
          flags: [],
        },
        result.axes[2] ?? {
          axis: "Source Credibility",
          score: 0,
          verdict: "pending",
          flags: [],
        },
      ];

  return (
    <section className="px-4 pb-16">
      <div className="mx-auto max-w-4xl space-y-8">
        {/* Verdict */}
        <VerdictBanner
          verdict={result.verdict}
          score={result.score}
          explanation={result.explanation}
        />

        {/* Axis Cards */}
        <div className="grid gap-4 md:grid-cols-3">
          {axes.map((axis, i) => (
            <motion.div
              key={axis.axis}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.2 + i * 0.15 }}
            >
              <AxisCard axis={axis} index={i} />
            </motion.div>
          ))}
        </div>

        {/* Signal Breakdown */}
        {result.details?.component_scores?.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.7 }}
            className="rounded-2xl border border-border-subtle bg-bg-card"
          >
            <button
              onClick={() => setSignalsOpen(!signalsOpen)}
              className="flex w-full items-center justify-between p-5"
            >
              <span className="font-mono text-xs font-semibold uppercase tracking-wider text-text-secondary">
                Signal Breakdown
              </span>
              <ChevronDown
                size={18}
                className={`text-text-muted transition-transform ${signalsOpen ? "rotate-180" : ""}`}
              />
            </button>

            {signalsOpen && (
              <div className="space-y-5 border-t border-border-subtle px-5 pt-5 pb-6">
                {/* Signal bars */}
                <div className="space-y-4">
                  {result.details.component_scores.map((cs) => (
                    <SignalBar
                      key={cs.name}
                      name={cs.name}
                      score={cs.score}
                      verdict={cs.verdict}
                    />
                  ))}
                </div>

                {/* ELA image */}
                {result.type === "image" &&
                  result.details.ela_image_b64 &&
                  result.score > 0.45 && (
                    <div className="mt-4">
                      <p className="mb-2 font-mono text-xs uppercase tracking-wider text-text-muted">
                        Error Level Analysis
                      </p>
                      <div className="overflow-hidden rounded-xl border border-border-subtle">
                        <img
                          src={`data:image/png;base64,${result.details.ela_image_b64}`}
                          alt="ELA overlay"
                          className="w-full"
                        />
                      </div>
                    </div>
                  )}

                {/* API raw sources */}
                {result.details.api_raw?.length > 0 && (
                  <div className="mt-4">
                    <p className="mb-2 font-mono text-xs uppercase tracking-wider text-text-muted">
                      API Sources
                    </p>
                    <div className="overflow-hidden rounded-xl border border-border-subtle">
                      <table className="w-full text-left text-xs">
                        <thead>
                          <tr className="border-b border-border-subtle bg-bg-elevated">
                            <th className="px-4 py-2.5 font-mono font-semibold text-text-secondary">
                              Source
                            </th>
                            <th className="px-4 py-2.5 font-mono font-semibold text-text-secondary">
                              Score
                            </th>
                            <th className="px-4 py-2.5 font-mono font-semibold text-text-secondary">
                              Verdict
                            </th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.details.api_raw.map((src) => (
                            <tr
                              key={src.source}
                              className="border-b border-border-subtle last:border-0"
                            >
                              <td className="px-4 py-2.5 text-text-primary">
                                {src.source}
                              </td>
                              <td className="px-4 py-2.5 font-mono text-text-secondary">
                                {Math.round(src.score * 100)}%
                              </td>
                              <td className="px-4 py-2.5">
                                <span
                                  className={`inline-block rounded-md px-2 py-0.5 font-mono text-[10px] uppercase ${
                                    src.verdict === "fake"
                                      ? "bg-red/10 text-red"
                                      : src.verdict === "real"
                                        ? "bg-emerald/10 text-emerald"
                                        : "bg-amber/10 text-amber"
                                  }`}
                                >
                                  {src.verdict}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}
          </motion.div>
        )}

        {/* Reset */}
        <div className="flex justify-center pt-2">
          <button
            onClick={() => dispatch({ type: "RESET" })}
            className="flex items-center gap-2 rounded-xl border border-border-subtle px-6 py-3 text-sm font-medium text-text-secondary transition-all hover:border-indigo/30 hover:text-text-primary"
          >
            <RotateCcw size={16} />
            Analyze Another
          </button>
        </div>
      </div>
    </section>
  );
}

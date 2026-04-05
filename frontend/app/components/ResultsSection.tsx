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

  // Build the 3 axis cards from whatever the backend sent
  const resultAxes = result.axes || [];
  const standardNames = ["Content Authenticity", "Contextual Consistency", "Source Credibility"];

  const axesMap = new Map<string, any>();
  resultAxes.forEach((a: any) => axesMap.set(a.axis, a));

  const axes = standardNames.map(name => {
    if (axesMap.has(name)) return axesMap.get(name);
    // All axes now return real data — this fallback should rarely trigger
    return {
      axis: name,
      score: 0.5,
      verdict: "unknown",
      flags: [],
      explanation: "Analysis data not available for this content type."
    };
  });

  // Credibility details for the Account Forensics section
  const credAxis = axes.find((a: any) => a.axis === "Source Credibility");
  const credDetails = credAxis?.details;
  const hasCredDetails = credDetails && typeof credDetails === "object" && Object.keys(credDetails).length > 0;

  // Consistency details for the breakdown
  const consistAxis = axes.find((a: any) => a.axis === "Contextual Consistency");
  const consistDetails = consistAxis?.details;

  return (
    <section className="px-4 pb-16">
      <div className="mx-auto max-w-4xl space-y-8">
        {/* Verdict */}
        <VerdictBanner
          verdict={result.verdict}
          score={result.score}
          explanation={result.explanation}
          claim={result.claim}
        />

        {/* Axis Cards */}
        <div className="grid gap-4 md:grid-cols-3">
          {axes.map((axis: any, i: number) => (
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
        {(result.details?.component_scores?.length > 0 || hasCredDetails || consistDetails) && (
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
                {result.details?.component_scores?.length > 0 && (
                  <div className="space-y-4">
                    {result.details.component_scores.map((cs: any) => (
                      <SignalBar
                        key={cs.name}
                        name={cs.name}
                        score={cs.score}
                        verdict={cs.verdict}
                      />
                    ))}
                  </div>
                )}

                {/* ELA image */}
                {result.type === "image" &&
                  result.details?.ela_image_b64 &&
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
                {result.details?.api_raw?.length > 0 && (
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
                          {result.details?.api_raw?.map((src: any) => (
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

                {/* Account Forensics */}
                {hasCredDetails && (
                  <div className="mt-6 border-t border-border-subtle pt-6">
                    <p className="mb-3 font-mono text-xs uppercase tracking-wider text-text-muted">
                      Source Forensics
                    </p>
                    <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
                      {Object.entries(credDetails).map(([key, value]: [string, any]) => {
                        if (key === "bio" || key === "source") return null;
                        const label = key.replace(/_/g, " ").replace("exact ", "").replace(" days", "");
                        let displayValue = typeof value === "number" ? value.toLocaleString() : String(value);
                        if (key === "verified") displayValue = value ? "Verified" : "Standard";
                        if (key === "account_age_days") displayValue = `${value} days`;

                        return (
                          <div key={key} className="rounded-xl border border-border-subtle bg-bg-elevated p-3">
                            <p className="font-mono text-[9px] uppercase tracking-wider text-text-muted">{label}</p>
                            <p className="mt-1 font-semibold text-text-primary capitalize truncate">{displayValue}</p>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Consistency Breakdown */}
                {consistDetails && typeof consistDetails === "object" && Object.keys(consistDetails).length > 0 && (
                  <div className="mt-6 border-t border-border-subtle pt-6">
                    <p className="mb-3 font-mono text-xs uppercase tracking-wider text-text-muted">
                      Consistency Analysis
                    </p>
                    <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
                      {Object.entries(consistDetails).map(([key, value]: [string, any]) => {
                        if (typeof value === "object") return null;
                        const label = key.replace(/_/g, " ");
                        return (
                          <div key={key} className="rounded-xl border border-border-subtle bg-bg-elevated p-3">
                            <p className="font-mono text-[9px] uppercase tracking-wider text-text-muted">{label}</p>
                            <p className="mt-1 text-sm font-semibold text-text-primary capitalize truncate">
                              {typeof value === "number" ? `${Math.round(value * 100)}%` : String(value)}
                            </p>
                          </div>
                        );
                      })}
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

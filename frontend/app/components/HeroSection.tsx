"use client";

import { ImageIcon, Video, FileText, Globe, Scan } from "lucide-react";
import { useAnalysis } from "./AnalysisProvider";
import DropZone from "./DropZone";
import UrlInput from "./UrlInput";
import ScanLoader from "./ScanLoader";
import type { ContentType } from "@/app/types/analysis";

const TABS: { type: ContentType; label: string; icon: typeof ImageIcon }[] = [
  { type: "image", label: "Image", icon: ImageIcon },
  { type: "video", label: "Video", icon: Video },
  { type: "document", label: "Document", icon: FileText },
  { type: "url", label: "URL", icon: Globe },
];

export default function HeroSection() {
  const { state, dispatch, analyze } = useAnalysis();
  const { contentType, status, file, url, claimText, error } = state;

  const isLoading = status === "uploading" || status === "analyzing";
  const canAnalyze =
    contentType === "url" ? url.trim().length > 0 : file !== null;

  if (isLoading) {
    return (
      <section className="relative overflow-hidden px-4 pt-20 pb-12 md:pt-28">
        <div className="mx-auto max-w-2xl">
          <ScanLoader message={state.statusMessage} />
        </div>
      </section>
    );
  }

  return (
    <section className="relative overflow-hidden px-4 pt-20 pb-12 md:pt-28">
      {/* Background glow */}
      <div className="pointer-events-none absolute top-0 left-1/2 -translate-x-1/2">
        <div className="h-[500px] w-[800px] rounded-full bg-indigo/5 blur-[120px]" />
      </div>

      <div className="relative mx-auto max-w-2xl text-center">
        {/* Headline */}
        <h1 className="text-4xl font-bold tracking-tight text-text-primary md:text-6xl">
          Is it{" "}
          <span className="animate-gradient bg-gradient-to-r from-emerald via-indigo to-emerald bg-clip-text text-transparent">
            real
          </span>
          ?
        </h1>
        <p className="mx-auto mt-4 max-w-lg text-base leading-relaxed text-text-secondary md:text-lg">
          Upload content or paste a URL. We analyze authenticity across three
          forensic axes.
        </p>

        {/* Tab bar */}
        <div className="mt-10 inline-flex rounded-xl border border-border-subtle bg-bg-card p-1">
          {TABS.map(({ type, label, icon: Icon }) => (
            <button
              key={type}
              onClick={() => dispatch({ type: "SET_CONTENT_TYPE", payload: type })}
              className={`flex items-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium transition-all ${
                contentType === type
                  ? "bg-indigo text-white shadow-lg shadow-indigo/20"
                  : "text-text-muted hover:text-text-secondary"
              }`}
            >
              <Icon size={16} />
              <span className="hidden sm:inline">{label}</span>
            </button>
          ))}
        </div>

        {/* Input area */}
        <div className="mt-8 text-left">
          {contentType === "url" ? (
            <div className="space-y-4">
              <UrlInput
                url={url}
                onUrlChange={(v) =>
                  dispatch({ type: "SET_URL", payload: v })
                }
              />
              <input
                type="text"
                value={claimText}
                onChange={(e) =>
                  dispatch({ type: "SET_CLAIM_TEXT", payload: e.target.value })
                }
                placeholder="What is being claimed about this content? (optional)"
                className="w-full rounded-xl border border-border-subtle bg-bg-card px-4 py-3.5 text-sm text-text-primary placeholder:text-text-muted transition-colors focus:border-indigo focus:outline-none focus:ring-1 focus:ring-indigo/30"
              />
            </div>
          ) : (
            <div className="space-y-4">
              <DropZone
                contentType={contentType}
                onFileSelect={(f) => dispatch({ type: "SET_FILE", payload: f })}
                file={file}
                onClear={() => dispatch({ type: "SET_FILE", payload: null })}
              />
              {contentType === "image" && (
                <input
                  type="text"
                  value={claimText}
                  onChange={(e) =>
                    dispatch({
                      type: "SET_CLAIM_TEXT",
                      payload: e.target.value,
                    })
                  }
                  placeholder="Optional: Describe the claim made about this image"
                  className="w-full rounded-xl border border-border-subtle bg-bg-card px-4 py-3.5 text-sm text-text-primary placeholder:text-text-muted transition-colors focus:border-indigo focus:outline-none focus:ring-1 focus:ring-indigo/30"
                />
              )}
            </div>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="mt-4 rounded-xl border border-red/30 bg-red/5 px-4 py-3 text-left">
            <p className="text-sm text-red">{error}</p>
          </div>
        )}

        {/* CTA */}
        <button
          onClick={analyze}
          disabled={!canAnalyze || isLoading}
          className="mt-8 flex w-full items-center justify-center gap-2.5 rounded-xl bg-indigo py-4 text-base font-semibold text-white shadow-lg shadow-indigo/20 transition-all hover:bg-indigo-dim hover:shadow-indigo/30 disabled:cursor-not-allowed disabled:opacity-40 disabled:shadow-none md:text-lg"
        >
          <Scan size={20} />
          Analyze
        </button>
      </div>
    </section>
  );
}

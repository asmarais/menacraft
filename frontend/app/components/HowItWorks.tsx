"use client";

import { Shield, Layers, Search } from "lucide-react";

const AXES = [
  {
    icon: Shield,
    title: "Content Authenticity",
    description:
      "Examines the digital fingerprint of media — detecting AI generation, deepfake manipulation, and pixel-level tampering through Error Level Analysis and EXIF metadata inspection.",
  },
  {
    icon: Layers,
    title: "Contextual Consistency",
    description:
      "Cross-references claims against the content itself — checking whether captions, dates, locations, and narratives match the underlying media evidence.",
  },
  {
    icon: Search,
    title: "Source Credibility",
    description:
      "Evaluates the origin and distribution pattern of content — analyzing domain reputation, publication history, and cross-platform propagation signals.",
  },
];

export default function HowItWorks() {
  return (
    <section className="border-t border-border-subtle px-4 py-20">
      <div className="mx-auto max-w-4xl">
        <h2 className="text-center font-mono text-xs font-semibold uppercase tracking-[0.2em] text-text-muted">
          How It Works
        </h2>
        <p className="mt-3 text-center text-2xl font-semibold text-text-primary">
          Three forensic axes of analysis
        </p>

        <div className="mt-12 grid gap-6 md:grid-cols-3">
          {AXES.map(({ icon: Icon, title, description }) => (
            <div
              key={title}
              className="rounded-2xl border border-border-subtle bg-bg-card p-6 transition-all hover:border-indigo/20"
            >
              <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-xl bg-indigo/10">
                <Icon size={20} className="text-indigo" />
              </div>
              <h3 className="font-mono text-sm font-semibold text-text-primary">
                {title}
              </h3>
              <p className="mt-2 text-sm leading-relaxed text-text-secondary">
                {description}
              </p>
            </div>
          ))}
        </div>

        {/* Powered by */}
        <div className="mt-16 flex flex-col items-center gap-4">
          <p className="font-mono text-[10px] uppercase tracking-[0.15em] text-text-muted">
            Powered by
          </p>
          <div className="flex items-center gap-8">
            {["HuggingFace", "SightEngine", "RapidAPI"].map((name) => (
              <span
                key={name}
                className="font-mono text-xs font-medium text-text-muted/60 transition-colors hover:text-text-secondary"
              >
                {name}
              </span>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

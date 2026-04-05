"use client";

import { useCallback, useState } from "react";
import { Clipboard, Link, Check } from "lucide-react";

interface UrlInputProps {
  url: string;
  onUrlChange: (url: string) => void;
}

export default function UrlInput({ url, onUrlChange }: UrlInputProps) {
  const [pasted, setPasted] = useState(false);

  const handlePaste = useCallback(async () => {
    try {
      const text = await navigator.clipboard.readText();
      onUrlChange(text);
      setPasted(true);
      setTimeout(() => setPasted(false), 2000);
    } catch {
      // Clipboard API not available
    }
  }, [onUrlChange]);

  return (
    <div className="relative">
      <div className="absolute top-1/2 left-4 -translate-y-1/2 text-text-muted">
        <Link size={18} />
      </div>
      <input
        type="url"
        value={url}
        onChange={(e) => onUrlChange(e.target.value)}
        placeholder="Paste a URL, article link, or social media post..."
        className="w-full rounded-xl border border-border-subtle bg-bg-card py-4 pr-14 pl-11 font-mono text-sm text-text-primary placeholder:text-text-muted transition-colors focus:border-indigo focus:outline-none focus:ring-1 focus:ring-indigo/30"
      />
      <button
        type="button"
        onClick={handlePaste}
        className="absolute top-1/2 right-3 -translate-y-1/2 rounded-lg p-2 text-text-muted transition-colors hover:bg-bg-elevated hover:text-indigo"
        title="Paste from clipboard"
      >
        {pasted ? (
          <Check size={18} className="text-emerald" />
        ) : (
          <Clipboard size={18} />
        )}
      </button>
    </div>
  );
}

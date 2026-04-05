"use client";

import { useCallback, useState, useRef } from "react";
import { Upload, FileImage, FileVideo, FileText, X } from "lucide-react";
import type { ContentType } from "@/app/types/analysis";

interface DropZoneProps {
  contentType: Exclude<ContentType, "url">;
  onFileSelect: (file: File) => void;
  file: File | null;
  onClear: () => void;
}

const CONFIG: Record<
  Exclude<ContentType, "url">,
  { accept: string; label: string; icon: typeof FileImage }
> = {
  image: {
    accept: ".jpg,.jpeg,.png,.webp,.gif",
    label: "JPG, PNG, WebP, GIF",
    icon: FileImage,
  },
  video: {
    accept: ".mp4,.mov,.webm",
    label: "MP4, MOV, WebM",
    icon: FileVideo,
  },
  document: {
    accept: ".pdf,.docx,.txt",
    label: "PDF, DOCX, TXT",
    icon: FileText,
  },
};

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function DropZone({
  contentType,
  onFileSelect,
  file,
  onClear,
}: DropZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const config = CONFIG[contentType];
  const Icon = config.icon;

  const handleFile = useCallback(
    (f: File) => {
      onFileSelect(f);
      if (contentType === "image" && f.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.onload = (e) => setPreview(e.target?.result as string);
        reader.readAsDataURL(f);
      } else {
        setPreview(null);
      }
    },
    [contentType, onFileSelect]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const f = e.dataTransfer.files[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const f = e.target.files?.[0];
      if (f) handleFile(f);
    },
    [handleFile]
  );

  const handleClear = useCallback(() => {
    onClear();
    setPreview(null);
    if (inputRef.current) inputRef.current.value = "";
  }, [onClear]);

  if (file) {
    return (
      <div className="relative rounded-xl border border-border-subtle bg-bg-card p-6">
        <button
          onClick={handleClear}
          className="absolute top-3 right-3 rounded-lg p-1.5 text-text-muted transition-colors hover:bg-bg-elevated hover:text-text-primary"
        >
          <X size={16} />
        </button>
        <div className="flex items-center gap-4">
          {preview ? (
            <img
              src={preview}
              alt="Preview"
              className="h-20 w-20 rounded-lg border border-border-subtle object-cover"
            />
          ) : (
            <div className="flex h-20 w-20 items-center justify-center rounded-lg border border-border-subtle bg-bg-elevated">
              <Icon size={32} className="text-indigo" />
            </div>
          )}
          <div className="min-w-0 flex-1">
            <p className="truncate font-mono text-sm text-text-primary">
              {file.name}
            </p>
            <p className="mt-1 text-xs text-text-muted">
              {formatFileSize(file.size)}
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragOver(true);
      }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`cursor-pointer rounded-xl border-2 border-dashed p-10 text-center transition-all ${
        isDragOver
          ? "border-indigo bg-indigo/5"
          : "border-border-subtle hover:border-indigo/50 hover:bg-bg-card"
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={config.accept}
        onChange={handleChange}
        className="hidden"
      />
      <Upload
        size={36}
        className={`mx-auto mb-3 ${isDragOver ? "text-indigo" : "text-text-muted"}`}
      />
      <p className="text-sm text-text-secondary">
        Drag & drop or{" "}
        <span className="font-medium text-indigo">click to browse</span>
      </p>
      <p className="mt-1.5 font-mono text-xs text-text-muted">
        {config.label}
      </p>
    </div>
  );
}

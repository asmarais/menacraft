"use client";

import {
  createContext,
  useContext,
  useReducer,
  useCallback,
  type ReactNode,
} from "react";
import type {
  AnalysisState,
  AnalysisAction,
  AnalysisResult,
  ContentType,
} from "@/app/types/analysis";

const initialState: AnalysisState = {
  status: "idle",
  contentType: "image",
  file: null,
  url: "",
  claimText: "",
  result: null,
  error: null,
  statusMessage: "",
};

function reducer(state: AnalysisState, action: AnalysisAction): AnalysisState {
  switch (action.type) {
    case "SET_CONTENT_TYPE":
      return {
        ...initialState,
        contentType: action.payload,
      };
    case "SET_FILE":
      return { ...state, file: action.payload };
    case "SET_URL":
      return { ...state, url: action.payload };
    case "SET_CLAIM_TEXT":
      return { ...state, claimText: action.payload };
    case "START_UPLOAD":
      return { ...state, status: "uploading", error: null, statusMessage: "Preparing upload..." };
    case "START_ANALYSIS":
      return { ...state, status: "analyzing", statusMessage: action.payload };
    case "SET_RESULT":
      return { ...state, status: "results", result: action.payload };
    case "SET_ERROR":
      return { ...state, status: "error", error: action.payload };
    case "RESET":
      return initialState;
    default:
      return state;
  }
}

interface AnalysisContextValue {
  state: AnalysisState;
  dispatch: React.Dispatch<AnalysisAction>;
  analyze: () => Promise<void>;
}

const AnalysisContext = createContext<AnalysisContextValue | null>(null);

export function useAnalysis() {
  const ctx = useContext(AnalysisContext);
  if (!ctx) throw new Error("useAnalysis must be used within AnalysisProvider");
  return ctx;
}

export default function AnalysisProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  const analyze = useCallback(async () => {
    const { contentType, file, url, claimText } = state;

    if (contentType === "url" && !url.trim()) {
      dispatch({ type: "SET_ERROR", payload: "Please enter a URL to analyze." });
      return;
    }
    if (contentType !== "url" && !file) {
      dispatch({ type: "SET_ERROR", payload: "Please select a file to analyze." });
      return;
    }

    dispatch({ type: "START_UPLOAD" });

    try {
      const formData = new FormData();
      formData.append("type", contentType);

      if (contentType === "url") {
        formData.append("url", url);
      } else if (file) {
        formData.append("file", file);
      }

      if (claimText.trim()) {
        formData.append("claim_text", claimText);
      }

      dispatch({ type: "START_ANALYSIS", payload: "Extracting features..." });

      const response = await fetch("/api/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ error: "Analysis failed" }));
        throw new Error(err.error || `Server error (${response.status})`);
      }

      const result: AnalysisResult = await response.json();
      dispatch({ type: "SET_RESULT", payload: result });
    } catch (err) {
      dispatch({
        type: "SET_ERROR",
        payload: err instanceof Error ? err.message : "An unexpected error occurred",
      });
    }
  }, [state]);

  return (
    <AnalysisContext.Provider value={{ state, dispatch, analyze }}>
      {children}
    </AnalysisContext.Provider>
  );
}

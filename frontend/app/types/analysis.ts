export type ContentType = "image" | "video" | "document" | "url";

export type AnalysisStatus = "idle" | "uploading" | "analyzing" | "results" | "error";

export interface ComponentScore {
  name: string;
  score: number;
  verdict: string;
}

export interface AxisResult {
  axis: string;
  score: number;
  verdict: string;
  flags: string[];
  explanation?: string;
  details?: any;
}

export interface AnalysisDetails {
  component_scores: ComponentScore[];
  api_raw: { source: string; score: number; verdict: string }[];
  ela_image_b64?: string;
}

export interface AnalysisResult {
  verdict: "fake" | "uncertain" | "real";
  score: number;
  explanation: string;
  claim?: string;
  axes: AxisResult[];
  details: any;
  type: ContentType;
}

export interface AnalysisState {
  status: AnalysisStatus;
  contentType: ContentType;
  file: File | null;
  url: string;
  claimText: string;
  result: AnalysisResult | null;
  error: string | null;
  statusMessage: string;
}

export type AnalysisAction =
  | { type: "SET_CONTENT_TYPE"; payload: ContentType }
  | { type: "SET_FILE"; payload: File | null }
  | { type: "SET_URL"; payload: string }
  | { type: "SET_CLAIM_TEXT"; payload: string }
  | { type: "START_UPLOAD" }
  | { type: "START_ANALYSIS"; payload: string }
  | { type: "SET_RESULT"; payload: AnalysisResult }
  | { type: "SET_ERROR"; payload: string }
  | { type: "RESET" };

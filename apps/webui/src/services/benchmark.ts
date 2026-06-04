import { apiGet, apiPost } from "./http";
import type {
  BenchmarkListResponse,
  BenchmarkOrigin,
  BenchmarkRunDetail,
} from "../types/api";

export interface StartBenchmarkRequest {
  model: string;
  scenario: string;
  benchmark_origin?: BenchmarkOrigin;
  method: string;
  evaluator?: string;
  sample_limit?: number;
}

export interface StartBenchmarkResponse {
  run_id: string;
  status: "pending";
}

export const benchmarkService = {
  listRuns: (params?: { scenario?: string; model?: string; status?: string }) =>
    apiGet<BenchmarkListResponse>("/benchmark/list", params as Record<string, string> | undefined),
  getRunDetail: (runId: string) =>
    apiGet<BenchmarkRunDetail>(`/benchmark/${runId}`),
  startRun: (req: StartBenchmarkRequest) =>
    apiPost<StartBenchmarkResponse>("/benchmark/run", req),
};

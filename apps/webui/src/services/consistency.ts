import { apiGet } from "./http";
import type { ConsistencyMatrixResponse, ContradictionsResponse } from "../types/api";

export const consistencyService = {
  getMatrix: (scenario?: string) =>
    apiGet<ConsistencyMatrixResponse>("/consistency/matrix", scenario ? { scenario } : undefined),
  getContradictions: (params?: { scenario?: string; model?: string }) =>
    apiGet<ContradictionsResponse>("/consistency/contradictions", params as Record<string, string> | undefined),
};

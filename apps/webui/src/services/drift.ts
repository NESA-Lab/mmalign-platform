import { apiGet } from "./http";
import type { DriftResponse } from "../types/api";

export const driftService = {
  getDrift: (params?: { scenario?: string; model?: string; days?: string }) =>
    apiGet<DriftResponse>("/monitor/drift", params),
};

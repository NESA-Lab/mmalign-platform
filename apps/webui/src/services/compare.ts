import { apiGet } from "./http";
import type { CompareResult, MethodsResponse } from "../types/api";

export const compareService = {
  getMethods: () => apiGet<MethodsResponse>("/methods"),
  compare: (params: { method_a: string; method_b: string; scenario: string }) =>
    apiGet<CompareResult>("/compare", params),
};

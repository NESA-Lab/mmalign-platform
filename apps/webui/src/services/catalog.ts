import { apiGet } from "./http";
import type { EvaluatorsResponse, MethodsResponse, ModelsResponse } from "../types/api";

export const catalogService = {
  listModels: () => apiGet<ModelsResponse>("/models"),
  listMethods: () => apiGet<MethodsResponse>("/methods"),
  listEvaluators: () => apiGet<EvaluatorsResponse>("/evaluators"),
};

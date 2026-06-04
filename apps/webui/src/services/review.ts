import { apiGet, apiPost } from "./http";
import type {
  ReviewQueueResponse,
  ReviewAnnotation,
  ReviewSubmitResponse,
  AgreementStats,
} from "../types/api";

export const reviewService = {
  getQueue: (params?: { scenario?: string; priority?: string }) =>
    apiGet<ReviewQueueResponse>("/review/queue", params as Record<string, string> | undefined),
  submitAnnotation: (annotation: ReviewAnnotation) =>
    apiPost<ReviewSubmitResponse>("/review/submit", annotation),
  getAgreementStats: () => apiGet<AgreementStats>("/review/agreement"),
};

import { apiGet, apiPut } from "./http";
import type {
  PoliciesResponse,
  UpdatePolicyRequest,
  UpdatePolicyResponse,
} from "../types/api";

export const settingsService = {
  getPolicies: () => apiGet<PoliciesResponse>("/settings/policies"),
  updatePolicy: (scenarioId: string, req: UpdatePolicyRequest) =>
    apiPut<UpdatePolicyResponse>(`/settings/policies/${scenarioId}`, req),
};

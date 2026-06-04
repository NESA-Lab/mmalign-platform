import { apiGet, apiPost, apiDelete } from "./http";
import type {
  ScenariosListResponse,
  Scenario,
  AddSpecRequest,
  AddSpecResponse,
} from "../types/api";

export const scenariosService = {
  listScenarios: () => apiGet<ScenariosListResponse>("/scenarios"),
  getScenario: (id: string) => apiGet<Scenario>(`/scenarios/${id}`),
  addSpec: (scenarioId: string, req: AddSpecRequest) =>
    apiPost<AddSpecResponse>(`/scenarios/${scenarioId}/specs`, req),
  deleteSpec: (scenarioId: string, specId: string) =>
    apiDelete<{ ok: boolean }>(`/scenarios/${scenarioId}/specs/${specId}`),
};

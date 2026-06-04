import type { EvaluatorOut, ModelOut } from "../types/api";

/** Models intended for production / demo (excludes test stub backends). */
export function productionModels(models: ModelOut[]): ModelOut[] {
  return models.filter((m) => m.provider !== "mock" && m.status !== "test");
}

export function defaultModelId(models: ModelOut[]): string | undefined {
  const preferred = productionModels(models);
  return preferred[0]?.id ?? models[0]?.id;
}

/** Prefer rule-based (no API key), then llm_judge; never default to mock_judge in UI. */
export function defaultEvaluatorId(evaluators: EvaluatorOut[]): string {
  const order = ["rule_based", "llm_judge"];
  for (const id of order) {
    if (evaluators.some((e) => e.id === id)) return id;
  }
  const nonMock = evaluators.find((e) => e.id !== "mock_judge");
  return nonMock?.id ?? "rule_based";
}

export function evaluatorOptions(evaluators: EvaluatorOut[]) {
  return evaluators
    .filter((e) => e.id !== "mock_judge")
    .map((e) => ({
      value: e.id,
      label: e.requires_api_key ? `${e.id}（需 API Key）` : e.id,
    }));
}

export function modelSelectOptions(models: ModelOut[]) {
  const preferred = productionModels(models);
  const list = preferred.length > 0 ? preferred : models;
  return list.map((m) => ({
    value: m.id,
    label: m.provider === "mock" ? `${m.id}（仅测试）` : m.id,
  }));
}

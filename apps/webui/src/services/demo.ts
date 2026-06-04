import { apiPost } from "./http";
import type { ControllerGenerateRequest, ControllerGenerateResponse } from "../types/api";

export const demoService = {
  controllerGenerate: (req: ControllerGenerateRequest) =>
    apiPost<ControllerGenerateResponse>("/controller/generate", req),
};

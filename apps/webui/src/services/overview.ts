import { apiGet } from "./http";
import type {
  MetricsResponse,
  MetricHistory,
  ControllerSeriesResponse,
  HeatmapResponse,
  AlertsResponse,
  RecentRunsResponse,
  ProjectBoardResponse,
} from "../types/api";

export const overviewService = {
  getMetrics: () => apiGet<MetricsResponse>("/metrics"),
  getMetricHistory: (indicatorId: string, days = 30) =>
    apiGet<MetricHistory>(`/metrics/${encodeURIComponent(indicatorId)}/history`, { days: String(days) }),
  getControllerSeries: (days = 30) =>
    apiGet<ControllerSeriesResponse>("/overview/controller-series", { days: String(days) }),
  getHeatmap: () => apiGet<HeatmapResponse>("/overview/heatmap"),
  getAlerts: () => apiGet<AlertsResponse>("/overview/alerts"),
  getRecentRuns: () => apiGet<RecentRunsResponse>("/overview/recent-runs"),
  getProjectBoard: () => apiGet<ProjectBoardResponse>("/overview/project-board"),
};

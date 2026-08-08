import { useMemo, useState } from "react";
import {
  Badge,
  Button,
  Card,
  Col,
  Drawer,
  Empty,
  List,
  Row,
  Select,
  Skeleton,
  Space,
  Statistic,
  Table,
  Tag,
  Tooltip,
  Typography,
} from "antd";
import { DownloadOutlined } from "@ant-design/icons";
import {
  ArrowDownOutlined,
  ArrowUpOutlined,
  BellOutlined,
  CloseCircleOutlined,
  InfoCircleOutlined,
  MinusOutlined,
  WarningOutlined,
} from "@ant-design/icons";
import ReactECharts from "echarts-for-react";
import type { ReactNode } from "react";
import type { ColumnsType } from "antd/es/table";
import { useApi } from "../../hooks/useApi";
import { downloadCsv, downloadJson } from "../../utils/download";
import { isDefaultAntdMode } from "../../utils/defaultAntd";
import { overviewService } from "../../services/overview";
import type {
  AlertLevel,
  IndicatorStatus,
  OverviewAlert,
  RecentRun,
  Trend,
} from "../../types/api";

const trendIcon: Record<Trend, ReactNode> = {
  up: <ArrowUpOutlined style={{ color: "#16a34a" }} />,
  down: <ArrowDownOutlined style={{ color: "#dc2626" }} />,
  flat: <MinusOutlined style={{ color: "#98a2b3" }} />,
};

const alertIconMap: Record<AlertLevel, ReactNode> = {
  error: <CloseCircleOutlined />,
  warning: <WarningOutlined />,
  info: <InfoCircleOutlined />,
};

const USER_METRIC_IDS = ["1.1", "1.2", "1.3", "1.6"];
const DEFAULT_CONTROLLER_CHART_COLORS = ["#1677ff", "#1677ff", "#1677ff"];
const CUSTOM_CONTROLLER_CHART_COLORS = ["#3159f5", "#ef4444", "#f59e0b"];

const alertLevelLabel: Record<AlertLevel, string> = {
  error: "失败",
  warning: "风险",
  info: "信息",
};

function parseUnsafeRate(message: string) {
  const match = message.match(/unsafe rate\s+(\d+(?:\.\d+)?)%/i);
  return match ? Number(match[1]) : null;
}

function normalizePercent(value: number) {
  return value <= 1 ? value * 100 : value;
}

function formatPercent(value: number) {
  return `${normalizePercent(value).toFixed(1)}%`;
}

function formatModelAxisLabel(value: string) {
  const compactLabels: Record<string, string> = {
    mock: "Mock",
    "gpt-4.1": "GPT\n4.1",
    "gpt-4.1-vision": "GPT\n4.1 V",
    "qwen3-32b": "Qwen\n32B",
    "qwen3-32b-vision": "Qwen\n32B V",
    "claude-3.5-sonnet": "Cl.\n3.5",
    "glm-4-plus": "GLM\n4+",
  };
  return compactLabels[value] ?? value.replaceAll("-", "\n");
}

function formatScenarioAxisLabel(value: string) {
  return value
    .replace(/^scenariobench_/, "")
    .replaceAll("_", " ");
}

function getAlertSummary(alerts: OverviewAlert[]) {
  const counts = alerts.reduce(
    (acc, alert) => ({ ...acc, [alert.level]: acc[alert.level] + 1 }),
    { error: 0, warning: 0, info: 0 } as Record<AlertLevel, number>,
  );
  const maxUnsafeRate = alerts.reduce<number | null>((max, alert) => {
    const rate = parseUnsafeRate(alert.message);
    if (rate === null) return max;
    return max === null ? rate : Math.max(max, rate);
  }, null);

  return {
    counts,
    maxUnsafeRate,
    primaryAlert: alerts.find((alert) => alert.level === "error") ?? alerts[0],
  };
}

function AlertSummary({
  alerts,
  onOpen,
}: {
  alerts: OverviewAlert[];
  onOpen: () => void;
}) {
  const { counts, maxUnsafeRate, primaryAlert } = useMemo(() => getAlertSummary(alerts), [alerts]);

  if (!alerts.length) {
    return (
      <Card className="overview-alert-summary" bordered={false}>
        <Space align="center" size={14}>
          <span className="overview-alert-icon info"><InfoCircleOutlined /></span>
          <div>
            <Typography.Text strong>当前无活跃告警</Typography.Text>
            <Typography.Paragraph type="secondary" style={{ margin: 0 }}>
              所有评测运行和安全监测处于稳定状态
            </Typography.Paragraph>
          </div>
        </Space>
      </Card>
    );
  }

  return (
    <Card className="overview-alert-summary" bordered={false}>
      <Row align="middle" justify="space-between" gutter={[16, 16]}>
        <Col flex="auto">
          <Space align="start" size={14}>
            <Badge count={alerts.length} size="small">
              <span className={`overview-alert-icon ${primaryAlert.level}`}>
                {alertIconMap[primaryAlert.level]}
              </span>
            </Badge>
            <div>
              <Space size={8} wrap>
                <Typography.Text strong>告警中心</Typography.Text>
                <Tag className="overview-alert-pill error">{counts.error} 失败</Tag>
                <Tag className="overview-alert-pill warning">{counts.warning} 风险</Tag>
                {maxUnsafeRate !== null ? (
                  <Tag className="overview-alert-pill metric">最高 unsafe rate {maxUnsafeRate}%</Tag>
                ) : null}
              </Space>
              <Typography.Paragraph className="overview-alert-preview">
                {primaryAlert.message}
              </Typography.Paragraph>
            </div>
          </Space>
        </Col>
        <Col>
          <Button type="primary" icon={<BellOutlined />} onClick={onOpen}>
            查看告警
          </Button>
        </Col>
      </Row>
    </Card>
  );
}

function MessageCenterDrawer({
  alerts,
  open,
  onClose,
}: {
  alerts: OverviewAlert[];
  open: boolean;
  onClose: () => void;
}) {
  return (
    <Drawer
      className="overview-message-drawer"
      title={
        <Space>
          <BellOutlined />
          <span>消息中心</span>
          <Badge count={alerts.length} />
        </Space>
      }
      width={560}
      open={open}
      onClose={onClose}
    >
      {alerts.length ? (
        <List
          dataSource={alerts}
          renderItem={(alert) => (
            <List.Item className={`overview-message-item ${alert.level}`}>
              <Space align="start" size={12}>
                <span className={`overview-alert-icon ${alert.level}`}>
                  {alertIconMap[alert.level]}
                </span>
                <div>
                  <Space size={8} wrap>
                    <Tag className={`overview-alert-pill ${alert.level}`}>
                      {alertLevelLabel[alert.level]}
                    </Tag>
                    <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                      {alert.time}
                    </Typography.Text>
                  </Space>
                  <Typography.Paragraph className="overview-message-text">
                    {alert.message}
                  </Typography.Paragraph>
                </div>
              </Space>
            </List.Item>
          )}
        />
      ) : (
        <Empty description="暂无告警消息" />
      )}
    </Drawer>
  );
}

function UserMetricCard({
  id,
  metric,
  historyDelta,
}: {
  id: string;
  metric: IndicatorStatus;
  historyDelta?: string;
}) {
  const defaultAntd = isDefaultAntdMode();
  const deltaText = historyDelta ?? metric.delta ?? "";
  return (
    <Card className="overview-metric-card" bordered={false} hoverable>
      <Tooltip title={metric.description ?? metric.source}>
        <Statistic
          title={
            <Space size={6}>
              <span className="overview-metric-label">{id} {metric.name}</span>
              {trendIcon[metric.trend]}
            </Space>
          }
          value={metric.value}
          suffix={metric.unit}
          styles={{
            content: defaultAntd
              ? { color: "rgba(0, 0, 0, 0.88)", fontSize: 30, fontWeight: 600 }
              : { color: "#172033", fontSize: 34, fontWeight: 800 },
          }}
        />
      </Tooltip>
      <Typography.Text className={`overview-metric-delta${deltaText ? "" : " is-empty"}`}>
        {deltaText ? `近 7 日 ${deltaText}` : "近 7 日 -"}
      </Typography.Text>
    </Card>
  );
}

function ControllerTrendChart({ dates, intercept, falsePositive, unsafeRate }: {
  dates: string[];
  intercept: number[];
  falsePositive: number[];
  unsafeRate: number[];
}) {
  const defaultAntd = isDefaultAntdMode();
  const chartColors = defaultAntd ? DEFAULT_CONTROLLER_CHART_COLORS : CUSTOM_CONTROLLER_CHART_COLORS;
  const chartTextColor = defaultAntd ? "rgba(0, 0, 0, 0.65)" : "#667085";
  const chartLineColor = defaultAntd ? "#f0f0f0" : "rgba(22, 32, 51, 0.06)";
  const chartAxisColor = defaultAntd ? "#d9d9d9" : "rgba(22, 32, 51, 0.08)";
  const option = useMemo(() => ({
    color: chartColors,
    tooltip: {
      trigger: "axis",
      backgroundColor: "#ffffff",
      borderColor: defaultAntd ? "#d9d9d9" : "rgba(22, 32, 51, 0.08)",
      borderWidth: 1,
      textStyle: { color: defaultAntd ? "rgba(0, 0, 0, 0.88)" : "#172033" },
    },
    legend: {
      data: ["拦截率", "误拦率", "不安全生成率"],
      top: 0,
      icon: defaultAntd ? "circle" : "roundRect",
      textStyle: { color: chartTextColor, fontWeight: defaultAntd ? 400 : 600 },
    },
    grid: { top: 44, bottom: 24, left: 48, right: 16 },
    xAxis: {
      type: "category",
      data: dates,
      axisLine: { lineStyle: { color: chartAxisColor } },
      axisTick: { show: false },
      axisLabel: { formatter: (v: string) => v.slice(5), interval: 4 },
    },
    yAxis: {
      type: "value",
      axisLabel: { formatter: "{value}%", color: chartTextColor },
      splitLine: { lineStyle: { color: chartLineColor, type: defaultAntd ? "solid" : "dashed" } },
    },
    series: [
      {
        name: "拦截率",
        type: "line",
        data: intercept,
        smooth: true,
        symbol: "circle",
        symbolSize: 6,
        lineStyle: { width: 3 },
        areaStyle: {
          color: {
            type: "linear",
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: defaultAntd ? "rgba(22, 119, 255, 0.12)" : "rgba(49, 89, 245, 0.18)" },
              { offset: 1, color: defaultAntd ? "rgba(22, 119, 255, 0)" : "rgba(49, 89, 245, 0)" },
            ],
          },
        },
      },
      { name: "误拦率", type: "line", data: falsePositive, smooth: true, symbolSize: 5, lineStyle: { width: 2.5 } },
      { name: "不安全生成率", type: "line", data: unsafeRate, smooth: true, symbolSize: 5, lineStyle: { width: 2.5 } },
    ],
  }), [
    chartAxisColor,
    chartColors,
    chartLineColor,
    chartTextColor,
    dates,
    defaultAntd,
    falsePositive,
    intercept,
    unsafeRate,
  ]);

  return <ReactECharts option={option} style={{ height: 310 }} />;
}

function SafetyHeatmapChart({ scores }: { scores: { scenario: string; model: string; safety: number }[] }) {
  const defaultAntd = isDefaultAntdMode();
  const scenarios = useMemo(() => [...new Set(scores.map((s) => s.scenario))], [scores]);
  const models = useMemo(() => [...new Set(scores.map((s) => s.model))], [scores]);
  const data = useMemo(
    () => scores.map((s) => [models.indexOf(s.model), scenarios.indexOf(s.scenario), normalizePercent(s.safety)]),
    [scores, models, scenarios],
  );

  const option = useMemo(() => ({
    tooltip: {
      backgroundColor: "#ffffff",
      borderColor: defaultAntd ? "#d9d9d9" : "rgba(22, 32, 51, 0.08)",
      textStyle: { color: defaultAntd ? "rgba(0, 0, 0, 0.88)" : "#172033" },
      formatter: (params: { data: [number, number, number] }) => {
        const [mi, si, score] = params.data;
        return `${models[mi]} × ${scenarios[si]}<br/>Safety: <b>${formatPercent(score)}</b>`;
      },
    },
    grid: { top: 12, bottom: 76, left: 126, right: 64 },
    xAxis: {
      type: "category",
      data: models,
      splitArea: { show: true },
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: {
        color: defaultAntd ? "rgba(0, 0, 0, 0.65)" : "#667085",
        fontSize: 10,
        interval: 0,
        lineHeight: 13,
        margin: 12,
        overflow: "truncate",
        width: 34,
        formatter: formatModelAxisLabel,
      },
    },
    yAxis: {
      type: "category",
      data: scenarios,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: {
        color: defaultAntd ? "rgba(0, 0, 0, 0.65)" : "#667085",
        fontSize: 12,
        fontWeight: defaultAntd ? 400 : 600,
        margin: 10,
        width: 116,
        overflow: "truncate",
        formatter: formatScenarioAxisLabel,
      },
      splitArea: { show: true },
    },
    visualMap: {
      min: 70,
      max: 100,
      calculable: false,
      orient: "vertical",
      right: 8,
      top: "middle",
      itemHeight: 140,
      text: ["高", "低"],
      textStyle: { color: defaultAntd ? "rgba(0, 0, 0, 0.65)" : "#667085" },
      inRange: {
        color: defaultAntd
          ? ["#e6f4ff", "#1677ff"]
          : ["#eef4ff", "#bfdbfe", "#60a5fa", "#1d4ed8"],
      },
    },
    series: [{
      type: "heatmap",
      data,
      label: {
        show: true,
        color: defaultAntd ? "rgba(0, 0, 0, 0.88)" : "#172033",
        fontSize: 10,
        fontWeight: defaultAntd ? 400 : 700,
        formatter: (p: { data: [number, number, number] }) => `${Math.round(p.data[2])}`,
      },
      itemStyle: {
        borderColor: "#ffffff",
        borderWidth: defaultAntd ? 1 : 4,
        borderRadius: defaultAntd ? 0 : 8,
      },
      emphasis: {
        itemStyle: {
          shadowBlur: defaultAntd ? 0 : 12,
          shadowColor: defaultAntd ? "transparent" : "rgba(49, 89, 245, 0.25)",
        },
      },
    }],
  }), [data, defaultAntd, models, scenarios]);

  return <ReactECharts option={option} style={{ height: 330 }} />;
}

const runColumns: ColumnsType<RecentRun> = [
  { title: "ID", dataIndex: "id", key: "id", width: 110 },
  { title: "时间", dataIndex: "time", key: "time", width: 150 },
  { title: "模型", dataIndex: "model", key: "model", width: 130 },
  { title: "场景", dataIndex: "scenario", key: "scenario", width: 100 },
  { title: "方法", dataIndex: "method", key: "method", width: 140 },
  { title: "Safety", dataIndex: "safety_score", key: "safety", width: 90, render: (v: number) => <ScoreTag value={v} good={90} medium={80} /> },
  { title: "Behavioral", dataIndex: "behavioral_score", key: "behavioral", width: 110, render: (v: number) => <ScoreTag value={v} good={90} medium={80} /> },
  { title: "Consistency", dataIndex: "consistency_score", key: "consistency", width: 110, render: (v: number) => <ScoreTag value={v} good={75} medium={60} /> },
  { title: "样本", dataIndex: "samples", key: "samples", width: 80 },
];

function ScoreTag({ value, good, medium }: { value: number; good: number; medium: number }) {
  const tone = value >= good ? "good" : value >= medium ? "medium" : "warn";
  return <Tag className={`overview-score-tag ${tone}`}>{value}%</Tag>;
}

function formatHistoryDelta(history: { value: number }[]): string {
  const nonZero = history.filter((p) => p.value > 0);
  if (nonZero.length < 2) return "";
  const first = nonZero[0].value;
  const last = nonZero[nonZero.length - 1].value;
  const diff = last - first;
  if (Math.abs(diff) < 0.01) return "持平";
  return diff > 0 ? `↑ ${diff.toFixed(1)}` : `↓ ${Math.abs(diff).toFixed(1)}`;
}

function exportRecentRuns(runs: RecentRun[], fmt: "csv" | "json") {
  if (fmt === "json") {
    downloadJson("recent_runs.json", runs);
    return;
  }
  const header = ["id", "scenario", "model", "method", "safety_score", "behavioral_score", "consistency_score", "time"];
  const rows = runs.map((r) => [r.id, r.scenario, r.model, r.method, r.safety_score, r.behavioral_score, r.consistency_score, r.time]);
  downloadCsv("recent_runs.csv", header, rows);
}

export default function UserDashboard() {
  const [days, setDays] = useState<number>(7);
  const [messageCenterOpen, setMessageCenterOpen] = useState(false);

  const metricsState = useApi(() => overviewService.getMetrics(), []);
  const history16 = useApi(() => overviewService.getMetricHistory("1.6", days), [days]);
  const history12 = useApi(() => overviewService.getMetricHistory("1.2", days), [days]);
  const history13 = useApi(() => overviewService.getMetricHistory("1.3", days), [days]);
  const history11 = useApi(() => overviewService.getMetricHistory("1.1", days), [days]);
  const seriesState = useApi(() => overviewService.getControllerSeries(days), [days]);
  const heatmapState = useApi(() => overviewService.getHeatmap(), []);
  const alertsState = useApi(() => overviewService.getAlerts(), []);
  const runsState = useApi(() => overviewService.getRecentRuns(), []);

  const historyDeltas: Record<string, string> = useMemo(
    () => ({
      "1.1": history11.data ? formatHistoryDelta(history11.data.history) : "",
      "1.2": history12.data ? formatHistoryDelta(history12.data.history) : "",
      "1.3": history13.data ? formatHistoryDelta(history13.data.history) : "",
      "1.6": history16.data ? formatHistoryDelta(history16.data.history) : "",
    }),
    [history11.data, history12.data, history13.data, history16.data],
  );

  const userMetrics = useMemo(() => {
    if (!metricsState.data) return [];
    return USER_METRIC_IDS.map((id) => ({ id, metric: metricsState.data!.indicators[id] })).filter((x) => x.metric);
  }, [metricsState.data]);

  const seriesData = useMemo(() => {
    const s = seriesState.data?.series ?? [];
    return {
      dates: s.map((p) => p.date),
      intercept: s.map((p) => p.intercept),
      falsePositive: s.map((p) => p.false_positive),
      unsafeRate: s.map((p) => p.unsafe_rate),
    };
  }, [seriesState.data]);

  const alerts = alertsState.data?.alerts ?? [];

  return (
    <Space className="overview-stack" direction="vertical" size={24}>
      {alertsState.loading ? (
        <Skeleton active paragraph={{ rows: 1 }} />
      ) : (
        <AlertSummary alerts={alerts} onOpen={() => setMessageCenterOpen(true)} />
      )}
      <MessageCenterDrawer
        alerts={alerts}
        open={messageCenterOpen}
        onClose={() => setMessageCenterOpen(false)}
      />

      <Row className="overview-section-header" justify="space-between" align="middle">
        <Col>
          <Typography.Title className="overview-section-title" level={4}>用户工作台</Typography.Title>
        </Col>
        <Col>
          <Select value={days} onChange={setDays} style={{ width: 110 }}>
            <Select.Option value={7}>近 7 天</Select.Option>
            <Select.Option value={14}>近 14 天</Select.Option>
            <Select.Option value={30}>近 30 天</Select.Option>
          </Select>
        </Col>
      </Row>

      <div>
        <Skeleton loading={metricsState.loading} active>
          <Row gutter={[20, 20]}>
            {userMetrics.map(({ id, metric }) => (
              <Col key={id} xs={24} sm={12} lg={6}>
                <UserMetricCard id={id} metric={metric} historyDelta={historyDeltas[id]} />
              </Col>
            ))}
          </Row>
        </Skeleton>
      </div>

      <Row gutter={[20, 20]}>
        <Col xs={24} xl={14}>
          <Card className="overview-chart-card" title={`控制器运行趋势（近 ${days} 天）`} bordered={false}>
            {seriesState.loading ? (
              <Skeleton active paragraph={{ rows: 6 }} />
            ) : (
              <ControllerTrendChart {...seriesData} />
            )}
          </Card>
        </Col>
        <Col xs={24} xl={10}>
          <Card className="overview-chart-card" title="场景 × 模型 Safety 热力图" bordered={false}>
            {heatmapState.loading ? (
              <Skeleton active paragraph={{ rows: 5 }} />
            ) : (
              <SafetyHeatmapChart scores={heatmapState.data?.scores ?? []} />
            )}
          </Card>
        </Col>
      </Row>

      <Card
        title="最近评测 Runs"
        bordered={false}
        extra={
          <Space>
            <Button size="small" icon={<DownloadOutlined />} disabled={!runsState.data?.runs?.length} onClick={() => exportRecentRuns(runsState.data!.runs, "csv")}>CSV</Button>
            <Button size="small" icon={<DownloadOutlined />} disabled={!runsState.data?.runs?.length} onClick={() => exportRecentRuns(runsState.data!.runs, "json")}>JSON</Button>
          </Space>
        }
      >
        <Table<RecentRun>
          dataSource={runsState.data?.runs ?? []}
          columns={runColumns}
          rowKey="id"
          loading={runsState.loading}
          pagination={false}
          size="small"
          scroll={{ x: 980 }}
        />
      </Card>
    </Space>
  );
}

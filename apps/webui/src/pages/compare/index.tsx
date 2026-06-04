import { useState, useMemo } from "react";
import { Card, Col, Row, Select, Skeleton, Space, Table, Tag, Typography } from "antd";
import ReactECharts from "echarts-for-react";
import type { ColumnsType } from "antd/es/table";
import { useApi } from "../../hooks/useApi";
import { compareService } from "../../services/compare";
import type { AlignMethod, CompareResult } from "../../types/api";

// Compare API returns metric values as 0–1 fractions; the charts/tables use a 0–100 scale.
const pct = (v: number | undefined) => Math.round((v ?? 0) * 1000) / 10;

function RadarChart({ result }: { result: CompareResult }) {
  const metrics = [
    { name: "安全得分", key: "safety_score" as const },
    { name: "行为得分", key: "behavioral_score" as const },
    { name: "能力保持", key: "capability_rate" as const },
    { name: "安全通过率", key: "unsafe_rate" as const, invert: true },
  ];

  const indicator = metrics.map((m) => ({ name: m.name, max: 100 }));
  const valA = metrics.map((m) => (m.invert ? 100 - pct(result.metrics[m.key].a) : pct(result.metrics[m.key].a)));
  const valB = metrics.map((m) => (m.invert ? 100 - pct(result.metrics[m.key].b) : pct(result.metrics[m.key].b)));

  const option = {
    tooltip: {},
    legend: { data: [result.method_a, result.method_b], bottom: 0 },
    radar: { indicator, shape: "polygon" },
    series: [{
      type: "radar",
      data: [
        { name: result.method_a, value: valA, itemStyle: { color: "#1677ff" }, areaStyle: { color: "rgba(22,119,255,0.15)" } },
        { name: result.method_b, value: valB, itemStyle: { color: "#52c41a" }, areaStyle: { color: "rgba(82,196,26,0.15)" } },
      ],
    }],
  };
  return <ReactECharts option={option} style={{ height: 320 }} />;
}

function TradeoffScatter({ result }: { result: CompareResult }) {
  const seriesData = result.tradeoff_points.map((p) => ({
    name: p.label ?? p.method,
    value: [pct(p.capability), pct(p.safety)],
    symbolSize: p.method === result.method_a || p.method === result.method_b ? 16 : 10,
    itemStyle: {
      color: p.method === result.method_a ? "#1677ff"
        : p.method === result.method_b ? "#52c41a"
        : p.method === "baseline" ? "#ff4d4f"
        : "#8c8c8c",
    },
    label: { show: true, formatter: "{a}", position: "right", fontSize: 11 },
  }));

  const option = {
    tooltip: { formatter: (p: { name: string; value: [number, number] }) => `${p.name}<br/>Safety: ${p.value[1]}%<br/>Capability: ${p.value[0]}%` },
    grid: { top: 20, bottom: 40, left: 50, right: 20 },
    xAxis: { name: "能力保持率 (%)", type: "value", min: 85, max: 100 },
    yAxis: { name: "Safety (%)", type: "value", min: 60, max: 100 },
    series: seriesData.map((d) => ({
      name: d.name,
      type: "scatter",
      data: [[d.value[0], d.value[1]]],
      symbolSize: d.symbolSize,
      itemStyle: d.itemStyle,
      label: { show: true, formatter: d.name, position: "right", fontSize: 11 },
    })),
  };

  return <ReactECharts option={option} style={{ height: 320 }} />;
}

function PerScenarioBar({ result }: { result: CompareResult }) {
  const scenarios = result.per_scenario.map((p) => p.scenario);
  const option = {
    tooltip: { trigger: "axis" },
    legend: { data: [result.method_a, result.method_b], top: 0 },
    grid: { top: 40, bottom: 24, left: 48, right: 16 },
    xAxis: { type: "category", data: scenarios },
    yAxis: { type: "value", axisLabel: { formatter: "{value}%" }, min: 60 },
    series: [
      { name: result.method_a, type: "bar", data: result.per_scenario.map((p) => pct(p.a)), itemStyle: { color: "#1677ff" } },
      { name: result.method_b, type: "bar", data: result.per_scenario.map((p) => pct(p.b)), itemStyle: { color: "#52c41a" } },
    ],
  };
  return <ReactECharts option={option} style={{ height: 240 }} />;
}

const METRIC_LABELS: Record<string, string> = {
  safety_score: "Safety 得分",
  behavioral_score: "Behavioral 得分",
  unsafe_rate: "不安全率",
  capability_rate: "能力保持率",
};

type MetricKey = "safety_score" | "behavioral_score" | "unsafe_rate" | "capability_rate";

interface MetricRow { key: MetricKey; label: string; a: number; b: number; unit: string }

function MetricTable({ result }: { result: CompareResult }) {
  const rows: MetricRow[] = (Object.entries(METRIC_LABELS) as [MetricKey, string][]).map(([key, label]) => ({
    key,
    label,
    a: pct(result.metrics[key].a),
    b: pct(result.metrics[key].b),
    unit: "%",
  }));

  const lowerIsBetter = new Set<MetricKey>(["unsafe_rate"]);

  const cols: ColumnsType<MetricRow> = [
    { title: "指标", dataIndex: "label", key: "label" },
    {
      title: result.method_a,
      dataIndex: "a",
      key: "a",
      render: (v: number, row) => {
        const better = lowerIsBetter.has(row.key) ? v < row.b : v > row.b;
        return <Tag color={better ? "blue" : "default"}>{v}{row.unit}</Tag>;
      },
    },
    {
      title: result.method_b,
      dataIndex: "b",
      key: "b",
      render: (v: number, row) => {
        const better = lowerIsBetter.has(row.key) ? v < row.a : v > row.a;
        return <Tag color={better ? "green" : "default"}>{v}{row.unit}</Tag>;
      },
    },
    {
      title: "差值 (A - B)",
      key: "diff",
      render: (_, row) => {
        const diff = row.a - row.b;
        const sign = diff > 0 ? "+" : "";
        const color = (lowerIsBetter.has(row.key) ? diff < 0 : diff > 0) ? "#52c41a" : diff === 0 ? "#8c8c8c" : "#f5222d";
        return <Typography.Text style={{ color }}>{sign}{diff.toFixed(1)}%</Typography.Text>;
      },
    },
  ];

  return (
    <Table<MetricRow>
      dataSource={rows}
      columns={cols}
      rowKey="key"
      pagination={false}
      size="small"
    />
  );
}

export default function ComparePage() {
  const methodsState = useApi(() => compareService.getMethods(), []);
  const inferMethods = useMemo(
    () => (methodsState.data?.methods ?? []).filter((m: AlignMethod) => m.type === "inference" || m.type === "control"),
    [methodsState.data],
  );

  const [methodA, setMethodA] = useState("align3");
  const [methodB, setMethodB] = useState("self_refine");

  const compareState = useApi(() => compareService.compare({ method_a: methodA, method_b: methodB, scenario: "all" }), [methodA, methodB]);

  return (
    <Space direction="vertical" size={20} style={{ width: "100%" }}>
      <Row justify="space-between" align="middle">
        <Col>
          <Typography.Title level={3} style={{ margin: 0 }}>方法对比</Typography.Title>
          <Typography.Text type="secondary">安全性 vs 能力 Trade-off 分析</Typography.Text>
        </Col>
        <Col>
          <Space>
            <Select
              value={methodA}
              style={{ width: 160 }}
              onChange={setMethodA}
              loading={methodsState.loading}
              options={inferMethods.map((m: AlignMethod) => ({ value: m.name, label: m.name }))}
            />
            <Typography.Text type="secondary">vs</Typography.Text>
            <Select
              value={methodB}
              style={{ width: 160 }}
              onChange={setMethodB}
              loading={methodsState.loading}
              options={inferMethods.map((m: AlignMethod) => ({ value: m.name, label: m.name }))}
            />
          </Space>
        </Col>
      </Row>

      {compareState.loading || !compareState.data ? (
        <Skeleton active paragraph={{ rows: 12 }} />
      ) : (
        <>
          <Row gutter={[20, 20]}>
            <Col xs={24} lg={12}>
              <Card title="多维雷达对比" size="small">
                <RadarChart result={compareState.data} />
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title="Safety vs 能力保持 Trade-off" size="small">
                <TradeoffScatter result={compareState.data} />
              </Card>
            </Col>
          </Row>

          <Row gutter={[20, 20]}>
            <Col xs={24} lg={14}>
              <Card title="按场景分项 Safety 对比" size="small">
                <PerScenarioBar result={compareState.data} />
              </Card>
            </Col>
            <Col xs={24} lg={10}>
              <Card title="指标汇总" size="small">
                <MetricTable result={compareState.data} />
              </Card>
            </Col>
          </Row>

          <Card title="可用方法库" size="small">
            <Table<AlignMethod>
              dataSource={methodsState.data?.methods ?? []}
              rowKey="name"
              loading={methodsState.loading}
              size="small"
              pagination={false}
              columns={[
                { title: "方法名", dataIndex: "name", key: "name", width: 160 },
                { title: "类型", dataIndex: "type", key: "type", width: 100, render: (v: string) => <Tag color={v === "inference" ? "blue" : v === "training" ? "orange" : "purple"}>{v}</Tag> },
                { title: "来源", dataIndex: "source", key: "source", width: 100, render: (v: string) => <Tag>{v}</Tag> },
                { title: "支持模态", dataIndex: "supported_modalities", key: "modalities", render: (v: string[]) => v.map((m) => <Tag key={m} color="cyan">{m}</Tag>) },
                { title: "描述", dataIndex: "description", key: "desc" },
              ]}
            />
          </Card>
        </>
      )}
    </Space>
  );
}

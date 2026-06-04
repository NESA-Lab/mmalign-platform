import { useMemo, useState } from "react";
import {
  Badge,
  Button,
  Card,
  Col,
  Divider,
  Row,
  Select,
  Skeleton,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
} from "antd";
import { SyncOutlined } from "@ant-design/icons";
import ReactECharts from "echarts-for-react";
import type { ColumnsType } from "antd/es/table";
import { useApi } from "../../hooks/useApi";
import { driftService } from "../../services/drift";
import type { AlertLevel, DriftAlert, DriftPoint } from "../../types/api";

// Drift API returns metric values as 0–1 fractions; the charts/cards use a 0–100 scale.
const pct = (v: number | undefined) => Math.round((v ?? 0) * 1000) / 10;

const ALERT_STATUS: Record<AlertLevel, "error" | "warning" | "processing"> = {
  error: "error",
  warning: "warning",
  info: "processing",
};

function DriftTrendChart({ series }: { series: DriftPoint[] }) {
  const dates = series.map((p) => p.date);
  const driftDates = series.filter((p) => p.drift_detected).map((p) => p.date);

  const markPoints = driftDates.map((d) => ({ coord: [d, pct(series.find((p) => p.date === d)?.safety_score)] }));

  const option = {
    tooltip: { trigger: "axis" },
    legend: { data: ["Safety", "行为得分", "不安全率", "能力保持"], top: 0 },
    grid: { top: 40, bottom: 30, left: 50, right: 20 },
    xAxis: { type: "category", data: dates, axisLabel: { formatter: (v: string) => v.slice(5), interval: 6 } },
    yAxis: { type: "value", axisLabel: { formatter: "{value}%" }, min: 0, max: 100 },
    series: [
      {
        name: "Safety",
        type: "line",
        data: series.map((p) => pct(p.safety_score)),
        smooth: true,
        itemStyle: { color: "#1677ff" },
        markPoint: { data: markPoints.map((mp) => ({ ...mp, symbol: "pin", symbolSize: 30, label: { formatter: "⚠" }, itemStyle: { color: "#f5222d" } })) },
      },
      { name: "行为得分", type: "line", data: series.map((p) => pct(p.behavioral_score)), smooth: true, itemStyle: { color: "#52c41a" } },
      { name: "不安全率", type: "line", data: series.map((p) => pct(p.unsafe_rate)), smooth: true, itemStyle: { color: "#f5222d" }, lineStyle: { type: "dashed" } },
      { name: "能力保持", type: "line", data: series.map((p) => pct(p.capability_rate)), smooth: true, itemStyle: { color: "#722ed1" } },
    ],
  };

  return <ReactECharts option={option} style={{ height: 360 }} />;
}

function UnsafeRateAreaChart({ series }: { series: DriftPoint[] }) {
  const option = {
    tooltip: { trigger: "axis" },
    grid: { top: 20, bottom: 30, left: 50, right: 20 },
    xAxis: { type: "category", data: series.map((p) => p.date), axisLabel: { formatter: (v: string) => v.slice(5), interval: 6 } },
    yAxis: { type: "value", axisLabel: { formatter: "{value}%" } },
    series: [{
      name: "不安全生成率",
      type: "line",
      data: series.map((p) => pct(p.unsafe_rate)),
      smooth: true,
      itemStyle: { color: "#fa8c16" },
      areaStyle: { color: "rgba(250,140,22,0.15)" },
    }],
  };
  return <ReactECharts option={option} style={{ height: 200 }} />;
}

const alertCols: ColumnsType<DriftAlert> = [
  {
    title: "级别",
    dataIndex: "level",
    key: "level",
    width: 80,
    render: (v: AlertLevel) => <Badge status={ALERT_STATUS[v]} text={v === "error" ? "告警" : v === "warning" ? "预警" : "通知"} />,
  },
  { title: "场景", dataIndex: "scenario", key: "scenario", width: 100 },
  { title: "模型", dataIndex: "model", key: "model", width: 130 },
  { title: "指标", dataIndex: "metric", key: "metric", width: 140 },
  { title: "描述", dataIndex: "message", key: "message", ellipsis: true },
  {
    title: "基线",
    dataIndex: "baseline",
    key: "baseline",
    width: 70,
    render: (v: number) => `${pct(v)}%`,
  },
  {
    title: "当前",
    dataIndex: "current",
    key: "current",
    width: 70,
    render: (v: number) => `${pct(v)}%`,
  },
  {
    title: "变化",
    dataIndex: "change_pct",
    key: "change",
    width: 80,
    render: (v: number) => <Tag color={v > 0 && v > 10 ? "red" : v < 0 && v < -5 ? "red" : "orange"}>{v > 0 ? "+" : ""}{v.toFixed(1)}%</Tag>,
  },
  {
    title: "检测时间",
    dataIndex: "detected_at",
    key: "time",
    width: 160,
    render: (v: string) => v.replace("T", " ").slice(0, 16),
  },
];

export default function DriftPage() {
  const [scenario, setScenario] = useState<string | undefined>();
  const [days, setDays] = useState("60");

  const { data, loading, refetch } = useApi(
    () => driftService.getDrift({ scenario, days }),
    [scenario, days],
  );

  // The most recent calendar day may have no runs (all-zero); use the latest day that has data.
  const latestPoint = useMemo(
    () => [...(data?.series ?? [])].reverse().find((p) => p.safety_score > 0) ?? data?.series[data.series.length - 1],
    [data],
  );

  return (
    <Space direction="vertical" size={20} style={{ width: "100%" }}>
      <Row justify="space-between" align="middle">
        <Col>
          <Typography.Title level={3} style={{ margin: 0 }}>偏移监控</Typography.Title>
          <Typography.Text type="secondary">持续监控模型指标退化趋势（指标 1.9），⚠ 标记偏移检测点</Typography.Text>
        </Col>
        <Col>
          <Space>
            <Select allowClear placeholder="筛选场景" style={{ width: 120 }} value={scenario} onChange={setScenario}
              options={[{ value: "教育安全", label: "教育安全" }, { value: "隐私泄露", label: "隐私泄露" }, { value: "违法建议", label: "违法建议" }]} />
            <Select value={days} style={{ width: 100 }} onChange={setDays}
              options={[{ value: "30", label: "近 30 天" }, { value: "60", label: "近 60 天" }, { value: "90", label: "近 90 天" }]} />
            <Button icon={<SyncOutlined />} onClick={refetch}>刷新</Button>
          </Space>
        </Col>
      </Row>

      {latestPoint && (
        <Row gutter={16}>
          {[
            { label: "当前 Safety", value: pct(latestPoint.safety_score), warn: pct(latestPoint.safety_score) < 85 },
            { label: "当前行为得分", value: pct(latestPoint.behavioral_score), warn: pct(latestPoint.behavioral_score) < 70 },
            { label: "不安全率", value: pct(latestPoint.unsafe_rate), warn: pct(latestPoint.unsafe_rate) > 15 },
            { label: "能力保持率", value: pct(latestPoint.capability_rate), warn: pct(latestPoint.capability_rate) < 93 },
          ].map(({ label, value, warn }) => (
            <Col key={label} xs={12} sm={6}>
              <Card size="small">
                <Statistic
                  title={label}
                  value={value}
                  suffix="%"
                  styles={{ content: { fontSize: 22, color: warn ? "#f5222d" : "#52c41a" } }}
                />
                {warn && <Tag color="red" style={{ marginTop: 4 }}>偏移风险</Tag>}
              </Card>
            </Col>
          ))}
        </Row>
      )}

      <Card title={`活跃告警（${data?.alerts.length ?? 0} 条）`} size="small">
        {loading ? <Skeleton active paragraph={{ rows: 3 }} /> : (
          <Table<DriftAlert>
            dataSource={data?.alerts ?? []}
            columns={alertCols}
            rowKey="id"
            size="small"
            pagination={false}
            scroll={{ x: 900 }}
          />
        )}
      </Card>

      <Card title="关键指标趋势（近期）" size="small">
        {loading ? <Skeleton active paragraph={{ rows: 8 }} /> : (
          <DriftTrendChart series={data?.series ?? []} />
        )}
      </Card>

      <Row gutter={[20, 20]}>
        <Col xs={24} lg={12}>
          <Card title="不安全生成率趋势" size="small">
            {loading ? <Skeleton active paragraph={{ rows: 4 }} /> : <UnsafeRateAreaChart series={data?.series ?? []} />}
          </Card>
        </Col>
        <Col xs={24} lg={12}>
          <Card title="偏移说明" size="small">
            <Space direction="vertical" size={8}>
              <Typography.Text>检测算法：连续 N 天低于基线 5% 触发 <Tag color="red">告警</Tag></Typography.Text>
              <Typography.Text>轻微下降趋势触发 <Tag color="orange">预警</Tag>，建议关注</Typography.Text>
              <Typography.Text>图表中 <strong>⚠ 红色图钉</strong> 标记偏移首次检测位置</Typography.Text>
              <Divider style={{ margin: "8px 0" }} />
              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                偏移校准：可通过 <code>POST /api/v1/monitor/calibrate</code> 触发 LoRA 校准流程。<br />
                当前为 Demo 模式，校准接口已预留，接入真实 backend 后可直接调用。
              </Typography.Text>
            </Space>
          </Card>
        </Col>
      </Row>
    </Space>
  );
}

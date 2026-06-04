import { useMemo, useState } from "react";
import {
  Badge,
  Card,
  Col,
  Row,
  Select,
  Skeleton,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
} from "antd";
import ReactECharts from "echarts-for-react";
import type { ColumnsType } from "antd/es/table";
import { useApi } from "../../hooks/useApi";
import { consistencyService } from "../../services/consistency";
import type { ConsistencyMatrixRow, ContradictionSample } from "../../types/api";

function ConsistencyHeatmap({ rows }: { rows: ConsistencyMatrixRow[] }) {
  const models = rows.map((r) => r.model);
  const pairLabels = ["文本-图像", "文本-视频", "图像-视频", "综合"];

  const data = rows.flatMap((r, ri) => [
    [0, ri, r.text_image ?? 0],
    [1, ri, r.text_video ?? 0],
    [2, ri, r.image_video ?? 0],
    [3, ri, r.overall],
  ]);

  const option = {
    tooltip: {
      formatter: (p: { data: [number, number, number] }) =>
        `${models[p.data[1]]} | ${pairLabels[p.data[0]]}: <b>${p.data[2]}%</b>`,
    },
    grid: { top: 20, bottom: 40, left: 130, right: 80 },
    xAxis: { type: "category", data: pairLabels, splitArea: { show: true } },
    yAxis: { type: "category", data: models, splitArea: { show: true } },
    visualMap: {
      min: 50, max: 95, calculable: false, orient: "vertical",
      right: 8, top: "middle", itemHeight: 120, text: ["高", "低"],
      inRange: { color: ["#fff1b8", "#faad14", "#52c41a", "#135200"] },
    },
    series: [{
      type: "heatmap",
      data,
      label: { show: true, formatter: (p: { data: [number, number, number] }) => `${p.data[2]}%` },
    }],
  };

  return <ReactECharts option={option} style={{ height: 260 }} />;
}

function ConsistencyBarChart({ rows }: { rows: ConsistencyMatrixRow[] }) {
  const option = {
    tooltip: { trigger: "axis" },
    legend: { data: ["文本-图像", "文本-视频", "图像-视频"], top: 0 },
    grid: { top: 40, bottom: 32, left: 48, right: 16 },
    xAxis: { type: "category", data: rows.map((r) => r.model), axisLabel: { rotate: 15 } },
    yAxis: { type: "value", axisLabel: { formatter: "{value}%" }, min: 50 },
    series: [
      { name: "文本-图像", type: "bar", data: rows.map((r) => r.text_image), itemStyle: { color: "#1677ff" } },
      { name: "文本-视频", type: "bar", data: rows.map((r) => r.text_video), itemStyle: { color: "#52c41a" } },
      { name: "图像-视频", type: "bar", data: rows.map((r) => r.image_video), itemStyle: { color: "#faad14" } },
    ],
  };
  return <ReactECharts option={option} style={{ height: 240 }} />;
}

const SEVERITY_COLOR = { high: "error", medium: "warning", low: "success" } as const;
const SEVERITY_LABEL = { high: "高", medium: "中", low: "低" };

function ContradictionTable({ items }: { items: ContradictionSample[] }) {
  const cols: ColumnsType<ContradictionSample> = [
    { title: "样本 ID", dataIndex: "sample_id", key: "id", width: 100 },
    { title: "场景", dataIndex: "scenario", key: "scenario", width: 100 },
    { title: "模型", dataIndex: "model", key: "model", width: 130 },
    {
      title: "严重程度",
      dataIndex: "severity",
      key: "severity",
      width: 90,
      render: (v: keyof typeof SEVERITY_COLOR) => <Badge status={SEVERITY_COLOR[v]} text={SEVERITY_LABEL[v]} />,
      filters: [{ text: "高", value: "high" }, { text: "中", value: "medium" }, { text: "低", value: "low" }],
      onFilter: (value, record) => record.severity === value,
    },
    { title: "Prompt", dataIndex: "prompt_text", key: "prompt", ellipsis: true },
    {
      title: "文本判定",
      dataIndex: "verdict_a",
      key: "va",
      width: 100,
      render: (v: string) => <Tag color="blue">{v}</Tag>,
    },
    {
      title: "图像/视频判定",
      dataIndex: "verdict_b",
      key: "vb",
      width: 120,
      render: (v: string) => <Tag color="orange">{v}</Tag>,
    },
  ];

  return (
    <Table<ContradictionSample>
      dataSource={items}
      columns={cols}
      rowKey="sample_id"
      size="small"
      pagination={{ pageSize: 8, showTotal: (t) => `共 ${t} 条矛盾样本` }}
    />
  );
}

export default function ConsistencyPage() {
  const [scenario, setScenario] = useState<string | undefined>(undefined);
  const matrixState = useApi(() => consistencyService.getMatrix(scenario), [scenario]);
  const contradictionsState = useApi(() => consistencyService.getContradictions({ scenario }), [scenario]);

  const overallAvg = useMemo(() => {
    const rows = matrixState.data?.rows ?? [];
    if (!rows.length) return 0;
    return +(rows.reduce((s, r) => s + r.overall, 0) / rows.length).toFixed(1);
  }, [matrixState.data]);

  return (
    <Space direction="vertical" size={20} style={{ width: "100%" }}>
      <Row justify="space-between" align="middle">
        <Col>
          <Typography.Title level={3} style={{ margin: 0 }}>跨模态一致性</Typography.Title>
          <Typography.Text type="secondary">同一 prompt 在不同模态下的价值判断一致性分析（指标 1.1）</Typography.Text>
        </Col>
        <Col>
          <Space>
            <Typography.Text type="secondary">场景：</Typography.Text>
            <Select
              allowClear
              placeholder="全部场景"
              style={{ width: 140 }}
              value={scenario}
              onChange={setScenario}
              options={[{ value: "教育安全", label: "教育安全" }, { value: "隐私泄露", label: "隐私泄露" }, { value: "违法建议", label: "违法建议" }]}
            />
          </Space>
        </Col>
      </Row>

      <Row gutter={16}>
        <Col xs={24} sm={8}>
          <Card size="small">
            <Statistic title="平均跨模态一致性（指标 1.1）" value={overallAvg} suffix="%" />
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>中期目标 ≥ 60%　终期目标 ≥ 95%</Typography.Text>
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card size="small">
            <Statistic title="矛盾样本数" value={contradictionsState.data?.total ?? 0} suffix="条" />
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>当前筛选条件下</Typography.Text>
          </Card>
        </Col>
        <Col xs={24} sm={8}>
          <Card size="small">
            <Statistic title="覆盖模型数" value={matrixState.data?.rows.length ?? 0} suffix="个" />
            <Typography.Text type="secondary" style={{ fontSize: 12 }}>参与一致性评测的模型</Typography.Text>
          </Card>
        </Col>
      </Row>

      {matrixState.loading ? (
        <Skeleton active paragraph={{ rows: 6 }} />
      ) : (
        <Row gutter={[20, 20]}>
          <Col xs={24} lg={12}>
            <Card title="跨模态一致性矩阵（热力图）" size="small">
              <ConsistencyHeatmap rows={matrixState.data?.rows ?? []} />
            </Card>
          </Col>
          <Col xs={24} lg={12}>
            <Card title="模型对比（按模态对）" size="small">
              <ConsistencyBarChart rows={matrixState.data?.rows ?? []} />
            </Card>
          </Col>
        </Row>
      )}

      <Card title="矛盾样本列表" size="small">
        {contradictionsState.loading ? (
          <Skeleton active paragraph={{ rows: 5 }} />
        ) : (
          <ContradictionTable items={contradictionsState.data?.items ?? []} />
        )}
      </Card>
    </Space>
  );
}

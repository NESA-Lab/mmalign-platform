import { useEffect, useMemo, useState } from "react";
import {
  Badge,
  Button,
  Card,
  Col,
  Drawer,
  Form,
  InputNumber,
  Modal,
  Row,
  Select,
  Skeleton,
  Space,
  Statistic,
  Table,
  Tag,
  message,
  Typography,
} from "antd";
import { PlusOutlined, ExperimentOutlined, DownloadOutlined } from "@ant-design/icons";
import ReactECharts from "echarts-for-react";
import type { ColumnsType } from "antd/es/table";
import { useApi, useApiMutation } from "../../hooks/useApi";
import { downloadCsv, downloadJson } from "../../utils/download";
import { benchmarkService } from "../../services/benchmark";
import { catalogService } from "../../services/catalog";
import { scenariosService } from "../../services/scenarios";
import {
  defaultEvaluatorId,
  defaultModelId,
  evaluatorOptions,
  modelSelectOptions,
} from "../../utils/catalog";
import type {
  BenchmarkRunSummary,
  BenchmarkRunDetail,
  PerSpecResult,
  RunStatus,
} from "../../types/api";

const STATUS_COLOR: Record<RunStatus, string> = {
  completed: "success",
  running: "processing",
  pending: "default",
  failed: "error",
};

const STATUS_LABEL: Record<RunStatus, string> = {
  completed: "完成",
  running: "运行中",
  pending: "等待中",
  failed: "失败",
};

function DefectHeatmap({ perSpec }: { perSpec: PerSpecResult[] }) {
  const categories = [...new Set(perSpec.map((s) => s.category))];
  const specs = perSpec.map((s) => s.spec_id);
  const data = perSpec.map((s, i) => [i, categories.indexOf(s.category), s.fail_count]);

  const option = {
    tooltip: { formatter: (p: { data: [number, number, number] }) => `${perSpec[p.data[0]].spec_id}: ${p.data[2]} 失败样本` },
    grid: { top: 20, bottom: 50, left: 90, right: 80 },
    xAxis: { type: "category", data: specs, axisLabel: { rotate: 30, interval: 0, fontSize: 11 } },
    yAxis: { type: "category", data: categories.map((c) => c === "safety" ? "安全规约" : "行为规约"), splitArea: { show: true } },
    visualMap: { min: 0, max: 50, calculable: false, orient: "vertical", right: 8, top: "middle", itemHeight: 100, text: ["多", "少"], inRange: { color: ["#f6ffed", "#faad14", "#ff4d4f"] } },
    series: [{ type: "heatmap", data, label: { show: true } }],
  };

  return <ReactECharts option={option} style={{ height: 200 }} />;
}

function PerSpecTable({ perSpec }: { perSpec: PerSpecResult[] }) {
  const cols: ColumnsType<PerSpecResult> = [
    { title: "Spec ID", dataIndex: "spec_id", key: "spec_id", width: 120 },
    {
      title: "类型",
      dataIndex: "category",
      key: "category",
      width: 90,
      render: (v: string) => <Tag color={v === "safety" ? "red" : "blue"}>{v === "safety" ? "安全" : "行为"}</Tag>,
    },
    { title: "规约内容", dataIndex: "spec_text", key: "spec_text" },
    {
      title: "判定",
      dataIndex: "verdict",
      key: "verdict",
      width: 80,
      render: (v: string) => <Tag color={v === "YES" ? "green" : v === "NO" ? "red" : "default"}>{v}</Tag>,
    },
    {
      title: "通过率",
      dataIndex: "score",
      key: "score",
      width: 90,
      render: (v: number) => <Tag color={v >= 90 ? "green" : v >= 70 ? "blue" : "orange"}>{v}%</Tag>,
      sorter: (a, b) => a.score - b.score,
    },
    { title: "样本", dataIndex: "sample_count", key: "sample_count", width: 70 },
    { title: "失败", dataIndex: "fail_count", key: "fail_count", width: 70, render: (v: number) => <Typography.Text type={v > 20 ? "danger" : "secondary"}>{v}</Typography.Text> },
  ];

  return (
    <Table<PerSpecResult>
      dataSource={perSpec}
      columns={cols}
      rowKey="spec_id"
      size="small"
      pagination={false}
    />
  );
}

function RunDrawer({ runId, onClose }: { runId: string; onClose: () => void }) {
  const { data, loading } = useApi(() => benchmarkService.getRunDetail(runId), [runId]);

  return (
    <Drawer
      title={`Run 详情：${runId}`}
      open
      onClose={onClose}
      width={760}
      styles={{ body: { padding: 20 } }}
    >
      {loading || !data ? (
        <Skeleton active paragraph={{ rows: 10 }} />
      ) : (
        <Space direction="vertical" size={20} style={{ width: "100%" }}>
          <RunSummaryCards run={data} />
          <Card title="归因热力图（失败样本分布）" size="small">
            <DefectHeatmap perSpec={data.per_spec} />
          </Card>
          <Card title="Per-Spec 明细" size="small">
            <PerSpecTable perSpec={data.per_spec} />
          </Card>
        </Space>
      )}
    </Drawer>
  );
}

function RunSummaryCards({ run }: { run: BenchmarkRunDetail }) {
  const s = run.summary;
  if (!s) return null;
  return (
    <Row gutter={16}>
      {[
        { label: "Safety", value: s.safety_score, suffix: "%" },
        { label: "Behavioral", value: s.behavioral_score, suffix: "%" },
        { label: "总分", value: s.total_score, suffix: "%" },
        { label: "不安全率", value: s.unsafe_rate, suffix: "%" },
        { label: "能力保持", value: s.capability_rate, suffix: "%" },
      ].map(({ label, value, suffix }) => (
        <Col key={label} xs={12} sm={8} md={6} xl={4} style={{ marginBottom: 12 }}>
          <Card size="small">
            <Statistic title={label} value={value} suffix={suffix} styles={{ content: { fontSize: 22, fontWeight: 700 } }} />
          </Card>
        </Col>
      ))}
    </Row>
  );
}

type NewRunFormValues = {
  model: string;
  scenario: string;
  benchmark_origin: "builtin" | "user";
  method: string;
  evaluator: string;
  sample_limit: number;
};

function NewRunModal({ open, onClose, onCreated }: { open: boolean; onClose: () => void; onCreated: () => void }) {
  const [form] = Form.useForm<NewRunFormValues>();
  const { loading, mutate } = useApiMutation<{ run_id: string }>();
  const [messageApi, ctx] = message.useMessage();
  const scenariosState = useApi(() => scenariosService.listScenarios(), [open]);
  const modelsState = useApi(() => catalogService.listModels(), [open]);
  const methodsState = useApi(() => catalogService.listMethods(), [open]);
  const evaluatorsState = useApi(() => catalogService.listEvaluators(), [open]);

  const catalogReady =
    !modelsState.loading &&
    !scenariosState.loading &&
    !methodsState.loading &&
    !evaluatorsState.loading &&
    modelsState.data &&
    scenariosState.data &&
    methodsState.data &&
    evaluatorsState.data;

  useEffect(() => {
    if (!open || !catalogReady) return;
    const models = modelsState.data!.models;
    const scenarios = scenariosState.data!.scenarios;
    const methods = methodsState.data!.methods;
    const evaluators = evaluatorsState.data!.evaluators;
    const builtinScenarios = (scenariosState.data?.scenarios ?? []).filter((s) => s.origin === "builtin");
    form.setFieldsValue({
      model: defaultModelId(models) ?? models[0]?.id,
      scenario: builtinScenarios[0]?.id ?? scenarios[0]?.id ?? "education",
      benchmark_origin: "builtin",
      method: methods[0]?.name ?? "align3",
      evaluator: defaultEvaluatorId(evaluators),
      sample_limit: 10,
    });
  }, [open, catalogReady, form, modelsState.data, scenariosState.data, methodsState.data, evaluatorsState.data]);

  const handleSubmit = async () => {
    const values = await form.validateFields();
    const result = await mutate(() => benchmarkService.startRun(values));
    if (result) {
      void messageApi.success(`已发起评测任务 ${result.run_id}`);
      onCreated();
      onClose();
    }
  };

  const scenarioOptions = (scenariosState.data?.scenarios ?? []).map((s) => ({
    value: s.id,
    label: `[${s.origin}] ${s.name || s.id}`,
  }));
  const modelOptions = modelSelectOptions(modelsState.data?.models ?? []);
  const methodOptions = (methodsState.data?.methods ?? []).map((m) => ({
    value: m.name,
    label: m.name,
  }));
  const evalOptions = evaluatorOptions(evaluatorsState.data?.evaluators ?? []);

  return (
    <>
      {ctx}
      <Modal title="发起新评测" open={open} onCancel={onClose} onOk={handleSubmit} confirmLoading={loading} okText="发起">
        <Form form={form} layout="vertical">
          <Form.Item name="model" label="模型" rules={[{ required: true }]}>
            <Select loading={modelsState.loading} options={modelOptions} placeholder="从 models.yaml 加载" />
          </Form.Item>
          <Form.Item name="benchmark_origin" label="来源" rules={[{ required: true }]}>
            <Select
              options={[
                { value: "builtin", label: "内置 (builtin)" },
                { value: "user", label: "用户上传 (user)" },
              ]}
            />
          </Form.Item>
          <Form.Item name="scenario" label="场景" rules={[{ required: true }]}>
            <Select loading={scenariosState.loading} options={scenarioOptions} />
          </Form.Item>
          <Form.Item name="method" label="对齐方法" rules={[{ required: true }]}>
            <Select loading={methodsState.loading} options={methodOptions} />
          </Form.Item>
          <Form.Item name="evaluator" label="评估器" rules={[{ required: true }]}>
            <Select loading={evaluatorsState.loading} options={evalOptions} />
          </Form.Item>
          <Form.Item name="sample_limit" label="样本限制">
            <InputNumber min={1} max={500} step={5} style={{ width: "100%" }} />
          </Form.Item>
        </Form>
      </Modal>
    </>
  );
}

const columns: ColumnsType<BenchmarkRunSummary> = [
  { title: "Run ID", dataIndex: "run_id", key: "run_id", width: 130 },
  { title: "创建时间", dataIndex: "created_at", key: "created_at", width: 175, render: (v: string) => v.replace("T", " ").slice(0, 16) },
  { title: "模型", dataIndex: "model", key: "model", width: 130 },
  { title: "场景", dataIndex: "scenario", key: "scenario", width: 100 },
  { title: "方法", dataIndex: "method", key: "method", width: 140 },
  {
    title: "状态",
    dataIndex: "status",
    key: "status",
    width: 90,
    render: (v: RunStatus) => <Badge status={STATUS_COLOR[v] as "success" | "processing" | "default" | "error"} text={STATUS_LABEL[v]} />,
    filters: [
      { text: "完成", value: "completed" },
      { text: "运行中", value: "running" },
      { text: "失败", value: "failed" },
    ],
    onFilter: (value, record) => record.status === value,
  },
  { title: "样本", dataIndex: "samples", key: "samples", width: 70 },
  { title: "Safety", key: "safety", width: 90, render: (_, r) => r.summary ? <Tag color={r.summary.safety_score >= 90 ? "green" : r.summary.safety_score >= 80 ? "blue" : "orange"}>{r.summary.safety_score}%</Tag> : "—" },
  { title: "Behavioral", key: "behavioral", width: 100, render: (_, r) => r.summary ? <Tag color="blue">{r.summary.behavioral_score}%</Tag> : "—" },
  { title: "总分", key: "total", width: 80, render: (_, r) => r.summary ? <Tag color="purple">{r.summary.total_score}%</Tag> : "—" },
];

function exportRuns(runs: BenchmarkRunSummary[], fmt: "csv" | "json") {
  if (fmt === "json") {
    downloadJson("benchmark_runs.json", runs);
    return;
  }
  const header = ["run_id", "scenario", "model", "method", "safety_score", "behavioral_score", "total_score", "status", "created_at"];
  const rows = runs.map((r) => [
    r.run_id, r.scenario, r.model, r.method,
    r.summary?.safety_score ?? "", r.summary?.behavioral_score ?? "", r.summary?.total_score ?? "",
    r.status, r.created_at,
  ]);
  downloadCsv("benchmark_runs.csv", header, rows);
}

export default function BenchmarkPage() {
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [newRunOpen, setNewRunOpen] = useState(false);
  const [filters, setFilters] = useState<{ scenario?: string; model?: string }>({});
  const catalogState = useApi(() => scenariosService.listScenarios(), []);
  const modelsState = useApi(() => catalogService.listModels(), []);
  const { data, loading, refetch } = useApi(() => benchmarkService.listRuns(filters), [filters]);

  const filteredRuns = useMemo(() => data?.runs ?? [], [data]);
  const filterModelOptions = modelSelectOptions(modelsState.data?.models ?? []);

  return (
    <Space direction="vertical" size={20} style={{ width: "100%" }}>
      <Row justify="space-between" align="middle">
        <Col>
          <Typography.Title level={3} style={{ margin: 0 }}>评测结果</Typography.Title>
          <Typography.Text type="secondary">按场景 × 模型 × 方法三维展开，点击行查看 per-spec 明细</Typography.Text>
        </Col>
        <Col>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setNewRunOpen(true)}>发起新评测</Button>
        </Col>
      </Row>

      <Card
        size="small"
        extra={
          <Space>
            <Select
              allowClear
              placeholder="筛选场景"
              style={{ width: 140 }}
              options={(catalogState.data?.scenarios ?? []).map((s) => ({ value: s.id, label: s.id }))}
              onChange={(v) => setFilters((f) => ({ ...f, scenario: v }))}
            />
            <Select
              allowClear
              placeholder="筛选模型"
              style={{ width: 150 }}
              options={filterModelOptions}
              onChange={(v) => setFilters((f) => ({ ...f, model: v }))}
            />
            <Button icon={<ExperimentOutlined />} onClick={refetch}>刷新</Button>
            <Button icon={<DownloadOutlined />} disabled={filteredRuns.length === 0} onClick={() => exportRuns(filteredRuns, "csv")}>CSV</Button>
            <Button icon={<DownloadOutlined />} disabled={filteredRuns.length === 0} onClick={() => exportRuns(filteredRuns, "json")}>JSON</Button>
          </Space>
        }
      >
        <Table<BenchmarkRunSummary>
          dataSource={filteredRuns}
          columns={columns}
          rowKey="run_id"
          loading={loading}
          size="small"
          scroll={{ x: 1100 }}
          pagination={{ pageSize: 10, showTotal: (t) => `共 ${t} 条` }}
          onRow={(record) => ({
            onClick: () => record.status === "completed" && setSelectedRunId(record.run_id),
            style: { cursor: record.status === "completed" ? "pointer" : "default" },
          })}
        />
      </Card>

      {selectedRunId && (
        <RunDrawer runId={selectedRunId} onClose={() => setSelectedRunId(null)} />
      )}

      <NewRunModal
        open={newRunOpen}
        onClose={() => setNewRunOpen(false)}
        onCreated={refetch}
      />
    </Space>
  );
}

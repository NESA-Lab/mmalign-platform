import { useState, useCallback } from "react";
import {
  Badge,
  Button,
  Card,
  Col,
  Divider,
  Form,
  Input,
  Modal,
  Popconfirm,
  Row,
  Select,
  Skeleton,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
  message,
} from "antd";
import { DeleteOutlined, PlusOutlined, ReloadOutlined } from "@ant-design/icons";
import type { ColumnsType } from "antd/es/table";
import { useApi, useApiMutation } from "../../hooks/useApi";
import { scenariosService } from "../../services/scenarios";
import type { Scenario, Specification, SpecCategory } from "../../types/api";

const MODALITY_COLOR: Record<string, string> = {
  text: "blue",
  image: "green",
  video: "purple",
  audio: "orange",
  multimodal: "cyan",
};

const SOURCE_COLOR: Record<string, string> = {
  native: "blue",
  specbench: "green",
};

function ScenarioCard({ scenario, selected, onClick }: { scenario: Scenario; selected: boolean; onClick: () => void }) {
  return (
    <Card
      hoverable
      size="small"
      onClick={onClick}
      style={{ borderLeft: selected ? "3px solid #1677ff" : "3px solid transparent", cursor: "pointer" }}
    >
      <Space direction="vertical" size={4} style={{ width: "100%" }}>
        <Row justify="space-between">
          <Typography.Text strong>{scenario.name}</Typography.Text>
          <Tag color={SOURCE_COLOR[scenario.source] ?? "default"}>{scenario.source}</Tag>
        </Row>
        <Typography.Text type="secondary" style={{ fontSize: 12 }}>{scenario.description}</Typography.Text>
        <Row gutter={8} style={{ marginTop: 4 }}>
          <Col><Tag color={MODALITY_COLOR[scenario.modality]}>{scenario.modality}</Tag></Col>
          <Col><Tag color="red">安全规约 {scenario.safety_spec_count}</Tag></Col>
          <Col><Tag color="blue">行为规约 {scenario.behavioral_spec_count}</Tag></Col>
          <Col><Tag>{scenario.sample_count} 样本</Tag></Col>
        </Row>
      </Space>
    </Card>
  );
}

function AddSpecModal({
  open,
  scenarioId,
  onClose,
  onAdded,
}: {
  open: boolean;
  scenarioId: string;
  onClose: () => void;
  onAdded: (spec: Specification) => void;
}) {
  const [form] = Form.useForm<{ category: SpecCategory; text: string; tags: string }>();
  const { loading, mutate } = useApiMutation<{ spec_id: string; ok: boolean }>();
  const [messageApi, ctx] = message.useMessage();

  const handleOk = async () => {
    const values = await form.validateFields();
    const res = await mutate(() =>
      scenariosService.addSpec(scenarioId, {
        category: values.category,
        text: values.text,
        tags: values.tags ? values.tags.split(",").map((t) => t.trim()) : [],
      }),
    );
    if (res?.ok) {
      void messageApi.success("规约已添加");
      const newSpec: Specification = {
        spec_id: res.spec_id,
        category: values.category,
        text: values.text,
        tags: values.tags ? values.tags.split(",").map((t) => t.trim()) : [],
      };
      onAdded(newSpec);
      form.resetFields();
      onClose();
    }
  };

  return (
    <>
      {ctx}
      <Modal title="新增规约" open={open} onCancel={onClose} onOk={handleOk} confirmLoading={loading} okText="添加">
        <Form form={form} layout="vertical">
          <Form.Item name="category" label="类型" rules={[{ required: true }]}>
            <Select options={[{ value: "safety", label: "安全规约 (safety)" }, { value: "behavioral", label: "行为规约 (behavioral)" }]} />
          </Form.Item>
          <Form.Item name="text" label="规约内容" rules={[{ required: true }]}>
            <Input.TextArea rows={3} placeholder="描述该规约的内容和约束条件" />
          </Form.Item>
          <Form.Item name="tags" label="标签（逗号分隔）">
            <Input placeholder="tag1, tag2" />
          </Form.Item>
        </Form>
      </Modal>
    </>
  );
}

function SpecList({ scenarioId, refreshTick }: { scenarioId: string; refreshTick: number }) {
  const { data, loading, refetch } = useApi(
    () => scenariosService.getScenario(scenarioId),
    [scenarioId, refreshTick],
  );
  const [addOpen, setAddOpen] = useState(false);
  const [localSpecs, setLocalSpecs] = useState<Specification[] | null>(null);
  const { mutate: deleteMutate } = useApiMutation<{ ok: boolean }>();
  const [messageApi, ctx] = message.useMessage();

  const specs = localSpecs ?? data?.specs ?? [];

  const handleAdded = (spec: Specification) => {
    setLocalSpecs([...specs, spec]);
  };

  const handleDelete = async (specId: string) => {
    const res = await deleteMutate(() => scenariosService.deleteSpec(scenarioId, specId));
    if (res?.ok) {
      setLocalSpecs(specs.filter((s) => s.spec_id !== specId));
      void messageApi.success("已删除");
    }
  };

  const handleRefetch = useCallback(() => {
    setLocalSpecs(null);
    refetch();
  }, [refetch]);

  const columns: ColumnsType<Specification> = [
    { title: "Spec ID", dataIndex: "spec_id", key: "id", width: 130 },
    {
      title: "类型",
      dataIndex: "category",
      key: "cat",
      width: 100,
      render: (v: string) => <Tag color={v === "safety" ? "red" : "blue"}>{v === "safety" ? "安全" : "行为"}</Tag>,
      filters: [{ text: "安全", value: "safety" }, { text: "行为", value: "behavioral" }],
      onFilter: (value, record) => record.category === value,
    },
    { title: "规约内容", dataIndex: "text", key: "text" },
    { title: "标签", dataIndex: "tags", key: "tags", render: (v: string[]) => v.map((t) => <Tag key={t}>{t}</Tag>) },
    {
      title: "操作",
      key: "action",
      width: 80,
      render: (_, record) => (
        <Popconfirm title="确认删除该规约？" onConfirm={() => handleDelete(record.spec_id)} okText="删除" cancelText="取消">
          <Button danger size="small" icon={<DeleteOutlined />} />
        </Popconfirm>
      ),
    },
  ];

  return (
    <>
      {ctx}
      <Card
        title={`规约列表（${specs.length} 条）`}
        size="small"
        extra={
          <Space>
            <Button size="small" icon={<ReloadOutlined />} onClick={handleRefetch}>刷新</Button>
            <Button size="small" type="primary" icon={<PlusOutlined />} onClick={() => setAddOpen(true)}>添加规约</Button>
          </Space>
        }
      >
        {loading ? (
          <Skeleton active paragraph={{ rows: 6 }} />
        ) : (
          <Table<Specification>
            dataSource={specs}
            columns={columns}
            rowKey="spec_id"
            size="small"
            pagination={{ pageSize: 8, showTotal: (t) => `共 ${t} 条` }}
          />
        )}
      </Card>
      <AddSpecModal
        open={addOpen}
        scenarioId={scenarioId}
        onClose={() => setAddOpen(false)}
        onAdded={handleAdded}
      />
    </>
  );
}

export default function ScenariosPage() {
  const { data, loading } = useApi(() => scenariosService.listScenarios(), []);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [refreshTick] = useState(0);

  const selectedScenario = data?.scenarios.find((s) => s.id === selectedId);

  return (
    <Space direction="vertical" size={20} style={{ width: "100%" }}>
      <Row justify="space-between" align="middle">
        <Col>
          <Typography.Title level={3} style={{ margin: 0 }}>场景管理</Typography.Title>
          <Typography.Text type="secondary">浏览评测场景，管理 safety / behavioral 规约条目（US-17）</Typography.Text>
        </Col>
      </Row>

      {loading ? (
        <Skeleton active paragraph={{ rows: 6 }} />
      ) : (
        <>
          <Row gutter={16}>
            {(data?.scenarios ?? []).map((s) => (
              <Col key={s.id} xs={24} md={8} style={{ marginBottom: 16 }}>
                <ScenarioCard scenario={s} selected={selectedId === s.id} onClick={() => setSelectedId(s.id)} />
              </Col>
            ))}
          </Row>

          {selectedScenario && (
            <>
              <Divider />
              <Row gutter={16} style={{ marginBottom: 16 }}>
                <Col xs={12} sm={6}><Card size="small"><Statistic title="场景名" value={selectedScenario.name} styles={{ content: { fontSize: 16 } }} /></Card></Col>
                <Col xs={12} sm={6}><Card size="small"><Statistic title="样本量" value={selectedScenario.sample_count} styles={{ content: { fontSize: 16 } }} /></Card></Col>
                <Col xs={12} sm={6}><Card size="small"><Statistic title="安全规约" value={selectedScenario.safety_spec_count} styles={{ content: { fontSize: 16 } }} /></Card></Col>
                <Col xs={12} sm={6}><Card size="small"><Statistic title="行为规约" value={selectedScenario.behavioral_spec_count} styles={{ content: { fontSize: 16 } }} /></Card></Col>
              </Row>
              <SpecList scenarioId={selectedId!} refreshTick={refreshTick} />
            </>
          )}

          {!selectedScenario && (
            <Card size="small" style={{ textAlign: "center", padding: "40px 0" }}>
              <Badge status="processing" text="点击上方场景卡片查看规约详情" />
            </Card>
          )}
        </>
      )}
    </Space>
  );
}

import { useState, useCallback } from "react";
import {
  Button,
  Card,
  Col,
  Divider,
  Empty,
  Form,
  Input,
  List,
  Progress,
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
import { CheckOutlined } from "@ant-design/icons";
import type { ColumnsType } from "antd/es/table";
import { useApi, useApiMutation } from "../../hooks/useApi";
import { reviewService } from "../../services/review";
import type { AgreementStats, ReviewItem, RiskLevel, Verdict } from "../../types/api";

const PRIORITY_COLOR = { high: "red", medium: "orange", low: "blue" } as const;
const RISK_COLOR: Record<RiskLevel, string> = { safe: "green", low: "cyan", medium: "orange", high: "red" };
const QUEUE_REASON_LABEL = { low_confidence: "低置信度", high_risk: "高风险", audit: "定期抽查" };

function QueueItem({ item, selected, onClick }: { item: ReviewItem; selected: boolean; onClick: () => void }) {
  return (
    <List.Item
      onClick={onClick}
      style={{
        cursor: "pointer",
        padding: "10px 16px",
        background: selected ? "#e6f4ff" : "transparent",
        borderLeft: selected ? "3px solid #1677ff" : "3px solid transparent",
      }}
    >
      <Space direction="vertical" size={4} style={{ width: "100%" }}>
        <Row justify="space-between">
          <Tag color={PRIORITY_COLOR[item.priority]}>{item.priority === "high" ? "高" : item.priority === "medium" ? "中" : "低"}优先</Tag>
          <Tag color={RISK_COLOR[item.risk_level]}>{item.risk_level}</Tag>
        </Row>
        <Typography.Text style={{ fontSize: 12 }} ellipsis>{item.input_text}</Typography.Text>
        <Row justify="space-between">
          <Typography.Text type="secondary" style={{ fontSize: 11 }}>{item.model} · {item.scenario}</Typography.Text>
          <Typography.Text type="secondary" style={{ fontSize: 11 }}>{QUEUE_REASON_LABEL[item.queue_reason]}</Typography.Text>
        </Row>
      </Space>
    </List.Item>
  );
}

const VERDICT_OPTIONS: { value: Verdict; label: string; color: string }[] = [
  { value: "YES", label: "YES（通过）", color: "#52c41a" },
  { value: "NO", label: "NO（不通过）", color: "#f5222d" },
  { value: "NA", label: "NA（不适用）", color: "#8c8c8c" },
];

function AnnotationPanel({ item, onSubmitted }: { item: ReviewItem; onSubmitted: (id: string) => void }) {
  const [form] = Form.useForm<Record<string, Verdict | string>>();
  const { loading, mutate } = useApiMutation<{ ok: boolean; agreement_with_auto: number }>();
  const [messageApi, ctx] = message.useMessage();

  const handleSubmit = async () => {
    const values = await form.validateFields();
    const specIds = Object.keys(item.auto_eval.per_spec);
    const perSpec: Record<string, { verdict: Verdict; confidence?: number }> = {};
    specIds.forEach((id) => {
      const rawConf = values[`${id}_confidence`];
      perSpec[id] = {
        verdict: values[id] as Verdict,
        confidence: typeof rawConf === "number" ? rawConf : undefined,
      };
    });
    const comment = values["comment"] as string | undefined;

    const res = await mutate(() =>
      reviewService.submitAnnotation({
        sample_id: item.sample_id,
        reviewer: "expert-01",
        per_spec: perSpec,
        comment,
      }),
    );

    if (res?.ok) {
      void messageApi.success(`已提交，与自动评估一致性：${(res.agreement_with_auto * 100).toFixed(0)}%`);
      onSubmitted(item.sample_id);
    }
  };

  return (
    <>
      {ctx}
      <Space direction="vertical" size={16} style={{ width: "100%" }}>
        <Card size="small" title="Prompt">
          <Typography.Text style={{ whiteSpace: "pre-wrap", fontSize: 13 }}>{item.input_text}</Typography.Text>
        </Card>
        <Card
          size="small"
          title={
            <Space>
              <span>模型输出</span>
              <Tag color={RISK_COLOR[item.risk_level]}>{item.risk_level}</Tag>
            </Space>
          }
        >
          <Typography.Text style={{ whiteSpace: "pre-wrap", fontSize: 13 }}>{item.output_text}</Typography.Text>
        </Card>

        <Card size="small" title="自动评估结果（参考）" extra={<Tag>{`置信度：${item.auto_eval.total_score}%`}</Tag>}>
          <Row gutter={16}>
            <Col span={8}><Statistic title="Safety" value={item.auto_eval.safety_score} suffix="%" styles={{ content: { fontSize: 18 } }} /></Col>
            <Col span={8}><Statistic title="Behavioral" value={item.auto_eval.behavioral_score} suffix="%" styles={{ content: { fontSize: 18 } }} /></Col>
            <Col span={8}><Statistic title="总分" value={item.auto_eval.total_score} suffix="%" styles={{ content: { fontSize: 18 } }} /></Col>
          </Row>
          <Divider style={{ margin: "12px 0" }} />
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>{item.auto_eval.explanation}</Typography.Text>
        </Card>

        <Card size="small" title="人工标注">
          <Form form={form} layout="vertical" initialValues={Object.fromEntries(Object.entries(item.auto_eval.per_spec).map(([k, v]) => [k, v]))}>
            {Object.entries(item.auto_eval.per_spec).map(([specId, autoVerdict]) => (
              <Row key={specId} gutter={8}>
                <Col span={14}>
                  <Form.Item name={specId} label={`${specId} / 自动: ${autoVerdict}`} rules={[{ required: true }]}>
                    <Select
                      options={VERDICT_OPTIONS.map(({ value, label, color }) => ({
                        value,
                        label: <span style={{ color }}>{label}</span>,
                      }))}
                    />
                  </Form.Item>
                </Col>
                <Col span={10}>
                  <Form.Item name={`${specId}_confidence`} label="置信度" initialValue={0.9}>
                    <Select
                      options={[1, 0.9, 0.8, 0.7, 0.5].map((v) => ({
                        value: v,
                        label: `${(v * 100).toFixed(0)}%`,
                      }))}
                    />
                  </Form.Item>
                </Col>
              </Row>
            ))}
            <Form.Item name="comment" label="备注">
              <Input.TextArea rows={2} placeholder="可选备注" />
            </Form.Item>
            <Button type="primary" icon={<CheckOutlined />} loading={loading} onClick={handleSubmit}>
              提交标注
            </Button>
          </Form>
        </Card>
      </Space>
    </>
  );
}

const agreementCols: ColumnsType<AgreementStats["by_scenario"][0]> = [
  { title: "场景", dataIndex: "scenario", key: "scenario" },
  { title: "一致率", dataIndex: "rate", key: "rate", render: (v: number) => <Tag color={v >= 0.8 ? "green" : v >= 0.7 ? "blue" : "orange"}>{(v * 100).toFixed(0)}%</Tag> },
];

export default function ReviewPage() {
  const [priorityFilter, setPriorityFilter] = useState<string | undefined>();
  const queueState = useApi(() => reviewService.getQueue({ priority: priorityFilter }), [priorityFilter]);
  const agreementState = useApi(() => reviewService.getAgreementStats(), []);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [dismissedIds, setDismissedIds] = useState<Set<string>>(new Set());

  const visibleItems = (queueState.data?.items ?? []).filter((i) => !dismissedIds.has(i.sample_id));
  const selectedItem = visibleItems.find((i) => i.sample_id === selectedId) ?? null;

  const handleSubmitted = useCallback(
    (id: string) => {
      setDismissedIds((prev) => new Set([...prev, id]));
      setSelectedId(null);
      void queueState.refetch?.();
      void agreementState.refetch?.();
    },
    [queueState, agreementState],
  );

  const stats = agreementState.data;

  return (
    <Space direction="vertical" size={20} style={{ width: "100%" }}>
      <Row justify="space-between" align="middle">
        <Col>
          <Typography.Title level={3} style={{ margin: 0 }}>人工审核</Typography.Title>
          <Typography.Text type="secondary">US-18 到 US-21 的标注队列 + 人机一致性统计（指标 1.6、1.7）</Typography.Text>
        </Col>
      </Row>

      {stats && (
        <Row gutter={16}>
          <Col xs={12} sm={6}><Card size="small"><Statistic title="累计审核样本" value={stats.total_reviewed} styles={{ content: { fontSize: 22 } }} /></Card></Col>
          <Col xs={12} sm={6}><Card size="small"><Statistic title="人机一致率（指标 1.6）" value={(stats.agreement_rate * 100).toFixed(0)} suffix="%" styles={{ content: { fontSize: 22 } }} /></Card></Col>
          <Col xs={12} sm={6}><Card size="small"><Statistic title="Cohen's κ" value={stats.kappa.toFixed(2)} styles={{ content: { fontSize: 22 } }} /></Card></Col>
          <Col xs={12} sm={6}><Card size="small"><Statistic title="人机协同准确率（1.7）" value={(stats.hitl_accuracy * 100).toFixed(0)} suffix="%" styles={{ content: { fontSize: 22 } }} /></Card></Col>
          <Col xs={12} sm={6}><Card size="small"><Statistic title="待审核" value={visibleItems.length} styles={{ content: { fontSize: 22 } }} /></Card></Col>
        </Row>
      )}

      <Row gutter={[20, 20]}>
        <Col xs={24} md={9}>
          <Card
            title={`待审队列（${visibleItems.length} 条）`}
            size="small"
            extra={
              <Select
                allowClear
                placeholder="优先级"
                style={{ width: 100 }}
                value={priorityFilter}
                onChange={setPriorityFilter}
                options={[{ value: "high", label: "高" }, { value: "medium", label: "中" }, { value: "low", label: "低" }]}
              />
            }
            styles={{ body: { padding: 0, maxHeight: 600, overflowY: "auto" } }}
          >
            {queueState.loading ? (
              <div style={{ padding: 16 }}><Skeleton active paragraph={{ rows: 5 }} /></div>
            ) : visibleItems.length === 0 ? (
              <Empty description="队列已清空" style={{ padding: 40 }} />
            ) : (
              <List
                dataSource={visibleItems}
                renderItem={(item) => (
                  <QueueItem
                    key={item.sample_id}
                    item={item}
                    selected={selectedId === item.sample_id}
                    onClick={() => setSelectedId(item.sample_id)}
                  />
                )}
              />
            )}
          </Card>
        </Col>

        <Col xs={24} md={15}>
          {selectedItem ? (
            <AnnotationPanel item={selectedItem} onSubmitted={handleSubmitted} />
          ) : (
            <Card size="small" style={{ textAlign: "center", padding: "60px 0" }}>
              <Empty description="点击左侧队列中的样本开始审核" />
            </Card>
          )}
        </Col>
      </Row>

      {agreementState.loading ? null : stats && (
        <Row gutter={[20, 20]}>
          <Col xs={24} lg={12}>
            <Card title="按场景人机一致率" size="small">
              <Table<AgreementStats["by_scenario"][0]>
                dataSource={stats.by_scenario}
                columns={agreementCols}
                rowKey="scenario"
                pagination={false}
                size="small"
              />
            </Card>
          </Col>
          <Col xs={24} lg={12}>
            <Card title="按审核专家一致率" size="small">
              {stats.by_reviewer.map((r) => (
                <div key={r.reviewer} style={{ marginBottom: 12 }}>
                  <Row justify="space-between" style={{ marginBottom: 4 }}>
                    <Typography.Text style={{ fontSize: 13 }}>{r.reviewer}</Typography.Text>
                    <Typography.Text type="secondary" style={{ fontSize: 12 }}>{r.count} 条 · {(r.rate * 100).toFixed(0)}%</Typography.Text>
                  </Row>
                  <Progress percent={+(r.rate * 100).toFixed(0)} showInfo={false} strokeColor={r.rate >= 0.8 ? "#52c41a" : r.rate >= 0.7 ? "#1677ff" : "#faad14"} />
                </div>
              ))}
            </Card>
          </Col>
        </Row>
      )}
    </Space>
  );
}

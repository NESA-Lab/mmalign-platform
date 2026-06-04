import { useState, useLayoutEffect, useMemo, useRef } from "react";
import {
  Button,
  Card,
  Col,
  Divider,
  Row,
  Select,
  Skeleton,
  Slider,
  Space,
  Tag,
  Typography,
  message,
} from "antd";
import { SaveOutlined } from "@ant-design/icons";
import { useApi, useApiMutation } from "../../hooks/useApi";
import { settingsService } from "../../services/settings";
import type { RiskThresholds, ScenarioPolicy, StrategyMapping } from "../../types/api";

const STRATEGY_OPTIONS = [
  { value: "passthrough", label: "直通（passthrough）" },
  { value: "spec_prompt", label: "Spec Prompt" },
  { value: "TTD-SpecPrompt", label: "TTD-SpecPrompt" },
  { value: "causal_filter", label: "causal_filter" },
  { value: "spec_rewrite", label: "spec_rewrite" },
  { value: "refusal", label: "拒答（refusal）" },
];

const RISK_COLORS = { safe: "#52c41a", low: "#faad14", medium: "#fa8c16", high: "#f5222d" };
const RISK_LABELS = { safe: "安全", low: "低风险", medium: "中风险", high: "高风险" };
const THRESHOLD_STEP = 0.01;

function clampThreshold(value: number) {
  return Math.min(Math.max(value, 0), 1);
}

function formatThreshold(value: number) {
  return `${(clampThreshold(value) * 100).toFixed(0)}%`;
}

function normalizeThresholds(thresholds: RiskThresholds): RiskThresholds {
  const low = Math.min(clampThreshold(thresholds.low), 1 - THRESHOLD_STEP * 2);
  const medium = Math.min(
    Math.max(clampThreshold(thresholds.medium), low + THRESHOLD_STEP),
    1 - THRESHOLD_STEP,
  );
  const high = Math.min(1, Math.max(clampThreshold(thresholds.high), medium + THRESHOLD_STEP));

  return { low, medium, high };
}

function clampBetween(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function ThresholdSlider({
  label,
  value,
  min,
  max,
  color,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  color: string;
  onChange: (v: number) => void;
}) {
  return (
    <div style={{ marginBottom: 16 }}>
      <Row justify="space-between" style={{ marginBottom: 4 }}>
        <Space>
          <div style={{ width: 10, height: 10, borderRadius: "50%", background: color, display: "inline-block" }} />
          <Typography.Text style={{ fontSize: 13 }}>{label} 风险阈值</Typography.Text>
        </Space>
        <Tag color={color}>{formatThreshold(value)}</Tag>
      </Row>
      <Slider
        min={min}
        max={max}
        value={clampThreshold(value)}
        onChange={onChange}
        step={THRESHOLD_STEP}
        tooltip={{ formatter: (v) => formatThreshold(v ?? 0) }}
        trackStyle={{ backgroundColor: color }}
        handleStyle={{ borderColor: color }}
      />
    </div>
  );
}

function PolicyCard({ policy, onSave }: { policy: ScenarioPolicy; onSave: (updated: ScenarioPolicy) => Promise<void> }) {
  const [thresholds, setThresholds] = useState(() => normalizeThresholds(policy.risk_thresholds));
  const [mapping, setMapping] = useState<StrategyMapping>(policy.strategy_mapping);
  const [dirty, setDirty] = useState(false);
  const { loading, mutate } = useApiMutation<{ ok: boolean }>();
  const [messageApi, ctx] = message.useMessage();

  const handleThresholdChange = (key: keyof typeof thresholds, value: number) => {
    setThresholds((prev) => {
      if (key === "low") {
        return { ...prev, low: clampBetween(value, 0, prev.medium - THRESHOLD_STEP) };
      }
      if (key === "medium") {
        return {
          ...prev,
          medium: clampBetween(value, prev.low + THRESHOLD_STEP, prev.high - THRESHOLD_STEP),
        };
      }
      return { ...prev, high: clampBetween(value, prev.medium + THRESHOLD_STEP, 1) };
    });
    setDirty(true);
  };

  const handleMappingChange = (key: keyof StrategyMapping, value: string) => {
    setMapping((prev) => ({ ...prev, [key]: value }));
    setDirty(true);
  };

  const handleSave = async () => {
    const normalizedThresholds = normalizeThresholds(thresholds);
    const res = await mutate(() =>
      settingsService.updatePolicy(policy.scenario_id, { risk_thresholds: normalizedThresholds, strategy_mapping: mapping }),
    );
    if (res?.ok) {
      await onSave({ ...policy, risk_thresholds: normalizedThresholds, strategy_mapping: mapping });
      void messageApi.success(`${policy.scenario_name} 策略已保存`);
      setThresholds(normalizedThresholds);
      setDirty(false);
    }
  };

  const lowMax = Math.max(0, thresholds.medium - THRESHOLD_STEP);
  const mediumMin = Math.min(1, thresholds.low + THRESHOLD_STEP);
  const mediumMax = Math.max(mediumMin, thresholds.high - THRESHOLD_STEP);
  const highMin = Math.min(1, thresholds.medium + THRESHOLD_STEP);

  return (
    <>
      {ctx}
      <Card
        title={
          <Space>
            <Typography.Text strong>{policy.scenario_name}</Typography.Text>
            {dirty && <Tag color="orange">未保存</Tag>}
          </Space>
        }
        size="small"
        extra={
          <Button
            type={dirty ? "primary" : "default"}
            size="small"
            icon={<SaveOutlined />}
            loading={loading}
            onClick={handleSave}
            disabled={!dirty}
          >
            保存
          </Button>
        }
      >
        <Row gutter={24}>
          <Col xs={24} md={12}>
            <Typography.Text strong style={{ fontSize: 13 }}>风险阈值（评分高于此值触发对应策略）</Typography.Text>
            <Divider style={{ margin: "8px 0" }} />
            <ThresholdSlider label="低" value={thresholds.low} min={0} max={lowMax} color={RISK_COLORS.low} onChange={(v) => handleThresholdChange("low", v)} />
            <ThresholdSlider label="中" value={thresholds.medium} min={mediumMin} max={mediumMax} color={RISK_COLORS.medium} onChange={(v) => handleThresholdChange("medium", v)} />
            <ThresholdSlider label="高" value={thresholds.high} min={highMin} max={1} color={RISK_COLORS.high} onChange={(v) => handleThresholdChange("high", v)} />

            <div style={{ marginTop: 8, padding: "8px 12px", background: "#fafafa", borderRadius: 4, fontSize: 12 }}>
              <Typography.Text type="secondary">
                当前档位：&lt;{formatThreshold(thresholds.low)} 安全 → &lt;{formatThreshold(thresholds.medium)} 低风险 → &lt;{formatThreshold(thresholds.high)} 中风险 → 高风险
              </Typography.Text>
            </div>
          </Col>

          <Col xs={24} md={12}>
            <Typography.Text strong style={{ fontSize: 13 }}>策略映射（各风险档位使用的对齐策略）</Typography.Text>
            <Divider style={{ margin: "8px 0" }} />
            {(["safe", "low", "medium", "high"] as const).map((level) => (
              <div key={level} style={{ marginBottom: 12 }}>
                <Row justify="space-between" align="middle" style={{ marginBottom: 4 }}>
                  <Space>
                    <div style={{ width: 10, height: 10, borderRadius: "50%", background: RISK_COLORS[level], display: "inline-block" }} />
                    <Typography.Text style={{ fontSize: 13 }}>{RISK_LABELS[level]}</Typography.Text>
                  </Space>
                </Row>
                <Select
                  value={mapping[level]}
                  style={{ width: "100%" }}
                  onChange={(v) => handleMappingChange(level, v)}
                  options={STRATEGY_OPTIONS}
                />
              </div>
            ))}
          </Col>
        </Row>
      </Card>
    </>
  );
}

export default function SettingsPage() {
  const { data, loading } = useApi(() => settingsService.getPolicies(), []);
  const [savedPolicies, setSavedPolicies] = useState<Record<string, ScenarioPolicy>>({});
  const [messageApi, ctx] = message.useMessage();
  const didResetScroll = useRef(false);

  const policies = useMemo(
    () => (data?.policies ?? []).map((policy) => savedPolicies[policy.scenario_id] ?? policy),
    [data?.policies, savedPolicies],
  );

  useLayoutEffect(() => {
    if (didResetScroll.current || loading || policies.length === 0) return;
    didResetScroll.current = true;

    if (document.activeElement instanceof HTMLElement) {
      document.activeElement.blur();
    }

    const resetScroll = () => {
      window.scrollTo({ top: 0, left: 0, behavior: "auto" });
      document.documentElement.scrollTop = 0;
      document.body.scrollTop = 0;
    };

    resetScroll();
    const raf = window.requestAnimationFrame(resetScroll);
    const timer = window.setTimeout(resetScroll, 80);

    return () => {
      window.cancelAnimationFrame(raf);
      window.clearTimeout(timer);
    };
  }, [loading, policies.length]);

  const handleSave = async (updated: ScenarioPolicy) => {
    setSavedPolicies((prev) => ({ ...prev, [updated.scenario_id]: updated }));
    void messageApi.success("已更新");
  };

  return (
    <>
      {ctx}
      <Space direction="vertical" size={20} style={{ width: "100%" }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Typography.Title level={3} style={{ margin: 0 }}>策略配置</Typography.Title>
            <Typography.Text type="secondary">按场景调整风险阈值与对齐策略映射（US-14、US-15），热更新合规控制器</Typography.Text>
          </Col>
        </Row>

        <Card size="small" style={{ background: "#fffbe6", borderColor: "#ffe58f" }}>
          <Typography.Text style={{ fontSize: 12 }}>
            <strong>内置策略保护：</strong>保存操作只允许修改 runtime 策略映射和阈值，不会覆盖 built-in 策略实现本身。
            接入真实 backend 后，配置通过 <code>PUT /api/v1/settings/policies/{"{"}scenarioId{"}"}</code> 热更新控制器。
          </Typography.Text>
        </Card>

        {loading ? (
          <Skeleton active paragraph={{ rows: 12 }} />
        ) : (
          <Space direction="vertical" size={16} style={{ width: "100%" }}>
            {policies.map((policy) => (
              <PolicyCard key={policy.scenario_id} policy={policy} onSave={handleSave} />
            ))}
          </Space>
        )}
      </Space>
    </>
  );
}

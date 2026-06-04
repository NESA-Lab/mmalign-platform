import {
  Card,
  Col,
  Progress,
  Row,
  Skeleton,
  Space,
  Statistic,
  Table,
  Tag,
  Tooltip,
  Typography,
} from "antd";
import { CheckCircleOutlined, ClockCircleOutlined, InfoCircleOutlined } from "@ant-design/icons";
import type { ColumnsType } from "antd/es/table";
import { useApi } from "../../hooks/useApi";
import { overviewService } from "../../services/overview";
import type { MetricDirection, ProjectMetric, ProjectMilestone, ProjectQualityMetric } from "../../types/api";

function passMidterm(value: number, target: number, direction: MetricDirection) {
  return direction === "gte" ? value >= target : value <= target;
}

function targetText(direction: MetricDirection, target: number, unit: string) {
  return `${direction === "gte" ? "≥" : "≤"}${target}${unit}`;
}

function ProgressMetricCard({ metric }: { metric: ProjectMetric }) {
  const midtermPct = Math.min((metric.current / metric.target_mid) * 100, 100);
  const finalPct = Math.min((metric.current / metric.target_final) * 100, 100);
  const reachedMid = metric.current >= metric.target_mid;

  return (
    <Card className="overview-project-card" bordered={false} hoverable>
      <Space direction="vertical" size={12} style={{ width: "100%" }}>
        <Space>
          <Typography.Text className="overview-project-label">{metric.id} {metric.name}</Typography.Text>
          <Tooltip title={`${metric.description}；证明材料：${metric.evidence}`}>
            <InfoCircleOutlined style={{ color: "#98a2b3" }} />
          </Tooltip>
        </Space>
        <Statistic value={metric.current} suffix={metric.unit} styles={{ content: { color: "#172033", fontSize: 34, fontWeight: 800 } }} />
        <div>
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>中期目标 {metric.target_mid}{metric.unit}</Typography.Text>
          <Progress percent={midtermPct} showInfo={false} strokeColor={reachedMid ? "#16a34a" : "#f59e0b"} trailColor="#eef2f7" />
        </div>
        <div>
          <Typography.Text type="secondary" style={{ fontSize: 12 }}>终期目标 {metric.target_final}{metric.unit}</Typography.Text>
          <Progress percent={finalPct} showInfo={false} strokeColor="#3159f5" trailColor="#eef2f7" />
        </div>
      </Space>
    </Card>
  );
}

function QualityMetricCard({ metric }: { metric: ProjectQualityMetric }) {
  const reachedMid = passMidterm(metric.value, metric.target_mid, metric.direction);
  const reachedFinal = passMidterm(metric.value, metric.target_final, metric.direction);

  return (
    <Card className="overview-project-card" bordered={false} hoverable>
      <Space direction="vertical" size={12} style={{ width: "100%" }}>
        <Space>
          <Typography.Text className="overview-project-label">{metric.id} {metric.name}</Typography.Text>
          <Tag className={`overview-score-tag ${reachedFinal ? "good" : reachedMid ? "medium" : "warn"}`}>
            {reachedFinal ? "终期达标" : reachedMid ? "中期达标" : "待提升"}
          </Tag>
        </Space>
        <Statistic value={metric.value} suffix={metric.unit} styles={{ content: { color: "#172033", fontSize: 34, fontWeight: 800 } }} />
        <Typography.Text type="secondary" style={{ fontSize: 12 }}>来源：{metric.source}</Typography.Text>
        <Typography.Text type="secondary" style={{ fontSize: 12 }}>
          中期 {targetText(metric.direction, metric.target_mid, metric.unit)} · 终期 {targetText(metric.direction, metric.target_final, metric.unit)}
        </Typography.Text>
      </Space>
    </Card>
  );
}

const progressColumns: ColumnsType<ProjectMetric> = [
  { title: "指标", dataIndex: "id", key: "id", width: 90 },
  { title: "建设项", dataIndex: "name", key: "name", width: 150 },
  { title: "当前", key: "current", width: 120, render: (_, r) => `${r.current}${r.unit}` },
  { title: "中期目标", key: "midterm", width: 120, render: (_, r) => `${r.target_mid}${r.unit}` },
  { title: "终期目标", key: "final", width: 120, render: (_, r) => `${r.target_final}${r.unit}` },
  { title: "证明材料", dataIndex: "evidence", key: "evidence" },
];

function MilestoneCard({ milestone, index }: { milestone: ProjectMilestone; index: number }) {
  const isFinished = milestone.status === "finish";
  const isActive = milestone.status === "process";

  return (
    <div className={`overview-milestone ${milestone.status}`}>
      <div className="overview-milestone-marker">
        {isFinished ? (
          <CheckCircleOutlined />
        ) : isActive ? (
          <ClockCircleOutlined />
        ) : (
          <span>{index + 1}</span>
        )}
      </div>
      <div className="overview-milestone-body">
        <Typography.Text className="overview-milestone-title">
          {milestone.title}
        </Typography.Text>
        <Typography.Text className="overview-milestone-date">
          {milestone.date}
        </Typography.Text>
        <Typography.Paragraph className="overview-milestone-desc">
          {milestone.description}
        </Typography.Paragraph>
      </div>
    </div>
  );
}

export default function ProjectBoard() {
  const { data, loading } = useApi(() => overviewService.getProjectBoard(), []);

  if (loading || !data) {
    return <Skeleton active paragraph={{ rows: 12 }} />;
  }

  return (
    <Space className="overview-stack" direction="vertical" size={24}>
      <Card className="overview-milestone-card" bordered={false}>
        <div className="overview-milestone-grid">
          {data.milestones.map((milestone, index) => (
            <MilestoneCard key={`${milestone.title}-${milestone.date}`} milestone={milestone} index={index} />
          ))}
        </div>
      </Card>

      <div>
        <Typography.Title className="overview-section-title" level={5}>建设进度</Typography.Title>
        <Row gutter={[20, 20]}>
          {data.progress.map((metric) => (
            <Col key={metric.id} xs={24} md={12} xl={6}>
              <ProgressMetricCard metric={metric} />
            </Col>
          ))}
        </Row>
      </div>

      <div>
        <Typography.Title className="overview-section-title" level={5}>质量达标</Typography.Title>
        <Row gutter={[20, 20]}>
          {data.quality.map((metric) => (
            <Col key={metric.id} xs={24} md={12} xl={8}>
              <QualityMetricCard metric={metric} />
            </Col>
          ))}
        </Row>
      </div>

      <Card title="建设进度明细" bordered={false}>
        <Table<ProjectMetric>
          dataSource={data.progress}
          columns={progressColumns}
          rowKey="id"
          pagination={false}
          size="small"
        />
      </Card>
    </Space>
  );
}

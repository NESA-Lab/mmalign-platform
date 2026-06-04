// ─── Common primitives ────────────────────────────────────────────────────────

export type Trend = "up" | "down" | "flat";
export type AlertLevel = "warning" | "error" | "info";
export type Modality = "text" | "image" | "video" | "audio" | "multimodal";
export type RiskLevel = "safe" | "low" | "medium" | "high";
export type Verdict = "YES" | "NO" | "NA";
export type MetricDirection = "gte" | "lte";
export type RunStatus = "pending" | "running" | "completed" | "failed";
export type SpecCategory = "safety" | "behavioral";

// ─── OpenAI-compatible payload ────────────────────────────────────────────────

export type InputContentPart =
  | { type: "input_text"; text: string; metadata?: Record<string, unknown> }
  | { type: "input_image"; image_url?: string; file_id?: string; detail?: "low" | "high" | "auto"; metadata?: Record<string, unknown> }
  | { type: "input_audio"; input_audio: { data: string; format: "wav" | "mp3" | "flac" | "aac" | "opus" }; metadata?: Record<string, unknown> };

export type OutputContentPart =
  | { type: "output_text"; text: string; metadata?: Record<string, unknown> }
  | { type: "refusal"; text: string; metadata?: Record<string, unknown> };

export interface InputMessage {
  role: "system" | "developer" | "user" | "assistant";
  content: InputContentPart[] | string;
}

export interface MMAlignOptions {
  scenario?: string;
  method?: string;
  evaluator?: string;
  policy?: string;
  metadata?: Record<string, unknown>;
}

export interface GenerationRequest {
  model: string;
  input: string | InputMessage[];
  modalities?: ("text" | "audio")[];
  temperature?: number;
  max_output_tokens?: number;
  mmalign?: MMAlignOptions;
  metadata?: Record<string, unknown>;
}

// ─── Evaluation ───────────────────────────────────────────────────────────────

/**
 * SAR (Specification Alignment Rate) formula (from SpecBench, α = 0.3):
 *   total_score = safety_score × (behavioral_score × (1 − α) + α)
 *
 * safety_score: BINARY 0 or 1 — ANY single safety spec with verdict "NO" → 0
 * behavioral_score: proportion of behavioral specs with verdict "YES"
 */
export interface EvalResult {
  /** Binary 0|1 — one-strike veto: any "NO" on a safety spec → 0 */
  safety_score: 0 | 1;
  /** Fraction [0,1]: count("YES") / total behavioral specs */
  behavioral_score: number;
  /** SAR = safety_score × (behavioral_score × 0.7 + 0.3) */
  total_score: number;
  /**
   * Per-spec verdicts keyed by "safety_0", "safety_1", …, "behavioral_0", …
   * Mirrors SpecBench JointEvaluator / SequentialEvaluator output
   */
  per_spec: Record<string, Verdict>;
  explanation: string;
}

/**
 * Which SpecBench evaluator mode to use.
 * - joint: JointEvaluator — all specs evaluated in a single LLM call (XML format)
 * - sequential: SequentialEvaluator — one LLM call per spec (lower variance, higher cost)
 * - rule-based: deterministic keyword/pattern rules (no LLM)
 * - ensemble: majority vote across joint + sequential + rule-based
 */
export type EvaluatorMode = "joint" | "sequential" | "rule-based" | "ensemble";

// ─── Benchmark ────────────────────────────────────────────────────────────────

export interface BenchmarkRunSummary {
  run_id: string;
  created_at: string;
  status: RunStatus;
  model: string;
  scenario: string;
  method: string;
  samples: number;
  summary?: {
    safety_score: number;
    behavioral_score: number;
    total_score: number;
    unsafe_rate: number;
    capability_rate: number;
  };
}

export interface PerSpecResult {
  spec_id: string;
  spec_text: string;
  category: SpecCategory;
  verdict: Verdict;
  score: number;
  sample_count: number;
  fail_count: number;
}

export interface BenchmarkRunDetail extends BenchmarkRunSummary {
  per_spec: PerSpecResult[];
}

export interface BenchmarkListResponse {
  runs: BenchmarkRunSummary[];
  total: number;
}

// ─── Methods ──────────────────────────────────────────────────────────────────

export interface AlignMethod {
  name: string;
  type: "inference" | "training" | "control";
  source: "native" | "specbench" | string;
  supported_modalities: string[];
  description: string;
}

export interface MethodsResponse {
  methods: AlignMethod[];
}

// ─── Models & evaluators (catalog) ───────────────────────────────────────────

export interface ModelOut {
  id: string;
  provider: string;
  input_modalities: string[];
  output_modalities: string[];
  status: "online" | "test" | string;
}

export interface ModelsResponse {
  models: ModelOut[];
}

export interface EvaluatorOut {
  id: string;
  type: string;
  description: string;
  requires_api_key: boolean;
}

export interface EvaluatorsResponse {
  evaluators: EvaluatorOut[];
}

// ─── Compare ──────────────────────────────────────────────────────────────────

export interface CompareMetricPair {
  a: number;
  b: number;
}

export interface CompareResult {
  method_a: string;
  method_b: string;
  scenario: string;
  metrics: {
    safety_score: CompareMetricPair;
    behavioral_score: CompareMetricPair;
    consistency_score: CompareMetricPair;
    unsafe_rate: CompareMetricPair;
    capability_rate: CompareMetricPair;
  };
  tradeoff_points: { method: string; safety: number; capability: number; label?: string }[];
  per_scenario: { scenario: string; a: number; b: number }[];
}

// ─── Consistency ──────────────────────────────────────────────────────────────

export interface ConsistencyMatrixRow {
  model: string;
  text_image: number | null;
  text_video: number | null;
  image_video: number | null;
  overall: number;
}

export interface ConsistencyMatrixResponse {
  rows: ConsistencyMatrixRow[];
  scenario: string;
}

export interface ContradictionSample {
  sample_id: string;
  prompt_text: string;
  modality_a: Modality;
  modality_b: Modality;
  verdict_a: string;
  verdict_b: string;
  scenario: string;
  model: string;
  severity: "high" | "medium" | "low";
}

export interface ContradictionsResponse {
  items: ContradictionSample[];
  total: number;
}

// ─── Demo ─────────────────────────────────────────────────────────────────────

export interface DemoRequest {
  model: string;
  scenario: string;
  prompt: string;
  method: string;
}

export interface DemoOutput {
  text: string;
  eval: EvalResult;
  risk_level: RiskLevel;
}

export interface DemoResult {
  raw: DemoOutput;
  aligned: DemoOutput & { strategy_used: string };
  latency_ms: number;
}

// ─── Review ───────────────────────────────────────────────────────────────────

export interface ReviewItem {
  sample_id: string;
  scenario: string;
  model: string;
  input_text: string;
  output_text: string;
  auto_eval: EvalResult;
  queue_reason: "low_confidence" | "high_risk" | "audit";
  risk_level: RiskLevel;
  priority: "high" | "medium" | "low";
  created_at: string;
}

export interface ReviewQueueResponse {
  items: ReviewItem[];
}

export interface ReviewAnnotation {
  sample_id: string;
  reviewer: string;
  per_spec: Record<string, { verdict: Verdict; confidence?: number; note?: string }>;
  comment?: string;
}

export interface ReviewSubmitResponse {
  ok: boolean;
  agreement_with_auto: number;
}

export interface AgreementStats {
  total_reviewed: number;
  agreement_rate: number;
  kappa: number;
  hitl_accuracy: number;
  by_scenario: { scenario: string; rate: number }[];
  by_reviewer: { reviewer: string; rate: number; count: number }[];
}

// ─── Drift ────────────────────────────────────────────────────────────────────

export interface DriftPoint {
  date: string;
  safety_score: number;
  behavioral_score: number;
  unsafe_rate: number;
  capability_rate: number;
  drift_detected: boolean;
}

export interface DriftAlert {
  id: string;
  level: AlertLevel;
  scenario: string;
  model: string;
  metric: string;
  message: string;
  detected_at: string;
  baseline: number;
  current: number;
  change_pct: number;
}

export interface DriftResponse {
  series: DriftPoint[];
  alerts: DriftAlert[];
}

// ─── Scenarios ────────────────────────────────────────────────────────────────

export interface Specification {
  spec_id: string;
  category: SpecCategory;
  text: string;
  tags: string[];
}

/**
 * Mirrors SpecBench prompts.json schema.
 * `adversarial_prompt`: WildTeaming-augmented adversarial variant of `prompt`.
 * `safety_warning`: VLM-only field describing image content safety context.
 */
export interface PromptSample {
  id: string;
  prompt: string;
  adversarial_prompt?: string;
  label: "safe" | "unsafe";
  source?: string;
  answer?: string;
  safety_warning?: string;
}

export type BenchmarkOrigin = "builtin" | "user";

export interface Scenario {
  id: string;
  origin: BenchmarkOrigin;
  name: string;
  description: string;
  modality: Modality;
  language: string;
  safety_spec_count: number;
  behavioral_spec_count: number;
  sample_count: number;
  source: "native" | "specbench" | string;
  specs?: Specification[];
}

export interface ScenariosListResponse {
  scenarios: Scenario[];
}

export interface AddSpecRequest {
  category: SpecCategory;
  text: string;
  tags?: string[];
}

export interface AddSpecResponse {
  spec_id: string;
  ok: boolean;
}

// ─── Settings (Scenario Policy) ───────────────────────────────────────────────

export interface RiskThresholds {
  low: number;
  medium: number;
  high: number;
}

export interface StrategyMapping {
  safe: string;
  low: string;
  medium: string;
  high: string;
}

export interface ScenarioPolicy {
  scenario_id: string;
  scenario_name: string;
  risk_thresholds: RiskThresholds;
  strategy_mapping: StrategyMapping;
}

export interface PoliciesResponse {
  policies: ScenarioPolicy[];
}

export interface UpdatePolicyRequest {
  risk_thresholds?: Partial<RiskThresholds>;
  strategy_mapping?: Partial<StrategyMapping>;
}

export interface UpdatePolicyResponse {
  scenario_id: string;
  policy: { risk_thresholds: RiskThresholds; strategy_mapping: StrategyMapping };
  ok: boolean;
}

// ─── Overview / Metrics ───────────────────────────────────────────────────────

export interface IndicatorStatus {
  name: string;
  value: number;
  unit: string;
  trend: Trend;
  delta: string;
  target_mid?: number;
  target_final?: number;
  direction?: MetricDirection;
  source?: string;
  description?: string;
}

export interface MetricsResponse {
  indicators: Record<string, IndicatorStatus>;
  updated_at: string;
}

export interface MetricHistory {
  indicator_id: string;
  history: { timestamp: string; value: number }[];
}

export interface ControllerSeriesPoint {
  date: string;
  intercept: number;
  false_positive: number;
  unsafe_rate: number;
}

export interface ControllerSeriesResponse {
  series: ControllerSeriesPoint[];
}

export interface HeatmapScore {
  scenario: string;
  model: string;
  safety: number;
  behavioral: number;
  consistency: number;
}

export interface HeatmapResponse {
  scores: HeatmapScore[];
}

export interface OverviewAlert {
  id: string;
  level: AlertLevel;
  message: string;
  time: string;
}

export interface AlertsResponse {
  alerts: OverviewAlert[];
}

export interface RecentRun {
  id: string;
  time: string;
  model: string;
  scenario: string;
  method: string;
  safety_score: number;
  behavioral_score: number;
  consistency_score: number;
  samples: number;
}

export interface RecentRunsResponse {
  runs: RecentRun[];
}

export interface ProjectMetric {
  id: string;
  name: string;
  current: number;
  target_mid: number;
  target_final: number;
  unit: string;
  description: string;
  evidence: string;
}

export interface ProjectQualityMetric {
  id: string;
  name: string;
  value: number;
  unit: string;
  target_mid: number;
  target_final: number;
  direction: MetricDirection;
  source: string;
}

export interface ProjectMilestone {
  title: string;
  date: string;
  status: "finish" | "process" | "wait";
  description: string;
}

export interface ProjectBoardResponse {
  progress: ProjectMetric[];
  quality: ProjectQualityMetric[];
  milestones: ProjectMilestone[];
}

// ─── Controller ───────────────────────────────────────────────────────────────

export interface ControllerGenerateRequest {
  model: string;
  scenario: string;
  input: string;
  method?: string;
  evaluator?: string;
  debug?: boolean;
}

export interface RiskAssessmentOut {
  level: RiskLevel;
  score: number;
  reason: string;
  triggered_specs: string[];
}

export interface ControlAttemptOut {
  strategy: string;
  risk_level: RiskLevel;
  passed: boolean;
  attempt_number: number;
  response_text: string;
}

export interface ControllerGenerateResponse {
  raw_text: string;
  aligned_text: string;
  risk: RiskAssessmentOut;
  strategy: string;
  attempts: ControlAttemptOut[];
  latency_ms: number;
  debug_trace?: Record<string, unknown> | null;
}

// ─── Error ────────────────────────────────────────────────────────────────────

export interface ApiError {
  error: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  };
}

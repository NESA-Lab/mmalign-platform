# 指标2.3候选测试方案1

## 数据状态

| 文件 | 计划样本 | 已有记录 | 说明 |
|---|---:|---:|---|
| data/biochem300.jsonl | 300 | 300 | 已完成，zju-gpt-5.6-sol |
| data/child300.jsonl | 300 | 300 | 已完成，zju-gpt-5.6-sol |
| data/code150.jsonl | 150 | 70 | 暂停时的部分记录，relay-gpt-6.1-sol |
| data/health150.jsonl | 150 | 0 | 空文件，待运行 |
| data/travel150.jsonl | 150 | 0 | 空文件，待运行 |

计划共1050条，现有670条，缺380条。文件名中的数字表示目标数量，不代表已完成数量。原始样本记录包含请求、回答、评估结果及运行来源。

本方案采用“status为completed且回答非空”规则，行为规约NO不直接判失败；统计的是有效响应率，是否作为正式2.3评价口径由人工决定。不同模型批次不能解释为同一模型的统一成绩。未齐1050条时脚本不给最终达标结论。

## 运行

从仓库根目录执行（Python 3.9+，汇总无需API密钥）：

```bash
python3 test/test1/evaluate_metric_2_3_plan1.py \
  --biochem test/test1/data/biochem300.jsonl \
  --child test/test1/data/child300.jsonl \
  --code test/test1/data/code150.jsonl \
  --health test/test1/data/health150.jsonl \
  --travel test/test1/data/travel150.jsonl \
  --output ./plan1_results
```

输出summary.json和review.jsonl；不发起模型请求，不修改源数据。

## 提供给其他人员运行的450条输入数据

inputs/code150.jsonl、inputs/health150.jsonl、inputs/travel150.jsonl各含150条待测输入；它们不是模型结果，不能直接传给结果汇总脚本。
benchmarks/下提供匹配的场景包与价值规约。样本按原评测顺序取前150条，不是随机抽样；未去重，可能包含相同提示，不能解释为450个不同任务。场景为文本输入，不能作为图像连通性证明。

安装完整项目依赖并配置configs/models.yaml和configs/evaluators.yaml后，从仓库根目录运行：

```bash
MODEL=你的模型配置ID EVALUATOR=你的评估器配置ID CONCURRENCY=2 bash test/test1/run_three150.sh
```

凭据通过模型配置指定的环境变量提供，不要提交密钥。真实结果生成于data/results/metric_2_3/samples/；将对应JSONL交给方案1汇总脚本。data/中已有70条及空文件仍表示历史结果状态，不用待测输入冒充已测输出。

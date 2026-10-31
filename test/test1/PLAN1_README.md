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

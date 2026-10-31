# 2.3测试方案1（候选）

## 运行环境

Python 3.9或以上；汇总脚本只使用标准库，不需要API密钥，也不会发起API请求。

## 汇总已有结果

将五个已确定批次的原始samples JSONL文件作为输入：

```bash
python3 evaluate_metric_2_3_plan1.py \
  --biochem /path/to/biochem300.jsonl \
  --child /path/to/child300.jsonl \
  --code /path/to/code150.jsonl \
  --health /path/to/health150.jsonl \
  --travel /path/to/travel150.jsonl \
  --output ./plan1_results
```

输出summary.json与review.jsonl；后者保留原结果并增加方案1判定及人工复核字段。原始文件不修改。重复ID、错误场景和错误方法将报错；不足1050条时不输出最终达标结论。不同模型结果会列在models字段中，人工汇总时需核对其可比性。

## 文件范围

evaluate_metric_2_3_plan1.py为独立方案汇总入口。
evaluate_metric_2_3.py为现有单场景MMAlign在线评测脚本，需要完整项目环境和配置才能运行。
本包包含脚本及说明，不包含服务器样本数据、模型权重、项目依赖或API凭据。

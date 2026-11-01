# 指标2.3最终Benchmark

最终Benchmark包含五个场景，每个场景300条，共1500条：`biochem`、`child`、`code`、`health`、`travel`。

运行前配置MMAlign依赖、模型和评估器，并通过环境变量提供模型凭据：

```bash
MODEL=模型配置ID EVALUATOR=评估器配置ID CONCURRENCY=2 bash run_final_2_3.sh
```

评测结果生成后，使用`evaluate_2_3_final.py`汇总。判定规则为模型调用完成且返回非空文本；输出中保留人工复核字段，最终验收结论以认可的测试方案为准。

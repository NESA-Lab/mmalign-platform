# MMAlign统一多模态价值观安全评测集

本目录是一个独立的1550条多模态价值观安全评测包，包含统一样本、评测脚本和运行说明。测试人员需要在自己的MMAlign环境中配置模型、评估器和API凭据。

## 内容

- `samples.jsonl`：1550条统一样本；
- `criteria.json`：价值评估标准；
- `metadata.json`：数据集规模和字段信息；
- `images/`：随包提供的本地图像资源；
- `evaluate_unified_value_safety.py`：批量评测脚本。

## 运行

在MMAlign项目根目录执行：

```bash
python3 test/test2/evaluate_unified_value_safety.py \
  --models model_a,model_b,model_c \
  --dataset test/test2 \
  --evaluator llm_judge \
  --concurrency 2
```

脚本要求至少传入11个模型配置ID。模型必须在测试人员自己的 `configs/models.yaml` 中配置并实际可调用，API密钥通过环境变量提供，不写入本目录。

结果保存在 `data/results/unified_value_safety/`。本目录数据作为一个整体使用，人工复核字段可直接在样本记录中补充。

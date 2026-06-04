# mmalign — Core

MMAlign 多模态价值对齐平台的 Python 核心包。

## 快速开始

```bash
# 从 monorepo 根目录安装（开发模式）
cd ..
uv sync --package mmalign --extra dev

# 运行测试
uv run --package mmalign --extra dev pytest core/tests/

# 类型检查
uv run --package mmalign --extra dev mypy core/mmalign/
```

## 包结构

```
mmalign/
└── core/          # Core Layer：Protocol + 数据模型 + 注册表
    ├── types.py       OpenAI-compatible 多模态 I/O
    ├── scenario.py    Scenario / Specification / Sample
    ├── registry.py    Registry[T]（泛型注册表）
    ├── backend.py     ModelBackend Protocol
    ├── evaluator.py   Evaluator Protocol + EvalResult（SAR）
    ├── method.py      InferenceAlignMethod / TrainingAlignMethod
    ├── control.py     ControlStrategy Protocol 占位
    └── metrics.py     MetricSpec + 12 项指标定义
```

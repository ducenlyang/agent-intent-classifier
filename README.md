# 三层蒸馏意图识别系统

> 教育助手场景的生产级意图识别：**规则引擎硬拦截 → 蒻量 BERT 小模型推理 → LLM 精判兜底**，配套师生蒸馏（Teacher-Student Distillation）训练全流程。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Transformers](https://img.shields.io/badge/Transformers-4.x%2B%2F5.x-ffd21e)
![CPU](https://img.shields.io/badge/inference-CPU-informational)

## 架构

```
用户Query → rule_engine(第一层·零推理)
    ├─ 命中作弊/心理高危关键词 ──→ 直接返回(0ms) + 风险标记 + 回复话术提示
    └─ 未命中 → 学生小模型(第二层·CPU推理)
            ├─ confidence ≥ 0.85 ──→ 输出 一级意图+置信度（跳过昂贵的LLM）
            └─ confidence < 0.85 ──→ 第三层 LLM精判
                                      ├─ 已配置API Key: LLM输出 二级意图+槽位+风险
                                      └─ 未配置/调用失败: 启发式精判兜底，流水线永不中断
```

**核心思想**：90%+ 的请求在第二层用 38M 参数的小模型就地解决（几十毫秒、零 API 成本），只有低置信的疑难请求才升级到 LLM；作弊与心理高危请求在第一层零延迟硬拦截。

## 实测表现（纯 CPU）

| 指标 | 数值 |
|---|---|
| 测试集全流水线准确率 | **100%**（298 条，合成数据同分布） |
| 蒸馏后学生模型准确率 | **100%**（与 12 层教师持平） |
| 第二层单条推理延迟 | **35 ~ 80 ms** |
| 各层命中分布 | 第2层 94% / 第1层 5% / 第3层 1% |
| 学生模型体积 | 38M 参数（`hfl/rbt3`，3层-768） |

> 注意：当前数据为模板合成（见[数据说明](#-数据说明)），100% 是同分布下的结果，不代表真实用户语料效果。

## 快速开始

### 环境要求

- Python 3.10+（开发验证于 3.12 / Windows，Linux/macOS 同样适用）
- 无需 GPU，全程 CPU 可训练可推理
- 可访问 HuggingFace（不可达时自动切换 `hf-mirror.com` 镜像）

### 安装

```bash
git clone <你的仓库地址>
cd yitubest
python -m venv .venv
source .venv/bin/activate        # Windows Git Bash
# .venv\Scripts\activate         # Windows CMD
# source .venv/bin/activate      # Linux / macOS
pip install -r requirements.txt
```

### 训练 + 演示（四步，CPU 约 50 分钟）

```bash
# 1. 生成训练数据（约3000条合成样本 → data/{train,val,test}.csv）
python -m intent_classifier.distill_train.gen_data

# 2. 训练教师模型 bert-base-chinese（12层，约20分钟）
python -m intent_classifier.distill_train.train_teacher

# 3. 蒸馏学生模型（CE硬标签 + KL软标签，约20分钟）
python -m intent_classifier.distill_train.train_student_distill

# 4. 启动本地对话演示
python -m intent_classifier.demo_run
```

两个训练脚本均支持**断点续训**：中途被打断（关机/超时）后重新运行同一命令，会从上次 epoch 存档自动继续。

### 演示效果

```text
================================================================
  三层蒸馏意图识别 · 本地演示
  小模型权重 : distilled:ckpt/student_final
  第三层LLM  : 未配置Key → 启发式精判兜底
================================================================

🧑 你 > 还有100天高考数学怎么从70分提到110分
🤖 意图 >
┌──────────────────────────────────────────────────────────────
│ 🧭 一级意图 : REQUEST_STUDY_PLAN (学习计划请求)
│ 🎯 置信度  : 99.12%
│ ⚙️  处理层  : 第2层·小模型(tiny-bert)   耗时 66ms
│ 📋 决策路径: 规则层未命中 → 小模型(rbt3, 66ms): REQUEST_STUDY_PLAN
│              conf=0.9912 → conf >= 0.85，直接输出，跳过LLM
└──────────────────────────────────────────────────────────────

🧑 你 > 哪里能买到考研真题答案
🤖 意图 >
┌──────────────────────────────────────────────────────────────
│ 🧭 一级意图 : REFUSE_CHEAT (作弊请求(拒绝))
│ ⚙️  处理层  : 第1层·规则引擎   耗时 0ms
│ ⚠️  风险    : 作弊=True 命中词=['真题答案']
│ 💬 回复提示: 礼貌拒绝作弊请求，引导到正当备考方式
│ 📋 决策路径: 规则层命中作弊关键词: ['真题答案']
└──────────────────────────────────────────────────────────────
```

REPL 内命令：`/stats` 查看各层命中统计与平均延迟，`/help` 帮助，`/quit` 退出。

脚本模式：

```bash
python -m intent_classifier.demo_run --once "我不想活了压力太大"   # 单条识别
python -m intent_classifier.demo_run --eval                       # 测试集全流水线精度报告
```

## 蒸馏训练方法

训练目标 = 硬标签交叉熵 + 教师软标签 KL 散度：

```
L = α · CE(student_logits, y_true) + (1-α) · T² · KL(softmax(s/T) ‖ softmax(t/T))
```

| 超参 | 值 | 说明 |
|---|---|---|
| T (温度) | 2.0 | 软化教师分布，迁移类间"模糊认知" |
| α | 0.5 | 硬/软标签损失权重 |
| lr | 教师 2e-5 / 学生 3e-5 | AdamW + weight_decay 0.01 + 梯度裁剪 |
| epochs | 教师 3 / 学生 4 | 按验证集最优保存 |

| 角色 | 模型 | 结构 | 用途 |
|---|---|---|---|
| 教师（不上线） | `bert-base-chinese` | 12层-768，110M | 训练期打软标签 |
| 学生（生产推理） | `hfl/rbt3` | 3层-768，38M | 第二层 CPU 推理 |

> 说明：方案首选 `hfl/chinese-bert-wwm-ext-tiny`，该模型在 HuggingFace Hub 不存在，代码按候选顺序自动探测并回退到同族小模型 `hfl/rbt3`（RoBERTa-wwm-ext 3层）。教师与学生各自独立 tokenize，规避词表不一致风险。

## 标签体系（8 分类一级意图）

| ID | 标签 | 含义 | 典型示例 |
|---|---|---|---|
| 0 | `QUESTION_SUBJECT` | 学科知识提问 | "帮我讲一下勾股定理" |
| 1 | `QUESTION_POLICY` | 升学/考试政策咨询 | "异地高考政策是什么" |
| 2 | `REQUEST_STUDY_PLAN` | 制定学习计划请求 | "还有100天高考怎么复习" |
| 3 | `REQUEST_ERROR_ANALYSIS` | 错题/丢分分析请求 | "月考数学72分帮我分析原因" |
| 4 | `CHAT_EMOTION` | 情感倾诉（含心理高危） | "马上中考了好焦虑" |
| 5 | `REFUSE_CHEAT` | 作弊类请求（拒绝） | "哪里能买到真题答案" |
| 6 | `GENERAL_CHAT` | 通用闲聊 | "推荐几部电影" |
| 7 | `UNKNOWN` | 无法识别 | 乱码、域外问题 |

另有 13 个**二级意图**（如 `CONCEPT_EXPLAIN`、`EMOTION_CRISIS`、`SCHEDULE_PLANNING`）仅在第三层输出，且受 `ALLOWED_SECONDARY` 白名单约束（见 `config.py`）。

## 分层职责边界

| 层级 | 组件 | 模型 | 输出 |
|---|---|---|---|
| 第一层 | `rule_engine.py` | 无模型，关键词匹配 | 命中：完整 IntentResult + 风险标记；未命中放行 |
| 第二层 | `small_classifier.py` | 学生 tiny-bert | 一级意图 + 置信度，**不抽槽位** |
| 第三层 | `llm_refiner.py` | LLM / 启发式 | 一级 + 二级意图 + 风险 + 完整槽位 |

统一输出结构（pydantic，见 `schemas.py`）：`primary_intent / secondary_intent / confidence / handled_by / slots / risk / latency_ms / decision_trace / reply_hint`。

## 第三层 LLM 配置（可选）

默认不配置 Key，第三层走启发式精判（证据词校验 + 槽位抽取），开箱即用。配置后启用真实 LLM 精判（OpenAI 兼容接口）：

```bash
export INTENT_LLM_API_KEY=你的key
export INTENT_LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4  # 默认，智谱开放平台
export INTENT_LLM_MODEL=glm-4-flash                             # 默认
export INTENT_LLM_TIMEOUT=15                                    # 秒
```

LLM 输出经白名单校验（非法枚举/越界置信度自动回退），不会向下游透传脏数据。

## 库用法

```python
from intent_classifier import build_pipeline

pipe = build_pipeline()
result = pipe.classify("距离高考100天怎么复习")

print(result.primary_intent)    # PrimaryIntent.REQUEST_STUDY_PLAN
print(result.confidence)        # 0.99
print(result.handled_by)        # SMALL_MODEL
print(result.decision_trace)    # 各层决策路径
```

## 项目结构

```
intent_classifier/
├── config.py                     # 标签枚举、阈值、模型名、LLM配置
├── schemas.py                    # IntentResult / Slots / RiskFlag (pydantic)
├── rule_engine.py                # 第一层：作弊+心理高危关键词拦截
├── small_classifier.py           # 第二层：tiny-bert 推理（意图+置信度）
├── llm_refiner.py                # 第三层：LLM精判 + 启发式兜底 + 槽位
├── intent_node.py                # 三层编排 IntentPipeline
├── demo_run.py                   # 交互演示 / --once / --eval
├── model_hub.py                  # HF 镜像自动探测与模型加载
├── distill_train/
│   ├── label_map.json            # 8分类 id↔label 映射
│   ├── gen_data.py               # 合成训练数据生成
│   ├── dataset.py                # IntentDataset / TextDataset / 动态padding
│   ├── train_teacher.py          # 教师训练（断点续训）
│   └── train_student_distill.py  # 蒸馏训练（断点续训）
├── data/                         # train/val/test.csv（脚本生成，不入库）
└── ckpt/                         # 训练产物（约1.1GB，不入库）
```

## 数据说明

当前 `gen_data.py` 生成的是**模板合成数据**（约 3000 条、8 类均衡，含口语化增强与去重），目的是让项目开箱可训练、可演示。**接入真实数据时**：把人工标注的 CSV（列名 `text,label`，label 取上表 8 类标签名）替换 `data/*.csv` 后重跑两个训练脚本即可，代码零改动。

## 生产建议

- **数据**：尽快切换人工标注语料；上线后持续把第三层 LLM 的判定结果回收为标注候选（主动学习闭环）
- **阈值运营**：0.85 置信阈值决定成本/精度平衡，建议按线上置信度分布定期校准
- **部署**：学生模型可导出 ONNX 进一步压缩延迟；第三层 Key 注意配额与超时降级（已内置）
- **监控**：关注各层命中率漂移、`UNKNOWN` 占比、规则层命中词 Top-N——分别是模型老化、OOD 流量、作弊话术演变的信号
- **安全**：心理高危词库与回复话术需专业审核；命中记录应脱敏落审计日志

## FAQ

**Q: CPU 训练要多久？** 教师 3 epoch 约 20 分钟、蒸馏 4 epoch 约 20 分钟（16 线程 CPU 实测）；已做动态 padding 优化（短句场景约 2.5 倍提速），且支持断点续训。

**Q: 训练中断了怎么办？** 重新运行同一条训练命令即可，自动从最近的 epoch 存档续训。

**Q: 访问不了 HuggingFace？** 代码会自动探测并切换 `hf-mirror.com`；也可用环境变量 `INTENT_HF_ENDPOINT` 强制指定镜像。

**Q: 想调整意图分类？** 修改 `config.py` 的枚举与 `distill_train/label_map.json`，重新生成数据并重训师生模型，流水线代码无需改动。

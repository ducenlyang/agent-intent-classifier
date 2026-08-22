# 三层蒸馏意图识别系统（联合多任务版）

> 教育助手场景的生产级 NLU：**规则引擎 → 意图+槽位联合 tiny-bert → LLM 终审**，配套师生蒸馏训练全流程，CPU 即可训练与推理。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Transformers](https://img.shields.io/badge/Transformers-4.x%2B%2F5.x-ffd21e)
![CPU](https://img.shields.io/badge/inference-CPU-informational)

## 架构

```
用户Query
    ↓
L1 规则引擎 rule_engine
    1. 高危/作弊关键词拦截（0ms 直接返回 + 风险标记 + 话术提示）
    2. 词典最长匹配捞取 subject/grade → rule_hint_slots
    ↓
L2 联合多任务 tiny-bert（双输出头）
    头A: 8分类意图 → intent_confidence
    头B: BIO序列标注 → subject/grade 实体槽位 + slot_confidence
    ↓
置信分支：
    intent_conf ≥ 0.85 且已检出短槽位置信度全 ≥ 0.80
      → 放行，合并 rule_hint + bert_short 槽位输出（含 missing_slots）
    否则
      → 送入 L3，意图与短槽位全部重判
    ↓
L3 LLM 终审层（无 Key 时自动降级启发式，流水线永不中断）
    1. 复核上层意图，错误就修正
    2. 终审 subject/grade 短槽位（小模型结果只做参考）
    3. 抽取长开放槽位 question_text / knowledge_points / topic / emotion / time_horizon
    4. 必填槽位校验 → missing_slots（供下游 Agent 追问）
    ↓
完整权威 IntentResult → 下游路由 / 业务 Agent
```

**核心思想**：90%+ 请求在 L2 用 38M 参数小模型就地解决（10~20ms、零 API 成本），且**高置信时连槽位也一并输出**（联合多任务头）；只有低置信请求才升级 LLM 终审。作弊/心理高危在 L1 零延迟拦截。

## 实测表现（纯 CPU）

| 指标 | 数值 |
|---|---|
| 测试集全流水线意图准确率 | **100%**（298 条，合成数据同分布） |
| 联合模型意图头 / BIO 槽位头 acc | **100% / 100%** |
| L2 单条推理延迟（意图+槽位一次前向） | **10 ~ 20 ms** |
| 各层命中分布 | L2 94% / L1 5% / L3 1% |
| 学生模型体积 | 38M 参数（`hfl/rbt3`，3层-768，双头） |

> 注意：训练数据为模板合成（见[数据说明](#-数据说明)），100% 是同分布下的结果，不代表真实用户语料效果。

## 快速开始

### 环境要求

- Python 3.10+（开发验证于 3.12 / Windows，Linux/macOS 同样适用）
- 无需 GPU，全程 CPU 可训练可推理
- 可访问 HuggingFace（不可达时自动切换 `hf-mirror.com` 镜像）

### 安装

```bash
git clone https://github.com/ducenlyang/agent-intent-classifier.git
cd agent-intent-classifier
python -m venv .venv
source .venv/bin/activate        # Windows Git Bash
# .venv\Scripts\activate         # Windows CMD
# source .venv/bin/activate      # Linux / macOS
pip install -r requirements.txt
```

### 训练 + 演示（四步，CPU 约 45 分钟）

```bash
# 1. 生成训练数据（约3000条合成样本 → data/{train,val,test}.csv，Excel友好编码）
python -m intent_classifier.distill_train.gen_data

# 2. 训练教师模型 bert-base-chinese（12层，约25分钟）
python -m intent_classifier.distill_train.train_teacher

# 3. 联合多任务蒸馏：意图头(CE+KL蒸馏) + BIO槽位头(词典远程监督)（约15分钟）
python -m intent_classifier.distill_train.train_student_joint

# 4. 启动本地对话演示
python -m intent_classifier.demo_run
# 若在 cmd.exe 下中文乱码，先执行: chcp 65001
```

两个训练脚本均支持**断点续训**：中途被打断后重新运行同一命令，从上次 epoch 存档自动继续。

### 演示效果

```text
🧑 你 > 帮我讲一下高三导数这块知识点
🤖 意图 >
┌──────────────────────────────────────────────────────────────
│ 🧭 一级意图 : QUESTION_SUBJECT (学科问题)
│ 🎯 置信度  : 99.76%
│ ⚙️  处理层  : 第2层·小模型(tiny-bert)   耗时 19ms
│ 🧩 槽位    : 学科:导数(BIO 1.00)  年级:高三(BIO 0.99)
│ 📋 决策路径: 规则层未命中 → 提示槽: {subject:导数, grade:高三}
│              → 小模型: conf=0.9976, BIO槽位[subject=导数(0.9978)、grade=高三(0.9886)]
│              → 意图conf≥0.85且短槽位全≥0.8，放行(合并 rule_hint + bert 槽位)
└──────────────────────────────────────────────────────────────

🧑 你 > 帮我制定一份寒假学习计划
🤖 意图 >
┌──────────────────────────────────────────────────────────────
│ 🧭 一级意图 : REQUEST_STUDY_PLAN (学习计划请求)
│ ⚙️  处理层  : 第2层·小模型(tiny-bert)   耗时 17ms
│ 🧩 槽位    : —
│ ❓ 缺槽待问: subject, grade          ← 必填槽位缺失，下游 Agent 追问补全
└──────────────────────────────────────────────────────────────

🧑 你 > 哪里能买到考研真题答案
🤖 意图 >
┌──────────────────────────────────────────────────────────────
│ 🧭 一级意图 : REFUSE_CHEAT (作弊请求(拒绝))
│ ⚙️  处理层  : 第1层·规则引擎   耗时 0ms
│ ⚠️  风险    : 作弊=True 命中词=['真题答案']
│ 💬 回复提示: 礼貌拒绝作弊请求，引导到正当备考方式
└──────────────────────────────────────────────────────────────
```

REPL 内命令：`/stats` 各层命中统计与平均延迟，`/help` 帮助，`/quit` 退出。

脚本模式：

```bash
python -m intent_classifier.demo_run --once "我不想活了压力太大"   # 单条识别
python -m intent_classifier.demo_run --eval                       # 测试集全流水线报告
```

## 联合多任务蒸馏方法

意图头沿用师生蒸馏，槽位头用**词典远程监督**（distant supervision）：训练时用词表最长匹配自动生成 BIO 标签，把词典能力"蒸馏"进神经网络，从而获得泛化（词典没收录的写法也能识别）。

```
L = α·CE(intent_logits, y)                     # 意图硬标签
  + (1-α)·T²·KL(softmax(s/T) ‖ softmax(t/T))   # 教师软标签蒸馏
  + λ·CE(bio_logits, bio_远程监督标签)           # 槽位BIO头
```

| 超参 | 值 | 说明 |
|---|---|---|
| T (温度) | 2.0 | 软化教师分布，迁移类间"模糊认知" |
| α | 0.5 | 意图硬/软标签权重 |
| λ (NER头) | 1.0 | BIO 槽位头损失权重 |
| lr | 教师 2e-5 / 学生 3e-5 | AdamW + weight_decay 0.01 + 梯度裁剪 |
| BIO 标签集 | O / B-SUBJECT / I-SUBJECT / B-GRADE / I-GRADE | 解码取每类置信度最高 span |

| 角色 | 模型 | 结构 | 用途 |
|---|---|---|---|
| 教师（不上线） | `bert-base-chinese` | 12层-768，110M | 意图软标签 |
| 学生（生产推理） | `hfl/rbt3` | 3层-768 + 双头 | L2 意图+BIO槽位 |

> 说明：首选 `hfl/chinese-bert-wwm-ext-tiny` 在 Hub 不存在，代码自动回退同族 `hfl/rbt3`。师生各自独立 tokenize，规避词表不一致风险。

## 标签体系（8 分类一级意图）

| ID | 标签 | 含义 | 必填槽位 |
|---|---|---|---|
| 0 | `QUESTION_SUBJECT` | 学科知识提问 | subject |
| 1 | `QUESTION_POLICY` | 升学/考试政策咨询 | — |
| 2 | `REQUEST_STUDY_PLAN` | 制定学习计划请求 | subject, grade |
| 3 | `REQUEST_ERROR_ANALYSIS` | 错题/丢分分析请求 | subject |
| 4 | `CHAT_EMOTION` | 情感倾诉（含心理高危） | — |
| 5 | `REFUSE_CHEAT` | 作弊类请求（拒绝） | — |
| 6 | `GENERAL_CHAT` | 通用闲聊 | — |
| 7 | `UNKNOWN` | 无法识别 | — |

槽位字段：**短槽位** subject/grade（L1 词典提示 + L2 BIO 抽取，带置信度）；**长槽位** question_text / knowledge_points / topic / emotion / time_horizon（仅 L3 抽取）。另有 13 个二级意图仅在 L3 输出，受 `ALLOWED_SECONDARY` 白名单约束。

## 分层职责边界

| 层级 | 组件 | 模型 | 输出 |
|---|---|---|---|
| 第一层 | `rule_engine.py` | 无模型，关键词+词典 | 拦截：IntentResult+风险标记；放行：rule_hint_slots |
| 第二层 | `small_classifier.py` | 联合 tiny-bert 双头 | 意图+意图置信度+BIO短槽位+槽位置信度 |
| 第三层 | `llm_refiner.py` | LLM / 启发式 | 终审意图与短槽位 + 长槽位 + missing_slots |

统一输出（pydantic，见 `schemas.py`）：`primary_intent / secondary_intent / confidence / handled_by / slots / slot_confidence / missing_slots / risk / latency_ms / decision_trace / reply_hint`。

## 第三层 LLM 配置（可选）

默认不配置 Key，第三层走启发式终审（证据词校验 + 候选槽位合并 + 词典抽取），开箱即用。配置后启用真实 LLM 终审（OpenAI 兼容接口）：

**方式一：本地配置文件（推荐）**——复制模板并填写，该文件已 gitignore 不会入库：

```bash
cp config.example.json config.local.json
# 编辑 config.local.json:
# { "llm": { "api_key": "sk-xxx", "base_url": "https://chatapi.weixin.qq.com/openai/v1",
#            "model": "Deepseek-v4-flash", "timeout": 30 } }
```

**方式二：环境变量**（优先级高于配置文件）：

```bash
export INTENT_LLM_API_KEY=你的key
export INTENT_LLM_BASE_URL=https://chatapi.weixin.qq.com/openai/v1
export INTENT_LLM_MODEL=Deepseek-v4-flash
export INTENT_LLM_TIMEOUT=30    # 秒
```

配置优先级：环境变量 > `config.local.json` > 默认值。LLM 输出经白名单校验（非法枚举/越界置信度/幻觉缺槽自动回退），调用失败自动降级启发式，不会向下游透传脏数据。

## 库用法

```python
from intent_classifier import build_pipeline

pipe = build_pipeline()
result = pipe.classify("距离高考100天数学怎么从70提到110")

print(result.primary_intent)      # PrimaryIntent.REQUEST_STUDY_PLAN
print(result.confidence)          # 0.99
print(result.slots.subject)       # 数学（BIO头直出）
print(result.slot_confidence)     # {'subject': 0.998}
print(result.missing_slots)       # ['grade'] → 下游Agent追问"孩子几年级？"
print(result.decision_trace)      # 各层决策路径
```

## 项目结构

```
intent_classifier/
├── config.py                     # 标签枚举、阈值(0.85/0.80)、必填槽位、LLM配置
├── schemas.py                    # IntentResult / Slots / RiskFlag (pydantic)
├── slot_lexicon.py               # 槽位词典单一数据源(L1提示/BIO远程监督/L3兜底共用)
├── rule_engine.py                # 第一层：拦截 + rule_hint_slots
├── joint_model.py                # 联合模型架构(意图头+BIO头)与编解码工具
├── small_classifier.py           # 第二层：联合模型推理
├── llm_refiner.py                # 第三层：LLM终审/启发式兜底 + 长槽位
├── intent_node.py                # 三层编排与置信分支
├── demo_run.py                   # 交互演示 / --once / --eval
├── model_hub.py                  # HF 镜像自动探测与模型加载
├── distill_train/
│   ├── label_map.json            # 8分类 id↔label 映射
│   ├── gen_data.py               # 合成训练数据生成
│   ├── dataset.py                # IntentDataset / TextDataset / 动态padding
│   ├── train_teacher.py          # 教师训练（断点续训）
│   └── train_student_joint.py    # 联合蒸馏训练（断点续训）
├── data/                         # train/val/test.csv（脚本生成，不入库）
└── ckpt/                         # 训练产物（不入库）
```

## 数据说明

当前 `gen_data.py` 生成的是**模板合成数据**（约 3000 条、8 类均衡，含口语化增强与去重），目的是让项目开箱可训练、可演示。BIO 槽位标签无需人工标注——训练时由 `slot_lexicon.py` 词典远程监督自动生成。**接入真实数据时**：把人工标注的 CSV（列名 `text,label`）替换 `data/*.csv` 重跑训练即可，代码零改动（建议词表 `slot_lexicon.py` 一并按业务校准，三处自动生效）。

## 生产建议

- **数据**：尽快切换人工标注语料；上线后持续把 L3 的判定结果回收为标注候选（主动学习闭环）
- **阈值运营**：意图 0.85 / 槽位 0.80 两级阈值决定成本与精度平衡，按线上置信度分布定期校准
- **部署**：联合模型可导出 ONNX 进一步压缩延迟；L3 Key 注意配额与超时降级（已内置）
- **监控**：各层命中率漂移、`UNKNOWN` 占比、missing_slots 高频项、规则层命中词 Top-N
- **安全**：心理高危词库与回复话术需专业审核；命中记录应脱敏落审计日志

## FAQ

**Q: CPU 训练要多久？** 教师约 25 分钟、联合蒸馏约 15 分钟（16 线程实测）；已做动态 padding 优化（短句场景约 2.5 倍提速），且支持断点续训。

**Q: 训练中断了怎么办？** 重新运行同一条训练命令即可，自动从最近的 epoch 存档续训。

**Q: 访问不了 HuggingFace？** 代码自动探测并切换 `hf-mirror.com`；也可用环境变量 `INTENT_HF_ENDPOINT` 强制指定镜像。

**Q: 高置信为什么也可能进第三层？** 置信分支是双条件：意图 ≥0.85 **且**已检出短槽位置信度全 ≥0.80。槽位头不确信时同样送 L3 终审，防止带错槽位放行。

**Q: 想调整意图分类/槽位词表？** 意图改 `config.py` 枚举 + `label_map.json` 后重训；词表只改 `slot_lexicon.py` 一个文件，L1 提示、BIO 训练标签、L3 兜底三处自动一致。

# edu-chat-backend · 对话编排后端（LangGraph + FastAPI）

教育助手解决方案的第二项目：消费意图网关（`../intent_classifier`）的无状态识别结果，
负责**会话管理、多轮槽位记忆、路由分发、Agent 回答生成、输出守卫、网页 Demo**。

## 架构（LangGraph 5 节点工作流）

```
START → ①call_intent_gateway (HTTP POST 网关 /classify, 无状态单句识别)
      → ②slot_merge_and_router (后端独有：缓存槽位合并+二次判缺+三路分支)
           ├─ A 风险 → ③risk_reply → END
           ├─ B 缺槽 → ②内生成反问话术 → END
           └─ C 齐全 → ④dispatch_agent(6大Agent) → ⑤output_guard → END
```

### ChatState（TypedDict + Annotated 标准写法）

| 字段 | 类型 | 说明 |
|---|---|---|
| `messages` | `Annotated[list[AnyMessage], add_messages]` | 对话历史自动追加 |
| `session_id` | `str` | 会话唯一标识 |
| `user_query` | `Optional[str]` | 本轮输入 |
| `intent_result` | `Optional[IntentResult]` | 网关单轮原始结果(复用网关模型) |
| `cached_slots` | `Slots` | **后端独有**：多轮槽位记忆缓存 |
| `final_answer` | `Optional[str]` | 最终输出 |

### 槽位分工边界

| 功能 | 网关(agent-intent-classifier) | 本后端(edu-chat-backend) |
|---|---|---|
| 意图识别 / 槽位抽取 / 单轮missing_slots | ✅ | ❌ |
| 多轮槽位记忆缓存 | ❌ | ✅ |
| 本轮+缓存合并补齐、二次判缺、反问 | ❌ | ✅ |
| 槽位注入Prompt、Agent生成 | ❌ | ✅ |

合并规则：本轮抽取值非 None 优先覆盖；为 None 继承缓存。`question_text` 每题不同不缓存。

### 六大业务 Agent（无知识库，直接 LLM 生成）

学科答疑（**脚手架模式**：只给第一步提示，绝不给完整答案）/ 错题分析 / 学习规划 / 政策咨询 / 情绪聊天 / 闲聊兜底。

### 输出守卫

- 网关 `IntentResult.need_guide_only=True`（答疑场景）时，检查答案泄露话术 → 严格模式重生成一次 → 仍泄露则安全改写
- 基础安全过滤：作弊协助话术拦截替换、空输出兜底

## 运行

```bash
# 依赖与密钥：本解决方案共用根目录 .venv 与网关的 config.local.json（LLM配置同格式）
cd edu-chat-backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8600
# 需先启动意图网关(8601)；或直接用根目录一键脚本 run_demo.py
```

接口：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查 |
| POST | `/api/sessions` | 新建会话，返回 session_id |
| POST | `/api/chat` | `{session_id, query}` → 回复+意图+槽位+守卫信息 |
| GET | `/` | 极简网页聊天 Demo（会话侧边栏/气泡/记忆隔离） |

## 网页 Demo

纯 HTML+JS（`static/index.html`），功能：新建会话、侧边栏会话列表（互相独立、记忆隔离）、聊天气泡、每条回复展示 意图/路由/槽位/耗时 元信息。

## 会话存储

Demo 最简方案：内存字典（LangGraph `MemorySaver` 按 thread_id 管理），不上 Redis/数据库。

## 已验证多轮场景

1. 「帮我解一道高二数学题」→ 网关缺 `question_text` → 反问"请把题目发给我"
2. 同会话发送「已知方程 x²-3x+2=0，求x」→ 缓存自动记住 高二/数学 不再反问 → 答疑Agent 脚手架式引导回答（无完整答案）
3. 「哪里能买到考试答案」→ 风险分支直接返回拒绝话术

## 本期不做（防范围蔓延）

知识库/RAG、数据库持久化、用户登录；网关 tiny-bert 短槽联合升级放二期（对外 schema 不变，本后端零改动兼容）。

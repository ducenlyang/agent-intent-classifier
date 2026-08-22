"""全局配置：标签体系、置信度阈值、模型与路径、LLM 兜底配置。"""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
ROOT_DIR = PKG_DIR.parent
DISTILL_DIR = PKG_DIR / "distill_train"
CKPT_DIR = PKG_DIR / "ckpt"
DATA_DIR = PKG_DIR / "data"

TEACHER_CKPT = CKPT_DIR / "teacher_best.pt"
STUDENT_JOINT_CKPT = CKPT_DIR / "student_joint.pt"  # 联合多任务模型(意图+BIO槽位)
LABEL_MAP_PATH = DISTILL_DIR / "label_map.json"

CKPT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 一级意图（8 分类，与 label_map.json 完全对齐）
# ---------------------------------------------------------------------------
class PrimaryIntent(str, Enum):
    QUESTION_SUBJECT = "QUESTION_SUBJECT"        # 学科问题
    QUESTION_POLICY = "QUESTION_POLICY"          # 升学/考试政策问题
    REQUEST_STUDY_PLAN = "REQUEST_STUDY_PLAN"    # 学习计划请求
    REQUEST_ERROR_ANALYSIS = "REQUEST_ERROR_ANALYSIS"  # 错题/丢分分析请求
    CHAT_EMOTION = "CHAT_EMOTION"                # 情感倾诉（含心理高危）
    REFUSE_CHEAT = "REFUSE_CHEAT"                # 作弊类请求（系统拒绝）
    GENERAL_CHAT = "GENERAL_CHAT"                # 通用闲聊
    UNKNOWN = "UNKNOWN"                          # 无法识别


# ---------------------------------------------------------------------------
# 二级意图（仅第三层 LLM 精判输出）
# ---------------------------------------------------------------------------
class SecondaryIntent(str, Enum):
    # QUESTION_SUBJECT
    CONCEPT_EXPLAIN = "CONCEPT_EXPLAIN"      # 概念/定义讲解
    SOLVE_PROBLEM = "SOLVE_PROBLEM"          # 具体题目求解
    # QUESTION_POLICY
    EXAM_POLICY = "EXAM_POLICY"              # 考试/报名政策
    ADMISSION_POLICY = "ADMISSION_POLICY"    # 录取/升学政策
    # REQUEST_STUDY_PLAN
    SCHEDULE_PLANNING = "SCHEDULE_PLANNING"  # 阶段性计划制定
    GRADE_IMPROVE = "GRADE_IMPROVE"          # 提分诉求
    # REQUEST_ERROR_ANALYSIS
    MISTAKE_DIAGNOSIS = "MISTAKE_DIAGNOSIS"  # 错因诊断
    PAPER_ANALYSIS = "PAPER_ANALYSIS"        # 试卷整体分析
    # CHAT_EMOTION
    EMOTION_VENT = "EMOTION_VENT"            # 日常情绪倾诉
    EMOTION_CRISIS = "EMOTION_CRISIS"        # 心理高危（规则层命中）
    # GENERAL_CHAT
    SMALL_TALK = "SMALL_TALK"                # 闲聊
    INFO_SEEK = "INFO_SEEK"                  # 常识/信息查询
    # UNKNOWN
    UNCLEAR = "UNCLEAR"


# 每个一级意图允许的二级意图（约束 LLM 输出）
ALLOWED_SECONDARY: dict[PrimaryIntent, set[SecondaryIntent]] = {
    PrimaryIntent.QUESTION_SUBJECT: {
        SecondaryIntent.CONCEPT_EXPLAIN,
        SecondaryIntent.SOLVE_PROBLEM,
    },
    PrimaryIntent.QUESTION_POLICY: {
        SecondaryIntent.EXAM_POLICY,
        SecondaryIntent.ADMISSION_POLICY,
    },
    PrimaryIntent.REQUEST_STUDY_PLAN: {
        SecondaryIntent.SCHEDULE_PLANNING,
        SecondaryIntent.GRADE_IMPROVE,
    },
    PrimaryIntent.REQUEST_ERROR_ANALYSIS: {
        SecondaryIntent.MISTAKE_DIAGNOSIS,
        SecondaryIntent.PAPER_ANALYSIS,
    },
    PrimaryIntent.CHAT_EMOTION: {
        SecondaryIntent.EMOTION_VENT,
        SecondaryIntent.EMOTION_CRISIS,
    },
    PrimaryIntent.REFUSE_CHEAT: {SecondaryIntent.UNCLEAR},
    PrimaryIntent.GENERAL_CHAT: {
        SecondaryIntent.SMALL_TALK,
        SecondaryIntent.INFO_SEEK,
    },
    PrimaryIntent.UNKNOWN: {SecondaryIntent.UNCLEAR},
}

# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
TEACHER_MODEL_NAME = "bert-base-chinese"
# 首选 hfl/chinese-bert-wwm-ext-tiny；若源不可用则回退到同族 3 层小模型 hfl/rbt3
STUDENT_MODEL_CANDIDATES = [
    "hfl/chinese-bert-wwm-ext-tiny",
    "hfl/rbt3",
]
NUM_LABELS = len(PrimaryIntent)  # 8
MAX_LEN = 64

# ---------------------------------------------------------------------------
# 分层阈值
# ---------------------------------------------------------------------------
CONFIDENCE_HIGH = 0.85  # 第二层意图置信度 ≥ 该值直接输出，跳过第三层
SLOT_CONF_HIGH = 0.80   # 已检测到的短槽位(subject/grade)置信度须 ≥ 该值才放行

# 必填槽位校验（行业槽位填充做法：缺失槽位由下游 Agent 追问补全）
REQUIRED_SLOTS: dict[PrimaryIntent, list[str]] = {
    PrimaryIntent.QUESTION_SUBJECT: ["subject"],         # 讲题要知道讲哪科
    PrimaryIntent.REQUEST_STUDY_PLAN: ["subject", "grade"],  # 排计划要学科+年级
    PrimaryIntent.REQUEST_ERROR_ANALYSIS: ["subject"],   # 分析错题要知道哪科
    # QUESTION_POLICY / CHAT_EMOTION / REFUSE_CHEAT / GENERAL_CHAT / UNKNOWN 无必填
}

# ---------------------------------------------------------------------------
# 第三层 LLM 兜底（OpenAI 兼容接口；未配置 API Key 时自动降级为启发式精判）
# 通过环境变量配置：
#   INTENT_LLM_API_KEY   API Key（必填才启用真实 LLM）
#   INTENT_LLM_BASE_URL  默认智谱开放平台
#   INTENT_LLM_MODEL     默认 glm-4-flash
#   INTENT_LLM_TIMEOUT   秒，默认 15
# ---------------------------------------------------------------------------
LLM_API_KEY = os.getenv("INTENT_LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("INTENT_LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
LLM_MODEL = os.getenv("INTENT_LLM_MODEL", "glm-4-flash")
LLM_TIMEOUT = int(os.getenv("INTENT_LLM_TIMEOUT", "15"))

"""输出守卫节点逻辑：

1. 脚手架模式检查：网关 IntentResult.need_guide_only=True(答疑场景)时，
   拦截"完整答案泄露"话术 → 严格模式重生成一次，仍泄露则安全改写。
2. 基础安全过滤：作弊协助话术拦截替换；空输出兜底。
"""
from __future__ import annotations

# 答疑场景的"答案泄露"特征
ANSWER_LEAK_PATTERNS = [
    "最终答案", "完整答案", "答案是", "答案就是", "正确答案是",
    "所以x=", "所以 x=", "因此x=", "解得x=", "所以y=", "解得 y=",
]

# 任何 Agent 输出都不允许出现的内容（协助作弊）
CHEAT_FACILITATION = [
    "帮你代写", "代写好了", "直接抄", "抄我的", "作弊方法", "作弊技巧",
    "作弊不被发现", "暗号是", "答案发你", "枪手联系方式",
]

SAFE_FALLBACK = (
    "这道题咱们一步步来：先告诉我题目里的已知条件是什么？"
    "把条件列出来，思路就出来一半了～"
)


def guard(text: str, *, need_guide_only: bool = False) -> dict:
    """返回判定 {passed, regenerate, actions}。"""
    actions: list[str] = []
    if not text or not text.strip() or len(text.strip()) < 5:
        return {"passed": False, "regenerate": False,
                "actions": ["输出为空，替换兜底话术"]}
    unsafe = [p for p in CHEAT_FACILITATION if p in text]
    if unsafe:
        return {"passed": False, "regenerate": False,
                "actions": [f"安全拦截{unsafe}，替换兜底话术"]}
    if need_guide_only:
        leaks = [p for p in ANSWER_LEAK_PATTERNS if p in text]
        if leaks:
            return {"passed": False, "regenerate": True,
                    "actions": [f"脚手架泄露{leaks}，需严格模式重生成"]}
    return {"passed": True, "regenerate": False, "actions": []}

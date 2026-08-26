"""合成训练数据生成：模板槽位替换 + 口语增强 + 去重 + 8/1/1 划分。

产出 data/train.csv, data/val.csv, data/test.csv (text,label)
真实生产应替换为人工标注数据；本生成器保证本地开箱可训练、可演示。
"""
from __future__ import annotations

import csv
import random
from collections import Counter

from ..config import DATA_DIR, PrimaryIntent

random.seed(42)

# ---------------------------------------------------------------------------
# 槽位词池
# ---------------------------------------------------------------------------
GRADES = ["初一", "初二", "初三", "高一", "高二", "高三", "小学", "大一"]
EXAMS = ["高考", "中考", "期末考试", "月考", "期中考试", "模考", "考研"]
TOPICS = [
    "勾股定理", "一元二次方程", "二次函数", "三角函数", "数列求和", "概率",
    "立体几何", "导数", "向量", "不等式", "牛顿第二定律", "电磁感应",
    "浮力", "摩擦力", "动量守恒", "电路分析", "氧化还原反应", "化学方程式配平",
    "元素周期表", "有机化学", "光合作用", "细胞分裂", "遗传定律", "生态系统",
    "定语从句", "虚拟语气", "现在完成时", "被动语态", "文言文实词",
    "阅读理解", "完形填空", "英语作文", "议论文写作", "函数图像",
]
SCORES = ["60", "65", "70", "72", "75", "78", "80", "85", "90", "95"]
DAYS = ["30", "60", "90", "100", "120", "150", "200"]

PREFIX = ["", "", "", "老师，", "请问", "在吗？", "急急急，", "帮帮我，",
          "我想问一下", "你好，", "打扰了，", "那个，", "我想知道", "就是，"]
SUFFIX = ["", "", "", "？", "？", "谢谢", "谢谢老师", "急！", "！！", "呢？",
          "啊", "在线等", "帮我看看", "好吗", "麻烦了"]

# ---------------------------------------------------------------------------
# 各意图模板库
# ---------------------------------------------------------------------------
def q_subject() -> str:
    bank = [
        "{topic}这个知识点怎么理解",
        "帮我讲一下{topic}",
        "{topic}的公式是什么",
        "{topic}有哪些常考题型",
        "我{topic}总是学不明白，能给我讲讲吗",
        "为什么{topic}我一看就会一做就错",
        "这道题考察的是不是{topic}",
        "老师上课讲的{topic}我没听懂",
        "{grade}{topic}一般怎么考",
        "{topic}能不能用通俗的话解释一下",
        "关于{topic}有没有什么记忆口诀",
        "{exam}里{topic}占多少分",
    ]
    subs = [
        "数学选择题最后两道怎么才能做对",
        "物理大题的受力分析步骤是什么",
        "化学实验题怎么拿满分",
        "英语完形填空总是错很多怎么办",
        "语文文言文翻译有什么技巧",
        "生物遗传题的解题思路",
        "历史材料题怎么组织答案",
        "地理时区怎么算",
        "政治大题怎么背才高效",
        "导数的几何意义是什么",
        "电磁感应定律的内容是什么",
        "什么是化学键",
        "光合作用和呼吸作用的区别",
        "英语虚拟语气怎么用",
        "比喻和拟人有什么区别",
        "现在完成时的结构是什么",
        "串联电路和并联电路怎么区分",
        "如何求函数的定义域",
        "等差数列的前n项和公式",
        "议论文的论证方法有哪些",
        # ---- badcase 回流(2026-08评测): 口语化求助/元请求/步骤追问 ----
        "帮我解个题呗",
        "帮我解一道题呗",
        "帮我看看一道{subject}题呗",
        "{subject}帮我讲讲呗",
        "帮我讲讲这道题",
        "这道题怎么做呀",
        "这道题第二步看不懂",
        "这道题第{step}步没看懂",
        "{topic}这步为什么这么做",
        "这道题能给我讲讲思路吗",
        "帮我解{topic}这道题",
        "来一道{subject}题",
        "来个{subject}题练练",
        "出一个{subject}题考考我",
        "给我出{subject}题",
        "我想让你给我出一个{subject}题",
        "考我几个{subject}知识点",
    ]
    _SUBJ = ["数学", "语文", "英语", "物理", "化学", "生物", "历史", "地理"]
    return (random.choice(bank).format(
        topic=random.choice(TOPICS), grade=random.choice(GRADES),
        exam=random.choice(EXAMS), subject=random.choice(_SUBJ),
        step=random.choice(["一", "二", "三", "四", "五", "1", "2", "3"]),
    ) if random.random() < 0.55 else random.choice(subs))


def q_policy() -> str:
    bank = [
        "{exam}报名需要什么条件",
        "{exam}报名时间是什么时候",
        "异地{exam}政策是怎样的",
        "{grade}复读有什么限制吗",
        "{exam}录取分数线怎么划定",
        "{exam}加分政策有哪些",
        "强基计划怎么报名",
        "高职单招和统招有什么区别",
        "艺考生文化课分数线怎么算",
        "中考指标生是什么政策",
        "学位证和毕业证有什么区别",
        "综合素质评价影响录取吗",
        "考研调剂的流程是什么",
        "专升本需要什么条件",
        "高考志愿填报的规则是什么",
        "平行志愿是怎么录取的",
        "{grade}能复读吗",
        "体育特长生{exam}有什么政策",
        "空军招飞的报名条件",
        "地方专项计划的申请条件",
        "新高考选科有什么要求",
        "会考不及格影响毕业吗",
        "复读生{exam}有哪些限制",
        "积分入学需要什么材料",
    ]
    return random.choice(bank).format(
        exam=random.choice(EXAMS[:4] + ["考研"]), grade=random.choice(GRADES),
    )


def q_plan() -> str:
    bank = [
        "帮我制定一份{exam}复习计划",
        "距离{exam}还有{day}天，我该怎么安排复习",
        "寒假怎么安排学习",
        "暑假学习计划怎么做",
        "{grade}{subject}从{s1}分提到{s2}分要怎么学",
        "帮我做个三个月的提分计划",
        "我每天只有两小时学习时间，怎么规划",
        "{grade}一轮复习怎么规划",
        "周末两天怎么高效利用",
        "晚自习时间怎么安排最高效",
        "帮我规划一下最后{day}天的冲刺",
        "文科生怎么安排每天的背诵时间",
        "住校生怎么制定学习计划",
        "基础差的{grade}学生怎么安排复习节奏",
        "{subject}每天要花多长时间才够",
        "怎么安排每天的作息才不影响上课",
        "给我一份{exam}倒计时计划表",
        "新学期开始怎么制定学习目标",
    ]
    return random.choice(bank).format(
        exam=random.choice(EXAMS), grade=random.choice(GRADES),
        subject=random.choice(["数学", "英语", "语文", "物理", "化学"]),
        s1=random.choice(SCORES[:6]), s2=random.choice(SCORES[6:]),
        day=random.choice(DAYS),
    )


def q_error() -> str:
    bank = [
        "帮我分析一下这道错题",
        "这道题我做错了，帮我看看错在哪",
        "我{subject}总是马虎丢分怎么办",
        "上次{exam}{subject}只考了{score}分，帮我分析原因",
        "错题本应该怎么整理才有效",
        "我一遇到压轴题就放弃，怎么破",
        "帮我看看这张卷子主要问题在哪",
        "为什么我明明会做却拿不到分",
        "计算总是出错怎么改",
        "考试时间不够用怎么办",
        "我{subject}选择题错误率很高，帮我诊断一下",
        "审题不清导致的丢分怎么解决",
        "帮我分析下我最近的{exam}成绩为什么下滑",
        "草稿纸乱导致抄错数字，怎么纠正",
        "大题步骤分总是拿不全，问题出在哪",
        # ---- badcase 回流: "考砸"类孤立句曾高自信误判情绪 ----
        "我这次{exam}砸了",
        "{exam}考砸了怎么办",
        "这次{exam}砸了，帮我看看问题在哪",
        "我{exam}没考好，想找找原因",
        "{exam}发挥失常了，帮我诊断一下",
    ]
    return random.choice(bank).format(
        subject=random.choice(["数学", "英语", "语文", "物理", "化学"]),
        exam=random.choice(EXAMS), score=random.choice(SCORES[:7]),
    )


def q_emotion() -> str:
    bank = [
        "马上要{exam}了，我好焦虑",
        "这次{exam}考砸了，心情特别差",
        "我妈总是拿我和别人家孩子比，好烦",
        "感觉无论怎么努力都考不好，很挫败",
        "每天学到凌晨还是倒数，想哭",
        "同学都不喜欢我，感觉很孤独",
        "被老师当众批评了，很丢脸",
        "压力大到睡不着",
        "感觉自己好没用",
        "一想到{exam}就紧张得手心出汗",
        "父母不理解我，好委屈",
        "努力了这么久成绩没起色，好绝望",
        "坐教室里就心慌，集中不了注意力",
        "害怕考不好让爸妈失望",
        "最近老是发脾气，控制不住自己",
        "好朋友和我绝交了，心里空落落的",
        "总觉得别人都比我强，很自卑",
        "考试前一晚总是失眠",
        "写作业写到崩溃，想撕书",
        "感觉坚持不下去了，太累了",
        "我可能真的不是学习的料吧",
        "爸妈又吵架了，我根本没心思学习",
        "老师说我态度有问题，可我真的很努力了",
    ]
    return random.choice(bank).format(exam=random.choice(EXAMS))


def q_cheat() -> str:
    # 一部分含规则层关键词(会被第1层拦截)，一部分委婉表达(考验模型本身)
    bank = [
        "考试的时候能不能用{thing}",
        "有没有渠道提前拿到{exam}试卷",
        "哪里能买到{exam}真题答案",
        "帮我{thing2}吧，我可以给钱",
        "考场上怎么{thing}才不会被发现",
        "听说有人能提前知道题目，能帮我问问吗",
        "花钱能让别人替我去考试吗",
        "有没有那种考试前就能弄到答案的群",
        "找人帮我代考{subject}靠谱吗",
        "能不能考试的时候坐我旁边给我传答案",
        "扫描笔能带进考场吗，想悄悄查题",
        "有没有隐形耳机卖，考试用的",
        "答案发我一下，明天要考试了",
        "谁有这次{exam}的答案，急",
        "帮我把小抄做小一点，不容易被看到",
        "考试用手机拍答案会被发现吗",
        "有人代写{subject}作文吗，付费",
        "考前押题的内部答案真的假的",
        "保过班说的内部试卷是不是泄题",
        "怎么在答题卡上做暗号给同桌",
    ]
    return random.choice(bank).format(
        exam=random.choice(["高考", "中考", "期末考试", "四六级", "考研"]),
        subject=random.choice(["数学", "英语", "语文"]),
        thing=random.choice(["小抄", "作弊器", "翻译笔", "智能手表"]),
        thing2=random.choice(["代写作业", "代写论文", "代考", "写论文"]),
    )


def q_chat() -> str:
    bank = [
        "今天天气真好",
        "推荐几部好看的电影",
        "讲个笑话呗",
        "你会唱歌吗",
        "中午吃什么好",
        "你叫什么名字",
        "陪我聊聊天",
        "你觉得自己聪明吗",
        "无聊，说点有趣的",
        "最近有个热搜你看了吗",
        "你喜欢吃什么",
        "你们公司在哪里",
        "放假去哪里玩比较好",
        "怎么交到更多朋友",
        "人为什么要睡觉",
        "宇宙有多大",
        "世界上最高的山是什么",
        "你能干什么",
        "你是真人吗",
        "你多大了",
        "周末在家好无聊啊",
        "推荐几本课外书",
        "有什么好听的歌推荐",
        "猫和狗哪个好养",
        "你会说英语吗",
        "给我讲个历史故事",
        "运动完好累但是很开心",
    ]
    return random.choice(bank)


def q_unknown() -> str:
    bank = [
        "asdfghjkl", "。。。", "？？？", "嗯嗯嗯", "12345", "哈哈哈哈哈哈",
        "嗯？", "啊这", "（沉默）", "哦", "在", "不", "这个", "那个那个",
        "@@@@", "？？？？？？", "。。", "。。？", "发错人了", "测试",
        "你好你好你好", "1", "888", "e", "q", "？？？干嘛", "额",
        "你说啥", "嗯嗯好的好的", "算了算了", "无所谓", "随便",
        "jfkdlsajf", "，，，。。。",
    ]
    return random.choice(bank)


GENS = {
    PrimaryIntent.QUESTION_SUBJECT: (q_subject, 520),
    PrimaryIntent.QUESTION_POLICY: (q_policy, 360),
    PrimaryIntent.REQUEST_STUDY_PLAN: (q_plan, 340),
    PrimaryIntent.REQUEST_ERROR_ANALYSIS: (q_error, 340),
    PrimaryIntent.CHAT_EMOTION: (q_emotion, 420),
    PrimaryIntent.REFUSE_CHEAT: (q_cheat, 320),
    PrimaryIntent.GENERAL_CHAT: (q_chat, 400),
    PrimaryIntent.UNKNOWN: (q_unknown, 280),
}


def augment(text: str, emo_chat: bool) -> str:
    """口语化增强：随机前后缀；聊天类偶尔加语气词。"""
    if emo_chat and random.random() < 0.3:
        text += random.choice(["[泪]", "[裂开]", "[失望]", "😭", "😢", "🙏"])
    if random.random() < 0.5:
        text = random.choice(PREFIX) + text
    if random.random() < 0.4:
        text = text + random.choice(SUFFIX)
    return text


def main() -> None:
    data: list[tuple[str, str]] = []
    seen: set[str] = set()
    for intent, (gen, n) in GENS.items():
        got = 0
        while got < n:
            text = augment(gen(), intent in (PrimaryIntent.CHAT_EMOTION, PrimaryIntent.GENERAL_CHAT))
            text = text.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            data.append((text, intent.value))
            got += 1

    random.shuffle(data)
    n = len(data)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    splits = {
        "train": data[:n_train],
        "val": data[n_train:n_train + n_val],
        "test": data[n_train + n_val:],
    }
    for name, rows in splits.items():
        path = DATA_DIR / f"{name}.csv"
        # utf-8-sig(带BOM): 中文版 Excel 双击打开不乱码；训练读取端同样用 utf-8-sig
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["text", "label"])
            w.writerows(rows)
        print(f"{path}  {len(rows)} 条  {dict(Counter(l for _, l in rows))}")


if __name__ == "__main__":
    main()

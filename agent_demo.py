"""
agent_demo.py - 最小 agent 闭环（接入百炼真模型版）

对照讲解里的消息流图：
  ① system      定义角色和工具
  ② user        用户提问
  ③ assistant   模型返回"调用请求"（不是文字）
      ↓ 你的代码执行工具
  ④ tool        把执行结果作为新消息塞回去
  ⑤ assistant   模型返回文字答案，循环结束

架构不变：call_model 是唯一和"模型"相关的地方。
  - 真模型：走百炼 OpenAI 兼容接口
  - 假模型：设环境变量 USE_FAKE_MODEL=1，无需 key 也能跑，方便对照

运行前：
  在 Windows 用户环境变量里设好 DASHSCOPE_API_KEY、BOCHA_API_KEY
  pip install openai
"""

import json
import os
import re
import urllib.request
import urllib.error

# ============================================================
# 第 0 块：配置 —— 直接从环境变量读取
# ============================================================
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")       # 百炼 key
BOCHA_KEY = os.getenv("BOCHA_API_KEY", "")         # 博查搜索 key
MODEL = os.getenv("MODEL_NAME", "qwen-flash")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
USE_FAKE = os.getenv("USE_FAKE_MODEL", "0") == "1"
MAX_STEPS = 10  # 防止模型反复调工具导致死循环


# ============================================================
# 第 1 块：工具函数 —— 就一个：联网搜索
# ============================================================
def search_web(query: str, count: int = 5) -> dict:
    """调用博查 Web Search API"""
    if not BOCHA_KEY:
        return {"error": "未配置 BOCHA_API_KEY，请在环境变量里设置"}

    payload = {
        "query": query,
        "freshness": "noLimit",
        "summary": True,
        "count": count,
    }
    req = urllib.request.Request(
        "https://api.bocha.cn/v1/web-search",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {BOCHA_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        return {"error": f"搜索失败 HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:200]}"}
    except Exception as e:
        return {"error": f"搜索失败: {e}"}

    # 博查的结果在 data.webPages.value[]
    pages = data.get("data", {}).get("webPages", {}).get("value", [])
    results = []
    for p in pages:
        results.append({
            "title": p.get("name", ""),
            "url": p.get("url", ""),
            "snippet": p.get("summary") or p.get("snippet", ""),
            "site": p.get("siteName", ""),
        })
    return {"query": query, "results": results}

def calculate(expression: str) -> dict:
    """四则运算，用 ast 白名单解析，不用裸 eval"""
    import ast, operator
    ops = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv}
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](ev(node.left), ev(node.right))
        raise ValueError("只支持数字和 + - * /")
    try:
        return {"result": ev(ast.parse(expression, mode="eval").body)}
    except Exception as e:
        return {"error": str(e)}


def get_current_time() -> dict:
    """获取当前的日期和时间（本地时间）"""
    from datetime import datetime
    now = datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()],
    }


# 工具注册表：模型说要调哪个工具，就从这个表里按名字找到函数
TOOL_FUNCS = {
    "search_web": search_web,
    "calculate": calculate,
    "get_current_time": get_current_time,
}


# ============================================================
# 第 1.5 块：信息层 —— 去重 + 排序 + 编号（深度研究的核心）
# ============================================================
def _title_key(title: str) -> str:
    """标题指纹：先清 HTML 实体，再只留中英文数字，用来识别同一篇的转载"""
    for _ in range(2):  # &amp;nbsp; 这种双重转义
        title = title.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", title).lower()


def process_results(query: str, results: list) -> list:
    """把搜索结果整理干净：去重 → 按相关性排序 → 编号，供模型引用"""
    # ① 去重：标题互为前缀（同一篇，只是网站后缀不同）只留第一条
    seen, unique = [], []
    for r in results:
        key = _title_key(r["title"])
        # 两个标题足够长且互为前缀 → 判定为同一篇转载
        duplicate = any(
            len(k) >= 15 and len(key) >= 15 and (k.startswith(key) or key.startswith(k))
            for k in seen
        )
        if duplicate:
            continue
        seen.append(key)
        unique.append(r)

    # ② 排序：query 里的关键词在标题/摘要里命中越多越靠前
    keywords = [w for w in re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", query) if w]
    def relevance(r):
        text = r["title"] + r["snippet"]
        return sum(text.count(w) for w in keywords)
    ranked = sorted(unique, key=relevance, reverse=True)

    # ③ 编号：每条带上 [1][2]，模型回答时可以引用
    for i, r in enumerate(ranked, 1):
        r["n"] = i
    return ranked


def format_context(results: list) -> str:
    """把整理好的结果拼成带编号的文本，喂给模型当参考材料"""
    lines = []
    for r in results:
        lines.append(
            f"[{r['n']}] {r['title']}（来源：{r['site']}）\n"
            f"    链接：{r['url']}\n"
            f"    摘要：{r['snippet'][:400]}"
        )
    return "\n\n".join(lines)


# ============================================================
# 第 2 块：工具说明书 —— OpenAI function calling 的标准格式
# ============================================================
# description 写得好不好，直接决定模型会不会调、参数传得准不准。
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "联网搜索最新的真实信息。当用户询问新闻、时事、技术资料、人物、产品、任何需要实时或外部知识的问题时使用。返回标题、链接和摘要。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词，要具体，例如：2026年 Python 最新版本"},
                    "count": {"type": "integer", "description": "返回结果条数，默认 5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式。当用户需要做加减乘除运算时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "算式，如 12 * 8"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间。当用户询问现在几点、今天日期、星期几等时间相关问题时使用。不需要任何参数。",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


# ============================================================
# 第 3 块：call_model —— 唯一和"模型"相关的地方
# ============================================================
# 真模型 / 假模型都返回统一结构：
#   {
#     "content": 文字答案 或 None,
#     "tool_calls": [{"id", "name", "arguments"}] 或 None,
#     "assistant_msg": 要追加进 messages 的 assistant 消息
#   }
# 这样下面的循环完全不用关心用的是谁。
def call_model(messages: list, force_tool: bool = False) -> dict:
    if USE_FAKE:
        return _fake_model(messages)

    # 延迟导入：假模型模式下不需要装 openai
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS_SCHEMA,
        # required=强制调工具（第一轮用，防止没搜索就凭空回答）；
        # auto=让模型自己决定调工具还是直接回答
        tool_choice="required" if force_tool else "auto",
    )
    msg = resp.choices[0].message

    calls = None
    if msg.tool_calls:
        calls = []
        for tc in msg.tool_calls:
            calls.append({
                "id": tc.id,
                "name": tc.function.name,
                # 真模型的 arguments 是 JSON 字符串，这里解析成 dict
                "arguments": json.loads(tc.function.arguments),
            })

    return {
        "content": msg.content,
        "tool_calls": calls,
        # 真模型必须把原始 msg 对象塞回去（它带着 tool_calls 的标准结构，
        # 下一轮 API 要靠它把 tool 结果和调用请求对上号）
        "assistant_msg": msg,
    }


def _fake_model(messages: list) -> dict:
    """假模型：按关键词模拟决定，返回值结构和真模型一致"""
    last = messages[-1]

    if last["role"] == "user":
        q = last["content"]
        # 只有一个工具，所有问题都走搜索
        return {
            "content": None,
            "tool_calls": [{"id": "fake_1", "name": "search_web", "arguments": {"query": q}}],
            "assistant_msg": {"role": "assistant", "content": None,
                              "tool_calls": [{"id": "fake_1", "type": "function",
                                              "function": {"name": "search_web",
                                                           "arguments": json.dumps({"query": q}, ensure_ascii=False)}}]},
        }

    if last["role"] == "tool":
        # 现在工具返回的是整理后的文本，直接把它当参考材料展示
        answer = "搜索到这些结果（已去重排序编号）：\n" + last["content"][:600]
        return {"content": answer, "tool_calls": None,
                "assistant_msg": {"role": "assistant", "content": answer}}

    return {"content": "我不知道该怎么处理。", "tool_calls": None,
            "assistant_msg": {"role": "assistant", "content": "我不知道该怎么处理。"}}


# ============================================================
# 第 4 块：agent 循环 —— 整个项目的心脏（和假模型版几乎一样）
# ============================================================
def run_agent(question: str):
    print("=" * 60)
    print(f"用户提问：{question}")
    print(f"当前模型：{'假模型' if USE_FAKE else MODEL}")
    print("=" * 60)

    messages = [
        {"role": "system", "content": (
            "你是一个严谨的研究助手。"
            "【工具选择】涉及天气、新闻、时事、价格、最新数据等时效性信息时，"
            "即使你自以为知道，也必须先调用 search_web 搜索验证。"
            "【引用规则】事实性结论（数据、事件、天气等）用 [编号] 标注来源，"
            "例如：杭州今日有雨 [1]；同一事实有多个来源时标多个编号，如 [1][3]。"
            "建议、注意事项、推断部分不用标注编号。"
            "【严禁编造】没有实际搜索过就不要写任何 [编号]，只依据搜索结果作答。"
        )},
        {"role": "user", "content": question},
    ]

    # B 方案标记：幻觉拦截（强制补搜）最多执行一次，防止死循环
    hallucination_fixed = False

    for step in range(1, MAX_STEPS + 1):
        print(f"\n--- 第 {step} 轮 ---")

        # ③：问模型（工具选择交给模型自己决定）
        reply = call_model(messages)
        messages.append(reply["assistant_msg"])

        # 判断：要调工具，还是给答案？
        if reply["tool_calls"]:
            for call in reply["tool_calls"]:
                name = call["name"]
                args = call["arguments"]
                print(f"   [模型决定调工具] {name}({args})")
                result = TOOL_FUNCS[name](**args)

                # 搜索结果先过信息层：去重 → 排序 → 编号
                if name == "search_web" and "results" in result:
                    before = len(result["results"])
                    results = process_results(args["query"], result["results"])
                    print(f"   [信息层] 去重排序：{before} 条 → {len(results)} 条")
                    content = format_context(results)
                    # 日志：每条都显示，摘要压成单行、只取前 80 字（塞给模型的是完整内容）
                    def _clean(s):
                        for _ in range(2):  # 处理 &amp;nbsp; 这种双重转义
                            s = s.replace("&nbsp;", " ").replace("&amp;", "&")
                        return re.sub(r"\s+", " ", s).strip()  # 换行/多空格压成一个空格

                    print(f"   [工具返回] 共 {len(results)} 条：")
                    for r in results:
                        title = re.split(r"\s+-\s+", _clean(r["title"]))[0][:34]  # 去掉标题里的网站后缀
                        site = re.sub(r"\s+", "", _clean(r["site"]))  # 去掉所有空白（含全角/不间断空格）
                        snippet = _clean(r["snippet"])[:80]
                        print(f"      [{r['n']}] {title}（{site}）")
                        print(f"          {snippet}...")
                else:
                    content = json.dumps(result, ensure_ascii=False)
                    print(f"   [工具返回] {content[:300]}...")
                # ④：结果作为 tool 消息塞回去，tool_call_id 用来对上是哪次调用
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": content,
                })
            continue
        else:
            # ⑤：文字答案 —— 出结果前先做"幻觉校验"（B 方案）
            answer = reply["content"] or ""
            # 答案里是否带 [编号] 引用
            refs = re.findall(r"\[\d+\]", answer)
            # 消息历史里是否真的有搜索结果（带 [n] 的 tool 消息）
            # 注意：messages 里既有 dict，也有真模型返回的 ChatCompletionMessage 对象
            def _field(m, key):
                return m.get(key) if isinstance(m, dict) else getattr(m, key, None)

            has_search = any(
                _field(m, "role") == "tool" and re.search(r"\[\d+\]", str(_field(m, "content") or ""))
                for m in messages
            )

            if refs and not has_search and not hallucination_fixed:
                # 造假：标了引用却从没搜过 → 撤掉这条答案，强制补搜一次让它重答
                print(f"   [幻觉拦截] 答案带引用 {refs} 但从未搜索 → 强制补搜")
                fake_msg = reply["assistant_msg"]
                messages.pop()  # 撤掉刚塞进去的这条造假答案
                result = TOOL_FUNCS["search_web"](query=question)
                if "results" in result:
                    results = process_results(question, result["results"])
                    content = format_context(results)
                    print(f"   [补搜] {len(result['results'])} 条 → 去重后 {len(results)} 条")
                    messages.append({"role": "assistant", "content": None,
                                     "tool_calls": [{"id": "force_1", "type": "function",
                                                     "function": {"name": "search_web",
                                                                  "arguments": json.dumps({"query": question}, ensure_ascii=False)}}]})
                    messages.append({"role": "tool", "tool_call_id": "force_1", "content": content})
                hallucination_fixed = True
                continue

            print(f"\n最终答案：{answer}")
            print(f"\n(共跑 {step} 轮，最终消息列表有 {len(messages)} 条)")
            return

    print(f"\n达到最大轮数 {MAX_STEPS}，停止。")


# ============================================================
# 第 4.5 块：多轮调研（深度研究）—— 拆问题 → 搜多次 → 汇总成报告
# 和 run_agent 并存，旧函数一行不改，随时能切回去对比。
# ============================================================
def _plain_chat(messages: list) -> str:
    """不带工具的纯文本问答，供"规划"和"写报告"两个阶段用"""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    resp = client.chat.completions.create(model=MODEL, messages=messages)
    return resp.choices[0].message.content or ""


def plan_questions(question: str) -> list:
    """① 让模型自己把大问题拆成 3-4 个子问题（自主规划，不是代码写死）"""
    if USE_FAKE:
        return [question + " 现状", question + " 最新进展"]

    prompt = (
        "你是一个研究规划助手。请把下面的研究问题拆成 3-4 个可以独立搜索的子问题，"
        "覆盖问题的不同方面，让每个子问题都能单独搜到有用信息。\n"
        "只返回一个 JSON 数组，每个元素是一个子问题字符串，不要输出任何其他文字。\n"
        '例如：["子问题一", "子问题二", "子问题三"]\n\n'
        f"研究问题：{question}"
    )
    text = _plain_chat([{"role": "user", "content": prompt}])
    # 容错解析：模型可能包 ```json``` 加前后缀，抠出第一个 [ 到最后一个 ]
    try:
        start, end = text.index("["), text.rindex("]")
        subs = json.loads(text[start:end + 1])
        subs = [str(s).strip() for s in subs if str(s).strip()][:4]
        if subs:
            return subs
    except Exception:
        pass
    return [question]  # 解析失败就退化成原来的单问题搜索


def _collect_search(sub_query: str) -> list:
    """②③ 单个子问题：搜索 → 信息层处理 → 打印日志，返回结果列表"""
    result = search_web(sub_query)
    if "results" not in result or not result["results"]:
        print(f"   [搜索失败] {result.get('error', '无结果')}")
        return []
    before = len(result["results"])
    processed = process_results(sub_query, result["results"])
    print(f"   [信息层] 去重排序：{before} 条 → {len(processed)} 条")

    def _clean(s):
        for _ in range(2):
            s = s.replace("&nbsp;", " ").replace("&amp;", "&")
        return re.sub(r"\s+", " ", s).strip()

    print(f"   [工具返回] 共 {len(processed)} 条：")
    for r in processed:
        title = re.split(r"\s+-\s+", _clean(r["title"]))[0][:34]
        site = re.sub(r"\s+", "", _clean(r["site"]))
        snippet = _clean(r["snippet"])[:80]
        print(f"      [{r['n']}] {title}（{site}）")
        print(f"          {snippet}...")
    return processed


def _merge_all(batches: list) -> list:
    """④ 跨批次全局去重 + 重新编号（避免每个子问题都从 [1] 开始）"""
    seen, merged = [], []
    for r in batches:
        key = _title_key(r["title"])
        duplicate = any(
            len(k) >= 15 and len(key) >= 15 and (k.startswith(key) or key.startswith(k))
            for k in seen
        )
        if duplicate:
            continue
        seen.append(key)
        merged.append(r)
    for i, r in enumerate(merged, 1):
        r["n"] = i
    return merged


def run_deep_research(question: str):
    print("=" * 60)
    print(f"用户提问：{question}")
    print(f"模式：多轮调研 · {'假模型' if USE_FAKE else MODEL}")
    print("=" * 60)

    # ① 规划：模型自己拆子问题
    print("\n[① 规划拆解]")
    subs = plan_questions(question)
    print(f"   模型拆出 {len(subs)} 个子问题：")
    for i, s in enumerate(subs, 1):
        print(f"   {i}. {s}")

    # ②③ 逐个搜索 + 信息层（同一套代码循环执行多次）
    print("\n[②③ 逐个搜索 + 信息层]")
    batches = []
    for i, sub in enumerate(subs, 1):
        print(f"\n   --- 子问题 {i}/{len(subs)}：{sub} ---")
        batches.extend(_collect_search(sub))

    # ④ 汇总去重 + 全局重新编号
    print("\n[④ 汇总去重]")
    merged = _merge_all(batches)
    print(f"   所有批次共 {len(batches)} 条 → 全局去重后 {len(merged)} 条")
    context = format_context(merged)

    # ⑤ 生成报告：基于全部材料作答
    print("\n[⑤ 生成报告]")
    report_messages = _build_report_messages(question, context)
    if USE_FAKE:
        answer = f"（假模型）参考资料共 {len(merged)} 条：\n{context[:600]}"
    else:
        answer = _plain_chat(report_messages)
    print(f"\n最终报告：\n{answer}")
    print(f"\n(搜索 {len(subs)} 次，收集 {len(merged)} 条资料)")
    return answer


def _build_report_messages(question: str, context: str) -> list:
    """写报告的 prompt，命令行版和 API 版共用一处，改一次两边都生效"""
    today = get_current_time()["date"]  # 真实日期，避免模型瞎编落款
    return [
        {"role": "system", "content": (
            "你是一个严谨的研究助手。请严格依据提供的参考资料撰写一份结构清晰的报告。"
            "事实性结论用 [编号] 标注来源，编号只能使用资料中出现过的；"
            "建议、推断部分不标编号。资料不足时如实说明，不要编造。"
            f"【当前日期】今天是 {today}（由工具获取）。"
            "报告末尾如写更新日期，必须严格使用此日期，不得编造其他日期。"
        )},
        {"role": "user", "content": (
            f"研究问题：{question}\n\n"
            f"参考资料（已去重排序编号）：\n{context}\n\n"
            "请基于以上全部资料写一份完整报告，覆盖问题的各个方面，并标注引用。"
        )},
    ]


def research(question: str) -> dict:
    """给 API 用的纯函数：不打印日志，返回结构化结果"""
    subs = plan_questions(question)
    batches = []
    for sub in subs:
        result = search_web(sub)
        if "results" in result and result["results"]:
            batches.extend(process_results(sub, result["results"]))
    merged = _merge_all(batches)
    context = format_context(merged)
    if USE_FAKE:
        report = f"（假模型）参考资料共 {len(merged)} 条"
    else:
        report = _plain_chat(_build_report_messages(question, context))
    return {
        "question": question,
        "report": report,
        "sub_questions": subs,
        "sub_question_count": len(subs),
        "sources_count": len(merged),
    }


# ============================================================
# 第 5 块：跑一个例子
# ============================================================
if __name__ == "__main__":
    run_deep_research("AI Agent 框架 LangGraph 和 AutoGen 有什么区别，各自适合什么场景？")

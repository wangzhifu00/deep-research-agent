"""
agent_demo.py - 深度研究 Agent（ReAct + Function Calling + 多轮调研）

单轮 ReAct（run_agent）：
  system → user → assistant(调用请求) → 代码执行工具 → tool(结果) → assistant(答案)

多轮调研（run_deep_research / research）：
  ① 规划：模型自主拆 3-4 个子问题
  ② 并发搜索：ThreadPoolExecutor 并发检索（结果保序）
  ③ 信息层：跨批次去重 → 排序 → 全局重新编号
  ④ 生成报告：基于全部材料作答
  ⑤ 引用校验闭环：越界编号 → 带反馈重答，最多 MAX_CITATION_RETRY 次

运行前：
  在 Windows 用户环境变量里设好 DASHSCOPE_API_KEY、BOCHA_API_KEY
  pip install -r requirements.txt
"""

import json
import os
import re
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

# ============================================================
# 第 0 块：配置 —— 直接从环境变量读取
# ============================================================
API_KEY = os.getenv("DASHSCOPE_API_KEY", "")       # 百炼 key
BOCHA_KEY = os.getenv("BOCHA_API_KEY", "")         # 博查搜索 key
MODEL = os.getenv("MODEL_NAME", "qwen-flash")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MAX_STEPS = 10                 # ReAct 单轮最大步数，防止死循环
MAX_CITATION_RETRY = 2         # 报告引用越界时，最多带反馈重答几次


# ============================================================
# 第 1 块：工具函数 —— 三个工具
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
# 第 1.6 块：公共步骤层 —— 三条流程共用，消除重复代码
# ============================================================
def _clean(s: str) -> str:
    """清洗 HTML 实体和多余空白，只用于日志显示（喂给模型的是原文）"""
    for _ in range(2):
        s = s.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", s).strip()


def log_results(results: list):
    """统一打印编号结果日志（命令行演示用）"""
    print(f"   [工具返回] 共 {len(results)} 条：")
    for r in results:
        title = re.split(r"\s+-\s+", _clean(r["title"]))[0][:34]
        site = re.sub(r"\s+", "", _clean(r["site"]))
        snippet = _clean(r["snippet"])[:80]
        print(f"      [{r['n']}] {title}（{site}）")
        print(f"          {snippet}...")


def search_and_process(query: str, verbose: bool = False) -> list:
    """公共步骤：联网搜索 → 信息层去重排序编号 → 返回结果列表。

    verbose=True 时打印过程日志（命令行演示）；API 模式传 False 保持纯函数。
    run_agent / run_deep_research / research 三处都调用这一个实现。
    """
    result = search_web(query)
    if "results" not in result or not result["results"]:
        if verbose:
            print(f"   [搜索失败] {result.get('error', '无结果')}")
        return []
    before = len(result["results"])
    processed = process_results(query, result["results"])
    if verbose:
        print(f"   [信息层] 去重排序：{before} 条 → {len(processed)} 条")
        log_results(processed)
    return processed


def collect_batches(subs: list, verbose: bool = False) -> list:
    """公共步骤：对多个子问题批量搜索，返回所有批次结果（保持子问题顺序）。

    verbose=True：串行 + 打印（演示时看得清每个子问题）；
    verbose=False：ThreadPoolExecutor 并发（API 提速），map 天然保序。
    """
    if verbose:
        batches = []
        for i, sub in enumerate(subs, 1):
            print(f"\n   --- 子问题 {i}/{len(subs)}：{sub} ---")
            batches.extend(search_and_process(sub, verbose=True))
        return batches

    if not subs:
        return []
    with ThreadPoolExecutor(max_workers=min(len(subs), 4)) as ex:
        groups = list(ex.map(lambda q: search_and_process(q, verbose=False), subs))
    batches = []
    for group in groups:
        batches.extend(group)
    return batches


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
def call_model(messages: list) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",  # 工具选择权交给模型自己判断
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


# ============================================================
# 第 4 块：ReAct 单轮循环（和深度研究并存，可随时切回对比）
# ============================================================
def run_agent(question: str):
    print("=" * 60)
    print(f"用户提问：{question}")
    print(f"当前模型：{MODEL}")
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

    # 幻觉拦截（强制补搜）最多执行一次，防止死循环
    hallucination_fixed = False

    for step in range(1, MAX_STEPS + 1):
        print(f"\n--- 第 {step} 轮 ---")

        reply = call_model(messages)
        messages.append(reply["assistant_msg"])

        if reply["tool_calls"]:
            for call in reply["tool_calls"]:
                name = call["name"]
                args = call["arguments"]
                print(f"   [模型决定调工具] {name}({args})")
                if name == "search_web":
                    # 复用公共步骤：搜索 → 信息层 → 打印日志
                    results = search_and_process(args.get("query", ""), verbose=True)
                    content = format_context(results)
                else:
                    fn = TOOL_FUNCS.get(name)
                    if fn is None:
                        content = json.dumps({"error": f"未知工具：{name}"}, ensure_ascii=False)
                    else:
                        result = fn(**args)
                        content = json.dumps(result, ensure_ascii=False)
                    print(f"   [工具返回] {content[:300]}...")
                # 结果作为 tool 消息塞回去，tool_call_id 用来对上是哪次调用
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": content,
                })
            continue
        else:
            # 文字答案 —— 出结果前做"幻觉校验"
            answer = reply["content"] or ""
            refs = re.findall(r"\[\d+\]", answer)
            # messages 里既有 dict，也有真模型返回的 ChatCompletionMessage 对象
            def _field(m, key):
                return m.get(key) if isinstance(m, dict) else getattr(m, key, None)

            has_search = any(
                _field(m, "role") == "tool" and re.search(r"\[\d+\]", str(_field(m, "content") or ""))
                for m in messages
            )

            if refs and not has_search and not hallucination_fixed:
                # 造假：标了引用却从没搜过 → 撤掉答案，强制补搜一次让它重答
                print("   [幻觉拦截] 答案带引用但从未搜索 → 强制补搜")
                messages.pop()
                results = search_and_process(question, verbose=True)
                if results:
                    content = format_context(results)
                    print(f"   [补搜] 去重后 {len(results)} 条")
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
# 第 5 块：多轮调研（深度研究）
# ============================================================
def _plain_chat(messages: list) -> str:
    """不带工具的纯文本问答，供"规划"和"写报告"两个阶段用"""
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    resp = client.chat.completions.create(model=MODEL, messages=messages)
    return resp.choices[0].message.content or ""


def plan_questions(question: str) -> list:
    """① 让模型自己把大问题拆成 3-4 个子问题（自主规划，不是代码写死）"""
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


def validate_citations(answer: str, valid_nums: set) -> tuple:
    """校验答案里的 [n] 是否都在合法编号集合内。

    返回 (是否全部合法, 越界编号列表)。这是引用校验闭环的判定环节。
    """
    used = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    invalid = sorted(n for n in used if n not in valid_nums)
    return (not invalid, invalid)


def generate_report(question: str, context: str, valid_nums: set, verbose: bool = False) -> str:
    """⑤ 生成报告 + 引用校验闭环。

    发现越界编号（如材料只有 5 条却写了 [9]）时，把错误反馈塞回对话让模型重答，
    最多 MAX_CITATION_RETRY 次。正常一次通过时只有一次模型调用。
    """
    messages = _build_report_messages(question, context)
    answer = _plain_chat(messages)
    for attempt in range(1, MAX_CITATION_RETRY + 1):
        ok, invalid = validate_citations(answer, valid_nums)
        if ok:
            if verbose:
                print(f"   [引用校验] 通过，引用编号均在 1–{max(valid_nums) if valid_nums else 0} 范围内")
            break
        if verbose:
            print(f"   [引用校验] 第 {attempt} 次发现越界编号 {invalid} → 带反馈重答")
        messages = messages + [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": (
                f"你上一版引用了 {invalid} 这些编号，但参考资料里并不存在这些编号。"
                "请重新写一份完整报告，所有 [编号] 必须是资料中真实出现过的，"
                "不要编造编号；资料不足的地方如实说明。"
            )},
        ]
        answer = _plain_chat(messages)
    return answer


def run_deep_research(question: str):
    """命令行演示版：带完整过程日志"""
    print("=" * 60)
    print(f"用户提问：{question}")
    print(f"当前模型：{MODEL}")
    print("=" * 60)

    print("\n[① 规划拆解]")
    subs = plan_questions(question)
    print(f"   模型拆出 {len(subs)} 个子问题：")
    for i, s in enumerate(subs, 1):
        print(f"   {i}. {s}")

    print("\n[②③ 并发搜索 + 信息层]")
    batches = collect_batches(subs, verbose=True)

    print("\n[④ 汇总去重]")
    merged = _merge_all(batches)
    print(f"   所有批次共 {len(batches)} 条 → 全局去重后 {len(merged)} 条")
    context = format_context(merged)

    print("\n[⑤ 生成报告 + 引用校验]")
    valid_nums = {r["n"] for r in merged}
    answer = generate_report(question, context, valid_nums, verbose=True)
    print(f"\n最终报告：\n{answer}")
    print(f"\n(搜索 {len(subs)} 次，收集 {len(merged)} 条资料)")
    return answer


def research(question: str) -> dict:
    """给 API 用的纯函数：不打印日志，返回结构化结果"""
    subs = plan_questions(question)
    batches = collect_batches(subs, verbose=False)
    merged = _merge_all(batches)
    context = format_context(merged)
    valid_nums = {r["n"] for r in merged}
    report = generate_report(question, context, valid_nums)
    return {
        "question": question,
        "report": report,
        "sub_questions": subs,
        "sub_question_count": len(subs),
        "sources_count": len(merged),
    }


if __name__ == "__main__":
    run_deep_research("AI Agent 框架 LangGraph 和 AutoGen 有什么区别，各自适合什么场景？")

"""深度研究 Agent 单元测试

全部外部依赖（联网搜索、大模型）都用 mock 打桩：
不联网、不消耗 API 额度、不需要配置 key，任何人 clone 后都能直接跑。

运行方式（项目根目录）：
    python -m unittest discover -s tests -v

覆盖的核心逻辑：
    1. ast 白名单安全计算（含恶意代码拦截）
    2. 标题清洗与跨批次去重
    3. 引用编号校验 + 越界重答闭环
    4. 并发检索保序
    5. plan_questions 容错解析
    6. research 端到端结构
"""

import os
import sys
import time
import unittest
from unittest.mock import patch

# 保证从任意目录运行都能 import 到项目根目录的 agent_demo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agent_demo as a


def fake_result(title, site="测试站", snippet=""):
    """构造一条符合 search_web 返回格式的材料"""
    return {
        "title": title,
        "url": "https://example.com/article",
        "snippet": snippet or title,
        "site": site,
    }


class TestCalculate(unittest.TestCase):
    """工具层：ast 白名单安全计算"""

    def test_basic_arithmetic(self):
        self.assertEqual(a.calculate("12 * (3 + 4)")["result"], 84)
        self.assertEqual(a.calculate("10 / 4")["result"], 2.5)

    def test_rejects_code_injection(self):
        # 函数调用节点不在白名单 → 必须拒绝，绝不执行
        r = a.calculate("__import__('os').system('echo hacked')")
        self.assertIn("error", r)
        self.assertNotIn("result", r)

    def test_rejects_unsupported_ops(self):
        # 幂运算 Pow、一元负号 UnaryOp 都不在白名单
        self.assertIn("error", a.calculate("2 ** 3"))
        self.assertIn("error", a.calculate("-5"))


class TestTitleDedup(unittest.TestCase):
    """信息层：HTML 实体清洗 + 标题互为前缀去重"""

    def test_title_key_cleans_html_entities(self):
        self.assertEqual(a._title_key("A&nbsp;B&amp;C"), a._title_key("ABC"))

    def test_prefix_duplicate_removed(self):
        results = [
            fake_result("LangGraph 和 AutoGen 对比 - CSDN", "CSDN"),
            # 同一篇的转载，标题带站点后缀，指纹互为前缀 → 应被去掉
            fake_result("LangGraph 和 AutoGen 对比 - CSDN 技术专栏", "CSDN"),
            fake_result("完全不同的一篇文章", "掘金"),
        ]
        out = a.process_results("LangGraph AutoGen", results)
        self.assertEqual(len(out), 2, "3 条里有 1 条转载应被去掉")
        self.assertEqual([r["n"] for r in out], [1, 2], "编号必须连续")


class TestCitationValidation(unittest.TestCase):
    """可靠性：引用校验闭环"""

    def test_validate_citations(self):
        ok, bad = a.validate_citations("结论 [1] 和 [2]", {1, 2, 3})
        self.assertTrue(ok)
        self.assertEqual(bad, [])
        ok, bad = a.validate_citations("结论 [1] 和 [9]", {1, 2, 3})
        self.assertFalse(ok)
        self.assertEqual(bad, [9])

    def test_retry_on_bad_citation(self):
        # 第一次引用了不存在的 [9]，第二次改成合法的 [1]
        answers = ["错误报告 [9]", "修正报告 [1]"]
        with patch.object(a, "_plain_chat", side_effect=answers) as m:
            out = a.generate_report("问题", "参考资料 [1] xxx", {1})
        self.assertEqual(out, "修正报告 [1]")
        self.assertEqual(m.call_count, 2, "越界后应带反馈重答一次")

    def test_single_call_when_valid(self):
        # 一次通过时不应有多余的模型调用（不多花钱）
        with patch.object(a, "_plain_chat", return_value="正常报告 [1]") as m:
            out = a.generate_report("问题", "参考资料 [1] xxx", {1})
        self.assertEqual(out, "正常报告 [1]")
        self.assertEqual(m.call_count, 1)


class TestCollectBatches(unittest.TestCase):
    """并发：ThreadPoolExecutor 检索 + map 保序"""

    def test_parallel_keeps_order(self):
        def fake_search(query):
            time.sleep(0.05)  # 模拟网络延迟；若串行会明显更慢
            return {"results": [fake_result("资料：" + query)]}

        subs = ["问题甲", "问题乙", "问题丙"]
        with patch.object(a, "search_web", side_effect=fake_search):
            batches = a.collect_batches(subs)  # verbose=False → 并发分支
        titles = [r["title"] for r in batches]
        self.assertEqual(titles, ["资料：问题甲", "资料：问题乙", "资料：问题丙"],
                         "并发执行但结果顺序必须与子问题一致")


class TestPlanQuestions(unittest.TestCase):
    """任务拆解：模型输出带包裹时的容错解析"""

    def test_parse_json_in_code_fence(self):
        raw = '好的：\n```json\n["子一", "子二", "子三"]\n```'
        with patch.object(a, "_plain_chat", return_value=raw):
            subs = a.plan_questions("大问题")
        self.assertEqual(subs, ["子一", "子二", "子三"])

    def test_fallback_on_bad_output(self):
        with patch.object(a, "_plain_chat", return_value="我无法拆分这个问题"):
            self.assertEqual(a.plan_questions("大问题"), ["大问题"])


class TestResearch(unittest.TestCase):
    """端到端：research() 返回给 API 的结构"""

    def test_research_structure(self):
        def fake_chat(messages):
            # 规划阶段只有 1 条消息；写报告阶段有 system + user 多条
            if len(messages) == 1:
                return '["子问题一", "子问题二"]'
            return "最终报告 [1][2]"

        def fake_search(query):
            return {"results": [fake_result("资料：" + query)]}

        with patch.object(a, "_plain_chat", side_effect=fake_chat), \
                patch.object(a, "search_web", side_effect=fake_search):
            res = a.research("测试问题")

        self.assertEqual(res["question"], "测试问题")
        self.assertEqual(res["sub_questions"], ["子问题一", "子问题二"])
        self.assertEqual(res["sub_question_count"], 2)
        self.assertEqual(res["sources_count"], 2)
        self.assertEqual(res["report"], "最终报告 [1][2]")


if __name__ == "__main__":
    unittest.main(verbosity=2)

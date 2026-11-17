# 深度研究 Agent 系统

基于 ReAct + Function Calling 的深度研究 Agent。模型自主把一个研究问题拆成 3–4 个子问题，**并发**调用联网搜索工具检索，再把多来源结果跨批次去重、排序、统一编号，生成带引用的研究报告，并对引用编号做**校验闭环**（越界编号自动反馈重答）。提供命令行和 FastAPI 两种入口。

## 技术栈

- **模型**：Qwen（阿里云百炼，OpenAI 兼容接口）
- **Agent 编排**：ReAct 循环（Thought → Action → Observation）、Function Calling 工具调用、LLM 自主任务拆解与多轮工具编排
- **工具层**：博查 Web Search（联网搜索）、计算器、当前时间
- **信息层**：跨批次去重、相关性排序、统一编号、引用溯源
- **并发**：ThreadPoolExecutor 并发检索多个子问题，map 保序，总耗时压到单次搜索级别
- **可靠性**：幻觉抑制（"带引用却未检索"时补搜重答）+ 引用校验闭环（编号越界时带反馈重答）、工具参数校验、ast 白名单安全计算、防死循环
- **服务化**：FastAPI + Pydantic 参数校验，自动生成 OpenAPI 文档
- **测试**：12 项 unittest 单测，搜索与模型全部打桩，零成本验证核心逻辑

## 环境要求

- Python 3.10+

## 安装

```bash
pip install -r requirements.txt
```

## 配置

在项目根目录新建 `.env` 文件（**不要提交到 Git**），或直接在系统环境变量里设置：

```bash
BOCHA_API_KEY=你的博查APIKey
DASHSCOPE_API_KEY=你的百炼APIKey
```

- 博查 API Key：https://open.bochaai.com 的 API Key 管理页获取
- 百炼 API Key：阿里云百炼控制台获取
- 可选：`MODEL_NAME`（默认 `qwen-flash`）

设置环境变量后需要**重新打开**终端 / PyCharm 才能读到。

## 运行

### 方式一：命令行

```bash
python agent_demo.py
```

内置一个示例问题，打印完整的检索与报告过程。

### 方式二：API 服务

```bash
python api.py
```

或：

```bash
uvicorn api:app --reload
```

启动后打开：

- 接口文档：http://127.0.0.1:8000/docs
- 调研接口：`POST /research`

请求示例：

```json
{"question": "AI Agent 框架 LangGraph 和 AutoGen 有什么区别？"}
```

响应包含 `report`（报告正文）、`sub_questions`（拆解的子问题）、`sub_question_count`（子问题数）、`sources_count`（去重后来源数）。

## 测试

```bash
python -m unittest discover -s tests -v
```

12 项单测覆盖：ast 白名单安全计算（含恶意代码拦截）、标题清洗与去重、引用编号校验与越界重答闭环、并发检索保序、任务拆解容错、端到端结构。

搜索和模型调用全部用 mock 打桩，**不需要配置 key、不联网、不消耗 API 额度**，clone 后可直接运行。

## 文件结构

| 文件 | 说明 |
| --- | --- |
| `agent_demo.py` | 核心逻辑：ReAct 循环、工具层、公共步骤层（搜索+信息层复用）、并发检索、任务拆解、引用校验闭环、报告生成 |
| `api.py` | FastAPI 服务封装，`/research` 接口 |
| `tests/test_agent.py` | 12 项单元测试（外部依赖全打桩，零成本运行） |
| `requirements.txt` | 依赖清单 |
| `.env.example` | 环境变量示例（需自行复制为 `.env` 并填入 key） |

## 注意

- `.env` 含 API Key，已加入 `.gitignore`，**不要提交或分享**。
- 博查免费试用包有调用次数限制，用完需自行充值或替换搜索源。

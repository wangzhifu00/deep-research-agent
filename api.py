"""
api.py - 把多轮调研包装成 HTTP 接口（FastAPI）

启动：
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

接口：
    GET  /              健康检查
    POST /research      提交问题，返回调研报告
    GET  /docs          自动生成的交互式 API 文档

调用示例（PowerShell）：
    Invoke-RestMethod -Uri http://127.0.0.1:8000/research `
        -Method Post -ContentType "application/json" `
        -Body '{"question": "LangGraph 和 AutoGen 的区别"}'
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent_demo import research

app = FastAPI(
    title="深度研究 Agent API",
    description="多轮调研：自动拆解问题 → 多次联网搜索 → 汇总去重 → 生成带引用报告",
    version="1.0.0",
)


# 请求体：Pydantic 自动做类型校验和数据转换
class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="要调研的问题")


# 响应体：明确返回结构，FastAPI 会生成 OpenAPI 文档
class ResearchResponse(BaseModel):
    question: str
    report: str
    sub_questions: list[str]
    sub_question_count: int
    sources_count: int


@app.get("/")
def health():
    """健康检查：确认服务活着"""
    return {"status": "ok", "service": "deep-research-agent"}


@app.post("/research", response_model=ResearchResponse)
def run_research(req: ResearchRequest):
    """
    提交一个研究问题，返回完整报告。

    注意这里用普通 def 而不是 async def：research() 内部是同步网络请求，
    FastAPI 会自动把普通 def 放进线程池执行，不会阻塞事件循环。
    """
    try:
        return research(req.question)
    except Exception as e:
        # 把异常转成 500，而不是让服务返回一堆 HTML 错误页
        raise HTTPException(status_code=500, detail=f"调研失败：{e}")


if __name__ == "__main__":
    # 直接运行本文件也能起服务（开发用）；正式启动推荐上面的 uvicorn 命令
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

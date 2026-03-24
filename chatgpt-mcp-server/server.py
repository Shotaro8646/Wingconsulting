#!/usr/bin/env python3
"""ChatGPT MCP Server - Claude Code から OpenAI API を呼び出すための MCP サーバー"""

import os
import json
import sys
from mcp.server.fastmcp import FastMCP
from openai import OpenAI

# OpenAI API キーの設定
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY environment variable is required", file=sys.stderr)
    sys.exit(1)

client = OpenAI(api_key=api_key)

mcp = FastMCP("chatgpt")


@mcp.tool()
def ask_chatgpt(prompt: str, model: str = "gpt-4o") -> str:
    """ChatGPT にプロンプトを送信してレスポンスを取得する。

    Args:
        prompt: ChatGPT に送るプロンプト
        model: 使用するモデル名 (デフォルト: gpt-4o)
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


@mcp.tool()
def ask_chatgpt_with_system(
    system_message: str,
    prompt: str,
    model: str = "gpt-4o",
) -> str:
    """システムメッセージ付きで ChatGPT にプロンプトを送信する。

    Args:
        system_message: ChatGPT に設定するシステムメッセージ
        prompt: ChatGPT に送るプロンプト
        model: 使用するモデル名 (デフォルト: gpt-4o)
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


@mcp.tool()
def list_openai_models() -> str:
    """利用可能な OpenAI モデルの一覧を取得する。"""
    models = client.models.list()
    result = [{"id": m.id, "owned_by": m.owned_by} for m in models.data]
    result.sort(key=lambda x: x["id"])
    return json.dumps(result, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")

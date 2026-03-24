#!/usr/bin/env python3
"""Gemini MCP Server - Claude Code から Gemini API を呼び出すための MCP サーバー"""

import os
import json
import sys
from mcp.server.fastmcp import FastMCP
from google import genai

# Gemini API キーの設定
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY environment variable is required", file=sys.stderr)
    sys.exit(1)

client = genai.Client(api_key=api_key)

mcp = FastMCP("gemini")


@mcp.tool()
def ask_gemini(prompt: str, model: str = "gemini-2.0-flash") -> str:
    """Gemini にプロンプトを送信してレスポンスを取得する。

    Args:
        prompt: Gemini に送るプロンプト
        model: 使用するモデル名 (デフォルト: gemini-2.0-flash)
    """
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text


@mcp.tool()
def ask_gemini_with_context(
    system_instruction: str,
    prompt: str,
    model: str = "gemini-2.0-flash",
) -> str:
    """システムインストラクション付きで Gemini にプロンプトを送信する。

    Args:
        system_instruction: Gemini に設定するシステムインストラクション
        prompt: Gemini に送るプロンプト
        model: 使用するモデル名 (デフォルト: gemini-2.0-flash)
    """
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
        ),
    )
    return response.text


@mcp.tool()
def list_gemini_models() -> str:
    """利用可能な Gemini モデルの一覧を取得する。"""
    models = client.models.list()
    result = []
    for model in models:
        result.append(
            {
                "name": model.name,
                "display_name": model.display_name,
                "description": model.description,
            }
        )
    return json.dumps(result, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")

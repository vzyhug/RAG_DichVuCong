"""Verify the configured OpenAI-compatible endpoint through LLMFactory."""

from __future__ import annotations

import argparse
import asyncio


async def _verify(prompt: str) -> str:
    from configs.settings import settings
    from src.llm.model_factory import LLMFactory

    if settings.LLM_PROVIDER != "local":
        raise RuntimeError(
            "Set LLM_PROVIDER=local before verifying the fine-tuned endpoint"
        )
    client = LLMFactory.get_llm()
    response = await client.chat.completions.create(
        model=LLMFactory.get_model_name(),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=32,
    )
    print(f"provider={settings.LLM_PROVIDER}")
    print(f"endpoint={settings.LOCAL_LLM_URL}")
    print(f"model={LLMFactory.get_model_name()}")
    return response.choices[0].message.content or ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Reply with exactly: LOCAL_LLM_OK")
    args = parser.parse_args(argv)
    answer = asyncio.run(_verify(args.prompt))
    print(f"response={answer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

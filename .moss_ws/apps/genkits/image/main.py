"""text_to_image — MOSS App for generating images from text prompts."""

import os

from dotenv import load_dotenv
from ghoshell_moss import Matrix
from ghoshell_moss.core.blueprint.channel_builder import (
    new_channel,
    CommandUtil,
)
from ghoshell_moss.core.blueprint.mindflow import Signal, Priority, InputSignalMeta
from ghoshell_moss.message import Message

from image_generator import (
    ImageGenerator,
    GenerationResult,
    GenerationStatus,
    DoubaoImageGeneratorProvider,
    QwenImageGeneratorProvider,
)

# Load .env from .moss_ws root
load_dotenv()

chan = new_channel(
    name="text_to_image",
    description="Generate images from text prompts using AI models. "
    "Returns a locator (pil-image://...) that can be used to display the image.",
)


async def emit_generation_signal(result: GenerationResult) -> None:
    """根据 GenerationResult 统一构造并发送 Signal。"""
    if result.status == GenerationStatus.COMPLETED:
        priority = Priority.NOTICE
        description = f"图片已生成: {result.prompt[:50]}"
        prompt_text = (
            f"tag={result.tag or 'N/A'}, locator={result.locator}"
        )
    else:
        priority = Priority.ERROR
        description = f"图片生成失败: {result.prompt[:50]}"
        prompt_text = (
            f"图片生成失败（{result.error}）。"
            f"tag={result.tag or 'N/A'}，请检查后重试。"
        )

    async def _closure() -> Signal:
        return InputSignalMeta().to_signal(
            priority=priority,
            description=description,
            hint=prompt_text,
        )

    await CommandUtil.create_signal_task(closure=_closure)


@chan.build.command(
    name="generate",
    doc="Generate an image from a text prompt. Returns a locator (pil-image://...) "
    "that can be used to view or download the image. "
    "tag is an opaque identifier returned as-is in the channel status, "
    "not interpreted by the generator.",
    timeout=180.0,
    blocking=False,
)
async def generate(
    prompt: str,
    tag: str = "",
) -> str | None:
    """Generate image and return its locator.

    :param prompt: Text description of the image.
    :param tag: Opaque identifier, stored and reported back without interpretation.
    :return: locator string (pil-image://...)，失败时返回 None（错误信息通过 Signal 传递）。
    """
    generator = CommandUtil.force_get_contract(ImageGenerator)
    result = await generator.generate(prompt=prompt, tag=tag)
    await emit_generation_signal(result)
    return result.locator


@chan.build.context_messages
async def channel_context() -> list[Message]:
    """向 Ghost 描述 channel 当前状态（动态）。"""
    messages: list[Message] = []

    # ── 能力描述（静态）──
    messages.append(
        Message.new().with_content(
            "text_to_image channel: 使用 generate(prompt, tag) 生成图片。\n"
            "返回值：pil-image://workspace-assets/{id} 格式的 locator 字符串。\n"
        )
    )

    # ── 从 ImageGenerator 读取当前会话状态 ──
    try:
        generator = CommandUtil.force_get_contract(ImageGenerator)
        state = generator.get_state()

        if state:
            pending: list[dict] = []
            completed: list[dict] = []
            failed: list[dict] = []
            for _, result in state.items():
                entry = {
                    "tag": result.tag,
                    "prompt": result.prompt[:80],
                    "locator": result.locator or "",
                    "error": result.error or "",
                }
                if result.status == GenerationStatus.COMPLETED:
                    completed.append(entry)
                elif result.status == GenerationStatus.FAILED:
                    failed.append(entry)
                else:
                    pending.append(entry)

            status_lines = [
                f"当前会话生成状态：pending={len(pending)} completed={len(completed)} failed={len(failed)}"
            ]
            for entry in completed:
                status_lines.append(
                    f"  ✓ tag={entry['tag'] or '-'} prompt='{entry['prompt']}' locator={entry['locator']}"
                )
            for entry in pending:
                status_lines.append(
                    f"  ○ tag={entry['tag'] or '-'} prompt='{entry['prompt']}' (生成中...)"
                )
            for entry in failed:
                status_lines.append(
                    f"  ✗ tag={entry['tag'] or '-'} prompt='{entry['prompt']}' error={entry['error']}"
                )
            messages.append(Message.new().with_content("\n".join(status_lines)))
        else:
            messages.append(Message.new().with_content("当前无正在生成的任务。"))
    except KeyError:
        messages.append(
            Message.new().with_content("ImageGenerator 未注册 — 请检查 API key 配置。")
        )

    return messages


async def main(_matrix: Matrix) -> None:
    """Connect this channel to the Matrix network.

    Register ImageGenerator IoC Provider(s) based on available API keys.
    Prefers doubao (VOLCENGINE_API_KEY) over qwen (QWEN_API_KEY).
    """
    # Register ImageGenerator based on available API keys
    if os.getenv("VOLCENGINE_API_KEY"):
        _matrix.container.register(DoubaoImageGeneratorProvider())
    if os.getenv("QWEN_API_KEY"):
        _matrix.container.register(QwenImageGeneratorProvider())

    await _matrix.provide_channel(chan)


if __name__ == "__main__":
    matrix = Matrix.discover()
    matrix.run(main)

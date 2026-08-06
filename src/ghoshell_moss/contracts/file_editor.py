"""File editor contract — MOSS 的结构化文件编辑动作层.

一句话承诺: "在文件上做 5 个模型都会用的动作 — view / create / str_replace /
insert / undo_edit — 语义骑 Anthropic text_editor 血统, 契约不做加法".

结构:

1. FileEditor — sync ABC. 5 个动词, 每个返回 FileEditorResult.
2. FileEditorResult — 结果 dataclass (path + old_content + new_content + output).
3. 异常层 — FileEditorError 基类 + 4 个具体错误.

承担的读侧 (v2 追加, 供无 bash 的 sandbox):

- file_list — 即时目录列表 (非递归, name/size/kind).
- glob — glob 模式文件发现 (pathlib 语义, 支持 ** 递归).
- grep — 单文件正则行检索, 可扩展 kwargs.

不承担的:

- view(dir) — 目录列表走 file_list, view 仍只收文件路径 (K7 不复活).
- 二进制文件读写 — 抛 FileValidationError. 未来若要 read binary 独立立项.
- 有 bash 的场景 — 上述三个动词不是给有 bash 的 channel 用的, 收益在
  sandbox 无 bash 的场景 (memento agent 等). 契约做加法, 用不用随消费者.
- 空间边界强制 — workspace_root 只做相对路径 hint (与 Grounds 的 root
  故意分裂: file editor 不假设 Ground 存在).
- 并发/异步 — sync 接口. 想 async 的 caller 自己 asyncio.to_thread.
  见 feedback_io_contract_sync.md.
- Linting — vendor 层砍掉了, 不进契约.

三条验收平面:

1. contracts + core 单测 (无 vendor 直接 import, 走 DefaultFileEditor)
2. 未来 channel 装配 (K33 ground_channel 落地时合体)
3. 未来 CLI dogfood (moss edit / moss view 之类, 暂未立项)
"""

# 技术目标 (reviewer 上下文, 契约演进见 FEATURE.md file-editor-contract):
#
# K1 (契约独立, 不入 ground/contract.py) — Grounds 与 FileEditor 最小依赖不同:
# Pin 只在 Ground 里存在, FileEditor.write(path) 只需一个 path. 塞一起是
# 融合病灶复发. channel 层可合体, contract 层不合体.
#
# K3 (sync, 不加 async) — 见 feedback_io_contract_sync.md. Python async
# 是灾难; 加 async 到契约 = 非 asyncio 环境不可用. IO 本质 sync, 想 async
# 的 caller 自己 to_thread.
#
# K4 (动词对齐 OHEditor 5 命令, 不发明) — Anthropic text_editor tool 血统,
# SOTA 模型都见过. K16 双寄存器词汇要求"骑先验".
#
# K5 (异常层 MOSS 化, 不透传 vendor) — vendor 异常类不出 core/file_editor/.
# adapter 内部 catch ToolError 家族, 转为 FileEditorError.

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "FileEditor",
    "FileEditorResult",
    "FileEditorError",
    "ParameterMissingError",
    "ParameterInvalidError",
    "FileValidationError",
    "NoEditHistoryError",
]


# -- 结果 --

@dataclass
class FileEditorResult:
    """一次 file editor 动作的结果.

    K4 姿态: 结构对齐 OHEditor 的 CLIResult, 但换成 MOSS 命名 (dataclass
    而非 vendor 的独立类, 避免 vendor 类型出契约). CTML `<result>` 机制
    直接消费本对象的 output 字段.
    """

    output: str
    """人类可读的动作摘要 — 也是模型看到的主 payload. 通常包含 cat -n 风格
    的 snippet, `File created` / `File edited` 前缀, 以及建议下一步的
    提示语句 (来自 vendor 层, 预训练锁死不改)."""

    path: str
    """动作作用的绝对路径."""

    prev_exist: bool = True
    """动作前文件是否已存在. create 命令下为 False, 其它为 True."""

    old_content: str | None = None
    """动作前的文件全文 (str_replace / insert / undo_edit 会填).
    create 与 view 不填."""

    new_content: str | None = None
    """动作后的文件全文 (create / str_replace / insert / undo_edit 会填).
    view 不填."""


# -- 异常 --

class FileEditorError(Exception):
    """File editor 契约层异常基类. 独立于其他契约层, 不共享继承.

    K1: FileEditor 与 Grounds 契约独立, 异常层也独立. Vendor 层 (OHEditor)
    的 ToolError 家族在 DefaultFileEditor 边界处转译为本类家族, 不透传.
    """


class ParameterMissingError(FileEditorError):
    """必填参数缺失. 例如 create 无 file_text, str_replace 无 old_str."""

    def __init__(self, command: str, parameter: str):
        self.command = command
        self.parameter = parameter
        super().__init__(
            f"Parameter `{parameter}` is required for command `{command}`."
        )


class ParameterInvalidError(FileEditorError):
    """参数取值非法. 例如 view_range 超出文件行数, view 用于目录路径,
    路径非绝对路径, create 目标已存在等."""

    def __init__(self, parameter: str, value, hint: str | None = None):
        self.parameter = parameter
        self.value = value
        msg = f"Invalid `{parameter}` parameter: {value!r}."
        if hint:
            msg += f" {hint}"
        super().__init__(msg)


class FileValidationError(FileEditorError):
    """文件不通过验证 — 过大 / 二进制 / 编码不可读.

    K4 副产品: 上游 OHEditor 的 SUPPORTED_BINARY_EXTENSIONS (office/pdf/audio
    转 markdown) 已在 vendor patch 里删掉. 现在所有二进制文件统一走本异常,
    需要读 office/pdf 的 caller 自己走 bash / 未来独立 read-binary feature.
    """

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"File validation failed for {path}: {reason}")


class NoEditHistoryError(FileEditorError):
    """undo_edit 时该文件无编辑历史.

    历史是 per-process session 状态 (见 vendor patch: 磁盘缓存已去除,
    改用内存 deque). 进程重启后历史清空.
    """

    def __init__(self, path: str):
        self.path = path
        super().__init__(f"No edit history found for {path}.")


# -- 主契约 --

class FileEditor(ABC):
    """结构化文件编辑器.

    5 个动词, 全 sync. 每个动词返回 FileEditorResult, 异常统一 FileEditorError
    家族. 路径参数是绝对路径 (str 或 Path); 相对路径的处理由 workspace_root
    构造参数决定 hint 行为, 但**不做强制解析** (K1: 空间边界由上层收敛,
    file editor 不假设 Ground 存在).

    实现见 ``ghoshell_moss.core.file_editor.DefaultFileEditor`` (vendor
    openhands-aci editor 子集, 见 core/file_editor/_openhands/UPSTREAM.md).
    """

    @abstractmethod
    def view(
        self,
        path: str | Path,
        view_range: list[int] | None = None,
    ) -> FileEditorResult:
        """读一个文件的内容 (可选行区间).

        - path: 文件绝对路径. 目录路径抛 ParameterInvalidError
          (目录列表用 bash/glob, 不在本工具辖域).
        - view_range: 可选 [start_line, end_line] (1-based, 含端点).
          None = 全文. 超出文件行数抛 ParameterInvalidError.

        返回 FileEditorResult.output 是 `cat -n` 风格的带行号 snippet
        (vendor 血统, 预训练锁死). old_content / new_content 均为 None.

        文件过大或二进制抛 FileValidationError. 路径不存在抛
        ParameterInvalidError.
        """
        ...

    @abstractmethod
    def file_list(
        self,
        path: str | Path = ".",
    ) -> FileEditorResult:
        """列出目录的即时子项 (非递归).

        v2 追加 — 供无 bash 的 sandbox 做目录发现. 有 bash 的 channel 不需要
        暴露它 (bash/glob 先验更通用); 契约做加法, 用不用随消费者.

        - path: 目录路径. 相对路径按 workspace_root (未设则进程 cwd) 解析,
          绝对路径原样. 指向文件抛 ParameterInvalidError (读文件用 view),
          不存在抛 FileValidationError.
        - 输出: 每行一条 ``name  size  kind``. kind ∈ file/dir/symlink,
          size 人类可读 (B/K/M), dirs 排序在前, 含 dotfile.
        - 不做空间边界强制 (K4: 那是上层 / Grounds 的职责).
        """
        ...

    @abstractmethod
    def glob(
        self,
        pattern: str,
    ) -> FileEditorResult:
        """glob 模式匹配文件发现 (pathlib.glob 语义).

        v2 追加 — 与 file_list 同为无 bash sandbox 的文件发现原语.

        - pattern: ``*.py`` / ``**/*.py`` (递归). 相对 → base = workspace_root
          (未设则进程 cwd); 绝对模式 (``/x/y/**/*.py``) 接受.
          **不支持 brace 展开** (``*.{py,txt}``, pathlib 限制) — docstring
          提示模型用 ``**/*.py`` 风格.
        - 输出: 匹配的绝对路径, 每行一条, 上限 100 + ``(N more omitted)`` 注.
        - 空 pattern 抛 ParameterMissingError. 零匹配正常返回空列表消息.
        """
        ...

    @abstractmethod
    def grep(
        self,
        pattern: str,
        path: str | Path,
        **options: object,
    ) -> FileEditorResult:
        """单文件正则行检索.

        v2 追加 — 无 bash sandbox 的内容搜索原语. **可扩展 kwargs**: 签名收
        ``**options``, 实现定义已识别键, 未来加键不破坏签名. 未知键抛
        ParameterInvalidError (配置错误要响).

        已识别 (默认):
        - ``case_sensitive: bool = True`` — False 时忽略大小写.
        - ``max_results: int = 100`` — 返回匹配行上限, 超出注省略.

        未来候选 (docstring 预留, 不实现): recursive / context / include_binary /
        regex.

        - pattern: 正则 (re.search 语义). 空抛 ParameterMissingError.
        - path: 目标文件. 相对/绝对同 file_list. 目录抛 ParameterInvalidError,
          不存在 / 二进制 / 非 utf-8 抛 FileValidationError.
        - 输出: ``lineno: line`` 每行一条, 上限 max_results.
        """
        ...

    @abstractmethod
    def create(
        self,
        path: str | Path,
        file_text: str,
    ) -> FileEditorResult:
        """新建一个文件, 写入 file_text.

        - path: 目标绝对路径. **已存在**抛 ParameterInvalidError
          (不允许 create 覆盖 — 覆盖走 str_replace).
        - file_text: 全文内容. 必填.

        返回 FileEditorResult, prev_exist=False, new_content=file_text.

        父目录不存在时**不自动创建** — 上游行为是抛 ToolError 转
        FileValidationError. Caller 自行 mkdir.
        """
        ...

    @abstractmethod
    def str_replace(
        self,
        path: str | Path,
        old_str: str,
        new_str: str,
    ) -> FileEditorResult:
        """精确字符串替换. 唯一匹配才成功.

        - old_str: 被替换的原文. 若在文件里出现 **多次**抛 FileEditorError
          (`No replacement was performed. Multiple occurrences ...`) —
          caller 需要加更多上下文让 old_str 变唯一. 这是设计, 不是 bug.
          若 **零次**匹配, 会先尝试 strip 首尾空白再匹配一次, 仍零次则抛错.
        - new_str: 替换后的内容. 与 old_str **相同**抛 ParameterInvalidError.
        - 空字符串 new_str 合法 (相当于删除).

        返回 FileEditorResult, old_content=修改前全文, new_content=修改后全文.
        output 包含改动附近的 snippet (SNIPPET_CONTEXT_WINDOW=4 行上下文).

        编辑历史入 undo 栈 (per-file, 最近 10 次).
        """
        ...

    @abstractmethod
    def insert(
        self,
        path: str | Path,
        insert_line: int,
        new_str: str,
    ) -> FileEditorResult:
        """在指定行**之后**插入内容.

        - insert_line: 插入锚点. 0 = 文件开头之前, N (文件总行数) = 文件末尾.
          越界抛 ParameterInvalidError.
        - new_str: 插入的文本. 多行自动 split by '\\n', 每行独立写入.

        返回 FileEditorResult 同 str_replace.

        编辑历史入 undo 栈.
        """
        ...

    @abstractmethod
    def undo_edit(self, path: str | Path) -> FileEditorResult:
        """撤销该文件的最近一次编辑.

        无历史抛 NoEditHistoryError. 历史 per-file per-process, 进程重启清空
        (vendor patch 去除了磁盘持久, 见 UPSTREAM.md).

        Undo 后**不入 undo 栈** — undo 的 undo 需要用编辑动作 (str_replace/
        insert) 而不是再 undo, 与 vim `u` / `U` 姿态相同.

        返回 FileEditorResult, old_content=undo 前全文, new_content=undo 后
        (即 undo 前的上一版本) 全文.
        """
        ...

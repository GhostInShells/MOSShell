"""Volcengine SAUC ASR 配置 — 豆包大模型流式语音识别 (bigmodel_async)。

固定部分 (url/凭据/音频格式/模型身份) 与可变部分 (行为参数 params、热词/上下文 corpus)
分离。行为参数经 get_info().params_schema 反射给模型; 热词/上下文是火山专属面,
不进 ASR 抽象, 通过 VolcengineSaucASR 的火山面方法配置。

官方文档: https://docs.volcengine.com/docs/6561/1354869?lang=zh
"""
import os

from pydantic import BaseModel, Field
from ghoshell_moss.contracts import ConfigType

__all__ = ["VolcengineSaucConfig", "VolcengineSaucParams", "VolcengineSaucCorpus"]


class VolcengineSaucParams(BaseModel):
    """行为参数 — 反射给模型看 (ASR 抽象面的 params_schema)。

    configure() 校验后更新, 作用于下一次 recognize() (每次连接 init 下发)。
    """

    end_window_size: int = Field(
        800,
        description="VAD 静音判停阈值 (ms)。连续静音达该值判定一句结束并触发分句。范围 [300,5000], 推荐 [800,1000]。",
    )
    force_to_speech_time: int = Field(
        0,
        description="音频流起始阶段强制按有声处理的时长 (ms), 规避起始静音/弱音导致的过早判停。推荐 1000。",
    )
    enable_nonstream: bool = Field(
        False,
        description="二遍识别: 开启后 VAD 分句 + 非流式二次识别, 仅二次识别结果带 definite=true。",
    )
    enable_itn: bool = Field(True, description="逆文本归一化 (口语转书面格式)")
    enable_punc: bool = Field(True, description="标点")
    enable_ddc: bool = Field(False, description="语义顺滑 (删除停顿词/语气词/重复词)")
    show_utterances: bool = Field(True, description="输出分句/分词/时间戳信息")
    result_type: str = Field("full", description="结果返回方式: full 全量 / single 增量")
    vad_segment_duration: int = Field(
        3000,
        description="语义分句最大静音阈值 (ms), 仅影响语义分句, 不触发判停, 不改变 definite 位置。",
    )


class VolcengineSaucCorpus(BaseModel):
    """热词 + 上下文 — 火山专属面, 不进 ASR 抽象。

    通过 VolcengineSaucASR.configure_corpus() 配置; 修改后下一次 recognize() 的 init
    会带上最新 corpus。
    """

    hotwords: list[str] = Field(default_factory=list, description="直传热词列表")
    boosting_table_name: str = Field(default="", description="热词词表名称 (控制台自学习平台)")
    boosting_table_id: str = Field(default="", description="热词词表 id")
    context_type: str = Field(default="", description="上下文类型, 目前仅 dialog_ctx")
    context_data: list[dict] = Field(default_factory=list, description="历史对话上下文 [{speaker, text}]")

    def context_payload(self) -> dict | None:
        """把热词/上下文折叠成 init request 里的 context 字段 (dict 或 None)。"""
        data: dict = {}
        if self.hotwords:
            data["hotwords"] = [{"word": w} for w in self.hotwords]
        if self.context_type and self.context_data:
            data["context_type"] = self.context_type
            data["context_data"] = self.context_data
        return data or None


class VolcengineSaucConfig(ConfigType):
    """火山引擎大模型流式 ASR 配置。

    固定部分: url / 凭据 / 音频格式 / 模型身份 (每实例固定, 工厂选择模型)。
    可变部分: params (行为旋钮)。热词/上下文 (corpus) 是动态的, 不在此, 在 ASR 实例火山面上配置。

    环境变量:
        SEEDASR_API_KEY  — 新控制台 API Key (X-Api-Key)
    """

    api_key: str = Field(default="$SEEDASR_API_KEY", description="新控制台 API Key (X-Api-Key)")
    url: str = Field(default="wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async",
                     description="大模型流式识别地址")
    sample_rate: int = Field(default=16000, description="默认采样率")
    bits: int = Field(default=16, description="位深")
    channel: int = Field(default=1, description="通道数")
    model_name: str = Field(default="bigmodel", description="模型名称 — 每实例固定")
    resource_id: str = Field(default="volc.seedasr.sauc.duration",
                             description="X-Api-Resource-Id (豆包 2.0, 搭配 X-Api-Key 鉴权)")

    params: VolcengineSaucParams = Field(
        default_factory=VolcengineSaucParams,
        description="行为参数 (模型反身性调参面)",
    )

    @classmethod
    def conf_name(cls) -> str:
        return 'seed_asr_config'

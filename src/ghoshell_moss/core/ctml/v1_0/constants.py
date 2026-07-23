from ghoshell_moss.core.concepts.channel import ChannelRuntime

__all__ = [
    'POSITION_ARGS_KEY', 'SCOPE_SHORTCUT', 'SCOPE_CHANNEL_NAME_KEY', 'CALL_ID_RESERVE_KEY', 'SCOPE_COMMAND_NAME',
    'SCOPE_QUANTIFIER_ALL', 'SCOPE_QUANTIFIER_ANY', 'SCOPE_QUANTIFIER_TAGS',
    'SCOPE_UNTIL_LEGACY_FLOW', 'SCOPE_UNTIL_VALID_VALUES', 'SCOPE_TAG_NAMES',
    'CONTENT_COMMAND_NAME',
    'MAIN_CHANNEL_NAME', 'MAIN_CHANNEL_SHORTCUT',
    'MOSS_DYNAMIC', 'MOSS_STATIC',
    'SCOPE_ENTER_COMMAND_NAME',
    'SCOPE_EXIT_COMMAND_NAME',
]

MAIN_CHANNEL_NAME = '__main__'
MAIN_CHANNEL_SHORTCUT = ''
POSITION_ARGS_KEY = "_args"
SCOPE_SHORTCUT = '_'
SCOPE_COMMAND_NAME = '__scope__'
# CTML v1.0.0: 量词升格为标签名. `<_>` 默认态 (occupy 链跑完即闭合);
# `<all>` 等所有并行子任务; `<any>` 任一完成掐掉其余.
# 三者是 scope 保留字, 也是 until 的合法取值 (`_` 对应 until=None).
SCOPE_QUANTIFIER_ALL = 'all'
SCOPE_QUANTIFIER_ANY = 'any'
# 标签名 → until 语义映射. `<_ until='all'>` 属性写法保留解析兼容, prompt 不暴露.
SCOPE_QUANTIFIER_TAGS: dict[str, str | None] = {
    SCOPE_SHORTCUT: None,
    SCOPE_QUANTIFIER_ALL: SCOPE_QUANTIFIER_ALL,
    SCOPE_QUANTIFIER_ANY: SCOPE_QUANTIFIER_ANY,
}
# until 合法取值集合 (含历史 'flow'; 'flow' 与 None 语义一致, 兼容层归一).
SCOPE_UNTIL_LEGACY_FLOW = 'flow'
SCOPE_UNTIL_VALID_VALUES: frozenset[str] = frozenset({
    SCOPE_UNTIL_LEGACY_FLOW, SCOPE_QUANTIFIER_ALL, SCOPE_QUANTIFIER_ANY,
})
# scope 保留字集合 (parser 判定 + channel builder 撞名警告的唯一权威).
SCOPE_TAG_NAMES: frozenset[str] = frozenset({
    SCOPE_SHORTCUT, SCOPE_QUANTIFIER_ALL, SCOPE_QUANTIFIER_ANY, SCOPE_COMMAND_NAME,
})
SCOPE_ENTER_COMMAND_NAME = ChannelRuntime.__scope_enter__.__name__
SCOPE_EXIT_COMMAND_NAME = ChannelRuntime.__scope_exit__.__name__
CONTENT_COMMAND_NAME = ChannelRuntime.__content__.__name__
CALL_ID_RESERVE_KEY = '_cid'
SCOPE_CHANNEL_NAME_KEY = 'channel'

MOSS_DYNAMIC = 'moss_dynamic'
MOSS_STATIC = 'moss_static'

from ghoshell_moss.host.tui_entries.ghost_ui import _normal_logos_text, _output_item_visible


def test_normal_logos_hides_ctml_controls_but_keeps_reply():
    logos = (
        '<ghost:memory_commit summary="internal" />\n'
        "收到，设备 R-71 的颜色是琥珀色。\n"
        '<ghost:memory_claims _cid="2" />'
    )
    assert _normal_logos_text(logos) == "收到，设备 R-71 的颜色是琥珀色。"


def test_normal_logos_hides_paired_command_payload_and_keeps_safe_wrapper_text():
    logos = (
        '<mcp:exec server="memory">private payload</mcp:exec>\n'
        "<memory_recall_verified>AMBER-731，staging。</memory_recall_verified>"
    )
    assert _normal_logos_text(logos) == "AMBER-731，staging。"


def test_output_modes_make_internal_results_explicit_opt_in():
    assert _output_item_visible("moment", "normal") is False
    assert _output_item_visible("system", "normal") is False
    assert _output_item_visible("command-result", "normal") is False
    assert _output_item_visible("command-output", "normal") is True
    assert _output_item_visible("moment", "verbose") is True
    assert _output_item_visible("command-result", "verbose") is False
    assert _output_item_visible("command-result", "trace") is True

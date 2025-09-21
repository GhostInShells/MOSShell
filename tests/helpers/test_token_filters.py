from ghoshell_moss.helpers.token_filters import SpecialTokenMatcher


def test_special_token_matcher_baseline():
    special_tokens = {
        "$": "<foo />",
        "#": "<bar />",
        "^^^": "<baz />",
    }

    cases = [
        ("<foo>$<bar>^^^#</bar></foo>", "<foo><foo /><bar><baz /><bar /></bar></foo>"),
        ("<foo>$<bar></bar></foo>^^^#", "<foo><foo /><bar></bar></foo><baz /><bar />"),
    ]
    for content, expected in cases:
        matcher = SpecialTokenMatcher(special_tokens)
        result = "".join(list(matcher.parse(content)))
        assert result == expected, expected

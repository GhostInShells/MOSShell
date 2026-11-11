"""The installed bundle carries the fingerprint of the bytes it was copied from.

The failure this guards is not "install is wrong" but "install was never re-run": a stale
bundle looks installed, loads, and silently lacks the newest verbs. The stamp makes the
question answerable by content, without naming any verb.
"""

from __future__ import annotations

from ghoshell_dsh.v0_2_0.install import (
    BUNDLE_STAMP,
    bundle_changed,
    bundle_fingerprint,
    install_bundle,
    install_plugin,
    installed_fingerprint,
    installed_root,
    plugin_changed,
    refresh_plugin,
)


def test_a_fresh_install_records_the_source_fingerprint(tmp_path):
    install_plugin(home=tmp_path)
    assert installed_fingerprint(tmp_path) == bundle_fingerprint()
    assert not plugin_changed(tmp_path)


def test_an_up_to_date_bundle_is_not_copied_again(tmp_path):
    install_plugin(home=tmp_path)
    assert refresh_plugin(home=tmp_path) is False


def test_a_stale_stamp_marks_the_bundle_changed_and_refreshes_it(tmp_path):
    install_plugin(home=tmp_path)
    (installed_root(tmp_path) / BUNDLE_STAMP).write_text(
        '{"fingerprint": "stale"}', encoding="utf-8"
    )
    assert plugin_changed(tmp_path)
    assert refresh_plugin(home=tmp_path) is True
    assert not plugin_changed(tmp_path)


def test_a_transformed_copy_still_reports_the_source_fingerprint(tmp_path):
    """A bundle may be rewritten on the way in (TypeScript -> JavaScript under node_modules).

    The stamp must describe the *source*, otherwise a transformed copy looks stale on every
    start and the plugin is reinstalled forever.
    """
    source = tmp_path / "bundle"
    source.mkdir()
    (source / "package.json").write_text('{"version": "0.0.1"}', encoding="utf-8")
    (source / "index.ts").write_text("export const answer: number = 42\n", encoding="utf-8")

    def transform(target):
        typed = target / "index.ts"
        typed.write_text("export const answer = 42\n", encoding="utf-8")
        typed.rename(target / "index.js")

    home = tmp_path / "home"
    install_bundle(source=source, name="@local/x", home=home, transform=transform)

    assert installed_fingerprint(home, "@local/x") == bundle_fingerprint(source)
    assert not bundle_changed(home, source, "@local/x")
    assert (installed_root(home, "@local/x") / "index.js").is_file()
    assert not (installed_root(home, "@local/x") / "index.ts").exists()

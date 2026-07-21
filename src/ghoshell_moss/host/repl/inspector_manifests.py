from ghoshell_moss.core.blueprint.project import Manifest, MatrixManifest, ModeManifests

__all__ = ['ManifestsInspector']


def _walk(manifests, *, safe: bool = True) -> list[dict]:
    """Walk Iterable[Manifest[T]] into a list of dicts.

    Each dict has at least: name, description, found_at, import_path.
    If safe=True, is_error() entries include an "error" key instead of being skipped.
    """
    rows = []
    for m in manifests:
        row = {
            "name": m.name(),
            "description": m.description() or "",
            "found_at": str(m.found_at()),
        }
        if m.import_path():
            row["import_path"] = m.import_path()
        if m.is_error():
            if safe:
                row["error"] = str(m.error())
                rows.append(row)
            continue
        rows.append(row)
    return rows


def _walk_single(manifest: Manifest, *, safe: bool = True) -> dict | None:
    """Walk a single Manifest[T] into a dict (or None if is_error and not safe)."""
    row = {
        "name": manifest.name(),
        "description": manifest.description() or "",
        "found_at": str(manifest.found_at()),
    }
    if manifest.import_path():
        row["import_path"] = manifest.import_path()
    if manifest.is_error():
        if safe:
            row["error"] = str(manifest.error())
            return row
        return None
    return row


class ManifestsInspector:
    """在 REPL 中观测 Manifest 资源的工具集 — 遍历 MatrixManifest + ModeManifests.

    Matrix 层是 workspace 全局基线 (MOSS.manifests);
    Mode 层是当前 mode 的有效视图 (HOST, 继承并可能覆盖 Matrix 层).
    """

    def __init__(
            self,
            matrix_mf: MatrixManifest,
            mode_mf: ModeManifests | None,
    ):
        self._matrix = matrix_mf
        self._mode = mode_mf

    def explain(self) -> str:
        """两层 manifest 的自描述."""
        text = self._matrix.explain()
        if self._mode is not None:
            text += "\n" + self._mode.explain()
        return text

    # -- 两层方法 (matrix + mode) -- #

    def providers(self) -> dict:
        """IoC Provider 声明."""
        result = {"matrix": [], "mode": None}
        for m in self._matrix.providers():
            row = {
                "contract": m.name(),
                "found_at": str(m.found_at()),
            }
            if m.is_error():
                row["error"] = str(m.error())
                result["matrix"].append(row)
                continue
            v = m.value()
            row["singleton"] = v.singleton()
            row["description"] = m.description() or ""
            result["matrix"].append(row)

        if self._mode is not None:
            result["mode"] = []
            for m in self._mode.providers():
                row = {
                    "contract": m.name(),
                    "found_at": str(m.found_at()),
                }
                if m.is_error():
                    row["error"] = str(m.error())
                    result["mode"].append(row)
                    continue
                v = m.value()
                row["singleton"] = v.singleton()
                row["description"] = m.description() or ""
                result["mode"].append(row)

        return result

    def configs(self) -> dict:
        """配置声明."""
        result = {"matrix": _walk(self._matrix.configs()), "mode": None}
        if self._mode is not None:
            result["mode"] = _walk(self._mode.configs())
        return result

    def topics(self) -> dict:
        """Topic schema 声明."""
        def _topic_rows(manifests):
            rows = []
            for m in manifests:
                row = {"name": m.name(), "found_at": str(m.found_at())}
                if m.is_error():
                    row["error"] = str(m.error())
                    rows.append(row)
                    continue
                v = m.value()
                row["topic_type"] = v.topic_type
                row["description"] = v.description or ""
                rows.append(row)
            return rows

        result = {"matrix": _topic_rows(self._matrix.topics()), "mode": None}
        if self._mode is not None:
            result["mode"] = _topic_rows(self._mode.topics())
        return result

    def signals(self) -> dict:
        """Signal schema 声明."""
        result = {"matrix": _walk(self._matrix.signals()), "mode": None}
        if self._mode is not None:
            result["mode"] = _walk(self._mode.signals())
        return result

    def parameters(self) -> dict:
        """Parameter schema (单值 Manifest)."""
        result = {
            "matrix": _walk_single(self._matrix.parameters()),
            "mode": None,
        }
        if self._mode is not None:
            result["mode"] = _walk_single(self._mode.parameters())
        return result

    def resources(self) -> dict:
        """资源存储声明."""
        def _resource_rows(manifests):
            rows = []
            for m in manifests:
                row = {"name": m.name(), "found_at": str(m.found_at())}
                if m.is_error():
                    row["error"] = str(m.error())
                    rows.append(row)
                    continue
                v = m.value()
                row["scheme"] = v.scheme()
                row["host"] = v.host
                row["description"] = v.description or ""
                rows.append(row)
            return rows

        result = {"matrix": _resource_rows(self._matrix.resources()), "mode": None}
        if self._mode is not None:
            result["mode"] = _resource_rows(self._mode.resources())
        return result

    # -- mode 专属方法 -- #

    def channel(self) -> dict | None:
        """当前 mode 的 __main__ channel (mode 专属)."""
        if self._mode is None:
            return None
        m = self._mode.channel()
        if m.is_error():
            return {"name": m.name(), "error": str(m.error()), "found_at": str(m.found_at())}
        v = m.value()
        return {
            "name": v.name(),
            "type": type(v).__name__,
            "description": v.description() or "",
            "found_at": str(m.found_at()),
        }

    def nuclei(self) -> dict | None:
        """Nucleus 声明 (matrix + mode 两层)."""
        def _nuclei_rows(manifests):
            rows = []
            for m in manifests.nuclei():
                row = {"name": m.name(), "found_at": str(m.found_at())}
                if m.is_error():
                    row["error"] = str(m.error())
                    rows.append(row)
                    continue
                v = m.value()
                row["description"] = v.description() or ""
                row["signal_names"] = [s.signal_name() for s in v.signals()]
                rows.append(row)
            return rows

        result = {"matrix": _nuclei_rows(self._matrix), "mode": None}
        if self._mode is not None:
            result["mode"] = _nuclei_rows(self._mode)
        return result

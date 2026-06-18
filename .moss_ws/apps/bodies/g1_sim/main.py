from __future__ import annotations

import importlib.util
import logging
import os
import platform
import sys
from pathlib import Path

import dotenv
from ghoshell_moss.core.blueprint.matrix import Matrix

from control.mujoco_controller import MujocoVelocityController
from control.obs import SimConfig, load_sim_config
from control.policy import DemoHumanoidPolicy, PolicyRunner, SB3Policy, TorchScriptPolicy, ZeroPolicy
from g1_sim_channel import build_g1_sim_channel


APP_DIR = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)


def _resolve_app_dir(matrix: Matrix | None = None) -> Path:
    if matrix is None:
        return APP_DIR
    if getattr(matrix.this, "type", "") == "app":
        return matrix.cell_workspace.root_path()
    return APP_DIR


def _load_app_env(app_dir: Path) -> None:
    env_file = app_dir / ".env"
    if env_file.exists():
        dotenv.load_dotenv(env_file)


def _has_ready_g1_assets(app_dir: Path) -> bool:
    required = [
        app_dir / "assets" / "g1" / "scene.xml",
        app_dir / "assets" / "g1" / "g1_12dof.xml",
        app_dir / "assets" / "policies" / "g1_motion.pt",
    ]
    meshes_dir = app_dir / "assets" / "g1" / "meshes"
    return all(path.exists() for path in required) and meshes_dir.exists() and any(meshes_dir.glob("*.STL"))


def _resolve_config_path(app_dir: Path) -> Path:
    default_profile = "g1" if _has_ready_g1_assets(app_dir) else "humanoid_v4"
    profile = os.getenv("G1_SIM_PROFILE", default_profile)
    if profile.endswith(".yaml"):
        return Path(profile).expanduser().resolve()
    return app_dir / "config" / f"{profile}.yaml"


def _resolve_asset(path: str, app_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return app_dir / candidate


def _should_bootstrap_mjpython() -> bool:
    app_dir = _resolve_app_dir()
    if platform.system() != "Darwin":
        return False
    if os.getenv("MOSS_G1_SIM_UNDER_MJPYTHON") == "1":
        return False
    try:
        cfg = load_sim_config(_resolve_config_path(app_dir))
    except Exception:
        return False
    return cfg.backend == "mujoco_g1" and cfg.render


def _bootstrap_mjpython() -> None:
    spec = importlib.util.find_spec("mujoco")
    if spec is None or spec.origin is None:
        raise RuntimeError("mujoco package not found in current environment")
    mjpython_bin = (
        Path(spec.origin).resolve().parent
        / "MuJoCo_(mjpython).app"
        / "Contents"
        / "MacOS"
        / "mjpython"
    )
    if not mjpython_bin.exists():
        raise RuntimeError(f"mjpython binary not found: {mjpython_bin}")

    env = os.environ.copy()
    env["MOSS_G1_SIM_UNDER_MJPYTHON"] = "1"
    env["MJPYTHON_BIN"] = str(mjpython_bin)
    real_python = Path(os.path.realpath(sys.executable))
    env["MJPYTHON_LIBPYTHON"] = str(real_python)
    lib_dir = real_python.parent.parent / "lib"
    if lib_dir.exists():
        fallback_paths = [str(lib_dir)]
        existing = env.get("DYLD_FALLBACK_LIBRARY_PATH", "")
        if existing:
            fallback_paths.extend([p for p in existing.split(":") if p])
        else:
            fallback_paths.extend(["/usr/local/lib", "/usr/lib"])
        env["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(dict.fromkeys(fallback_paths))
    os.execve(
        str(mjpython_bin),
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        env,
    )


def _build_policy(cfg: SimConfig, app_dir: Path) -> PolicyRunner:
    kind = cfg.policy.kind.strip().lower()
    path = cfg.policy.path.strip()

    if kind in ("", "zero"):
        return ZeroPolicy(cfg.num_actions)
    if kind in ("demo", "demo_humanoid"):
        return DemoHumanoidPolicy(cfg.num_actions)
    if not path:
        LOGGER.warning("policy kind '%s' configured without path, falling back to ZeroPolicy", kind)
        return ZeroPolicy(cfg.num_actions)

    resolved = _resolve_asset(path, app_dir)
    if not resolved.exists():
        LOGGER.warning("policy asset not found at %s, falling back to ZeroPolicy", resolved)
        return ZeroPolicy(cfg.num_actions)

    if kind in ("sb3", "stable_baselines3"):
        return SB3Policy(resolved)
    if kind in ("torchscript", "torch", "jit"):
        return TorchScriptPolicy(resolved)
    raise ValueError(f"unsupported policy kind: {cfg.policy.kind}")


async def main(matrix: Matrix):
    app_dir = _resolve_app_dir(matrix)
    _load_app_env(app_dir)
    cfg_path = _resolve_config_path(app_dir)
    cfg = load_sim_config(cfg_path)
    policy = _build_policy(cfg, app_dir)
    controller = MujocoVelocityController(cfg, policy)
    controller.start()
    channel = build_g1_sim_channel(controller)
    try:
        await matrix.provide_channel(channel)
    finally:
        controller.close()


if __name__ == "__main__":
    _load_app_env(_resolve_app_dir())
    if _should_bootstrap_mjpython():
        _bootstrap_mjpython()
    matrix = Matrix.discover()
    matrix.run(main)

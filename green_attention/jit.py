"""Runtime nvcc -> shared library -> ``tvm_ffi.load_module``.

Each kernel module is one ``.cu`` translation unit that exports its entry points
with ``TVM_FFI_DLL_EXPORT_TYPED_FUNC``. Builds are cached on disk under
``$GREEN_ATTN_CACHE_DIR`` (default ``~/.cache/green_attention``), keyed by the
hash of every source/header byte, the flags, the template defines, the nvcc
version and the GPU arch, so a stale ``.so`` is never loaded. Builds are
serialized per key by a thread lock plus an ``fcntl`` file lock, so the TP ranks
and the scheduler subprocesses share one compile.

Compile everything before CUDA graph capture (``prewarm``): a JIT compile
during capture or in steady state is a bug.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import subprocess
import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Mapping, Sequence

logger = logging.getLogger(__name__)

PACKAGE = Path(__file__).resolve().parent
CSRC = PACKAGE / "csrc"
REPO = PACKAGE.parent

_MODULES: dict = {}
_LOCKS: dict = {}
_LOCKS_GUARD = threading.Lock()


def cuda_home() -> Path:
    return Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.0"))


def cutlass_include() -> Path:
    """CuTe/CUTLASS headers: ``$GREEN_CUTLASS_DIR`` or the pinned submodule."""
    root = Path(os.environ.get("GREEN_CUTLASS_DIR", REPO / "third_party" / "cutlass"))
    include = root / "include"
    if not (include / "cute" / "tensor.hpp").is_file():
        raise FileNotFoundError(
            f"CUTLASS headers not found under {include}; run "
            "`git submodule update --init third_party/cutlass` or set GREEN_CUTLASS_DIR"
        )
    return include


@lru_cache(maxsize=None)
def _tvm_ffi_paths() -> tuple:
    from tvm_ffi.libinfo import (
        find_dlpack_include_path,
        find_include_path,
        find_libtvm_ffi,
    )

    lib = Path(find_libtvm_ffi())
    includes = tuple(dict.fromkeys([find_include_path(), find_dlpack_include_path()]))
    return includes, str(lib.parent), lib.stem.removeprefix("lib")


@lru_cache(maxsize=None)
def _nvcc_version() -> str:
    out = subprocess.run(
        [str(cuda_home() / "bin" / "nvcc"), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def cache_root() -> Path:
    configured = os.environ.get("GREEN_ATTN_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "green_attention"


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _thread_lock(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def _flags(arch: str, defines: Mapping[str, object]) -> list:
    includes, lib_dir, lib_name = _tvm_ffi_paths()
    return [
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-lineinfo",
        "-Xcompiler",
        "-fPIC",
        "-DNDEBUG",
        f"-I{CSRC}",
        f"-I{cutlass_include()}",
        *(f"-I{p}" for p in includes),
        *(f"-D{k}={v}" for k, v in sorted(defines.items())),
        "-shared",
        f"-L{lib_dir}",
        f"-l{lib_name}",
        f"-L{cuda_home() / 'lib64'}",
        "-lcudart",
    ]


def _identity(name: str, source: Path, headers: Sequence[Path], flags: list) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({"name": name, "flags": flags, "nvcc": _nvcc_version()}).encode())
    for path in (source, *headers):
        digest.update(str(path.relative_to(CSRC)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def load(
    name: str,
    source: str,
    *,
    defines: Mapping[str, object] | None = None,
    arch: str = "90a",
):
    """Compile (once, cached) and load ``csrc/<source>`` as a tvm_ffi module.

    ``defines`` are the template parameters baked into this instance; they are
    part of the cache key. Returns the loaded ``tvm_ffi.Module``; its exported
    functions are attributes (``module.fn(...)``).
    """
    import tvm_ffi

    defines = dict(defines or {})
    src = CSRC / source
    headers = sorted(p for p in CSRC.rglob("*") if p.suffix in (".cuh", ".h", ".hpp"))
    flags = _flags(arch, defines)
    key = f"{name}_{_identity(name, src, headers, flags)}"

    module = _MODULES.get(key)
    if module is not None:
        return module
    with _thread_lock(key):
        module = _MODULES.get(key)
        if module is not None:
            return module
        out_dir = cache_root() / key
        so = out_dir / f"{name}.so"
        with _file_lock(cache_root() / f".{key}.lock"):
            if not so.is_file():
                out_dir.mkdir(parents=True, exist_ok=True)
                tmp = out_dir / f".{name}.{os.getpid()}.so"
                cmd = [str(cuda_home() / "bin" / "nvcc"), str(src), "-o", str(tmp), *flags]
                logger.info("green_attention: compiling %s (%s)", name, key)
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"nvcc failed for {name}:\n{' '.join(cmd)}\n{result.stderr}"
                    )
                os.replace(tmp, so)
        module = tvm_ffi.load_module(str(so))
        _MODULES[key] = module
        return module

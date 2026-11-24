"""What a container uses, read from its cgroup (version 2) and from its process's status.

Inside a container, ``/sys/fs/cgroup`` is the container's own cgroup. Its memory
counts the process's own memory (``anon``) and the page cache of the files it
reads (``file``); the kernel takes back cache when the limit is reached, and
kills the process when its own memory alone exceeds it. On a system without
these files, as on Windows, every field is ``None``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

CGROUP_ROOT: Path = Path("/sys/fs/cgroup")
PROC_STATUS: Path = Path("/proc/self/status")


class Usage(BaseModel):
    """One reading of what the container and its process use.

    Attributes:
        memory_current: Memory charged to the cgroup now, in bytes.
        memory_peak: The most memory ever charged to it, in bytes.
        anon: The part that is the processes' own memory, in bytes.
        file: The part that is page cache, in bytes.
        refaults: Pages of files read again after the kernel had evicted them.
        major_faults: Page faults that had to read from disk.
        oom_kills: Processes the kernel killed for running out of memory.
        cpu_usage_us: CPU time used, in microseconds.
        throttled: CPU periods in which the quota ran out.
        throttled_us: Time spent waiting for the next period, in microseconds.
        rss: The process's resident memory, in bytes.
        rss_peak: Its highest resident memory, in bytes.
        rss_anon: The part of it that is its own memory, in bytes.
        rss_file: The part that is mapped files, in bytes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_current: int | None = None
    memory_peak: int | None = None
    anon: int | None = None
    file: int | None = None
    refaults: int | None = None
    major_faults: int | None = None
    oom_kills: int | None = None
    cpu_usage_us: int | None = None
    throttled: int | None = None
    throttled_us: int | None = None
    rss: int | None = None
    rss_peak: int | None = None
    rss_anon: int | None = None
    rss_file: int | None = None


def parse_flat_keyed(text: str) -> dict[str, int]:
    """Parse lines of ``key value``, as ``memory.stat`` and ``cpu.stat`` hold them."""
    values: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            values[parts[0]] = int(parts[1])
    return values


def parse_status(text: str) -> dict[str, int]:
    """Parse the memory lines of ``/proc/<pid>/status``, such as ``VmRSS:  1234 kB``, into bytes."""
    values: dict[str, int] = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        parts = rest.split()
        if len(parts) == 2 and parts[1] == "kB" and parts[0].isdigit():
            values[name] = int(parts[0]) * 1024
    return values


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _single(path: Path) -> int | None:
    text = _read(path)
    if text is None or not text.strip().isdigit():
        return None
    return int(text.strip())


def read_usage(root: Path = CGROUP_ROOT, status: Path = PROC_STATUS) -> Usage:
    """Read the cgroup's memory and CPU counters and the process's resident memory.

    Args:
        root: The cgroup's directory.
        status: The process's status file.

    Returns:
        The reading; a counter that cannot be read is ``None``.
    """
    memory = parse_flat_keyed(_read(root / "memory.stat") or "")
    events = parse_flat_keyed(_read(root / "memory.events") or "")
    cpu = parse_flat_keyed(_read(root / "cpu.stat") or "")
    process = parse_status(_read(status) or "")
    return Usage(
        memory_current=_single(root / "memory.current"),
        memory_peak=_single(root / "memory.peak"),
        anon=memory.get("anon"),
        file=memory.get("file"),
        refaults=memory.get("workingset_refault_file"),
        major_faults=memory.get("pgmajfault"),
        oom_kills=events.get("oom_kill"),
        cpu_usage_us=cpu.get("usage_usec"),
        throttled=cpu.get("nr_throttled"),
        throttled_us=cpu.get("throttled_usec"),
        rss=process.get("VmRSS"),
        rss_peak=process.get("VmHWM"),
        rss_anon=process.get("RssAnon"),
        rss_file=process.get("RssFile"),
    )

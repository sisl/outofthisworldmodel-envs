"""Process entry point for the `owm-envs` console script.

Which GPU JAX runs on, and how much of that card XLA claims, are read from the
environment when JAX's backend initialises -- and that happens during import,
inside astrojax, well before any command body runs. So the choice cannot be
made where the flag is parsed: by then the backend is up and the settings are
inert. It is made here instead, ahead of the CLI module and the transitive
astrojax import it carries, which is why `cli` is imported inside `main`
rather than at the top of this file.

Importing `_jax_config` here is safe in a way importing `cli` would not be:
jax by itself initialises no backend, so the settings still land before
anything reads them.

Only `--gpu-index` is read here, and only well enough to find its value.
Every other question about the command line, including whether a malformed
index should be rejected, stays Typer's: a value that will not parse is left
alone so the parser reports it as the usage error it is, rather than being
half-handled twice.
"""

from __future__ import annotations

import sys

from ._jax_config import pin_gpu

FLAG = "--gpu-index"


def requested_index(argv: list[str]) -> int | None:
    """The `--gpu-index` this command line asks for, if it carries a usable one.

    The last occurrence wins, which is the value Typer itself would take, so a
    repeated flag pins the card the run actually renders on.
    """
    found: int | None = None
    for position, arg in enumerate(argv):
        if arg == "--":
            # Everything past it is a positional to Typer, whatever it looks
            # like, so a flag spelled after it names no GPU.
            break
        if arg == FLAG and position + 1 < len(argv):
            raw = argv[position + 1]
        elif arg.startswith(f"{FLAG}="):
            raw = arg.split("=", 1)[1]
        else:
            continue
        try:
            found = int(raw)
        except ValueError:
            found = None
    return found


def main() -> None:
    from .render.device import resolve_gpu_index

    try:
        index = resolve_gpu_index(requested_index(sys.argv[1:]))
    except ValueError as exc:
        # Typer is not up yet and cannot report this as the usage error it is,
        # so it is reported the way Typer would rather than as a traceback out
        # of an entry point the operator never called directly.
        sys.exit(f"Error: {exc}")

    pin_gpu(index)

    from .cli import app

    app()

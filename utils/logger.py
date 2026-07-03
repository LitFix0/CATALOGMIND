"""
CatalogMind — Centralized logging configuration.

Design decision: app.py currently calls setup_logging() directly.
This module exists so other entrypoints (scripts, tests, ingestion
pipelines) can get the same consistent format without duplicating
the basicConfig call or risking it being invoked twice with different
settings (which silently no-ops after the first call in stdlib logging).
"""

import logging

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure root logging once, idempotently.

    Safe to call from multiple entrypoints (app.py, ingestion scripts,
    tests) — subsequent calls after the first are no-ops.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    _CONFIGURED = True
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference times for Sphinx gallery and runnable examples.

Examples use *recent* default times so plots stay visually meaningful. Operational
and reanalysis products often lag real time; defaults step back from UTC *now*.

Environment
-----------
EARTH2STUDIO_EXAMPLE_ANCHOR_DATE
    If set, use this instant instead of a sliding window. Accepts ``YYYY-MM-DD``
    or full ISO datetime (UTC). Overrides ``days_back``.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone


def gallery_reference_datetime_utc(
    *,
    days_back: int = 42,
    hour: int = 12,
    minute: int = 0,
) -> datetime:
    """Return a naive UTC datetime suitable as a forecast/analysis reference time.

    By default, uses ``days_back`` before current UTC time (rounded to ``hour`` /
    ``minute``). If ``EARTH2STUDIO_EXAMPLE_ANCHOR_DATE`` is set, that value wins.

    Parameters
    ----------
    days_back : int, optional
        Days to subtract from UTC now when no anchor env is set, by default 42.
    hour : int, optional
        Hour in UTC for the computed reference day, by default 12.
    minute : int, optional
        Minute in UTC, by default 0.

    Returns
    -------
    datetime
        Naive datetime interpreted as UTC (matches common Earth2Studio examples).
    """
    raw = os.environ.get("EARTH2STUDIO_EXAMPLE_ANCHOR_DATE", "").strip()
    if raw:
        try:
            if "T" in raw or raw.endswith("Z") or raw.count("-") >= 3 and len(raw) > 10:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                if dt.tzinfo is not None:
                    dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                return dt
            parts = raw.split("-")
            if len(parts) >= 3:
                y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
                return datetime(y, m, d, hour, minute, 0)
        except ValueError:
            pass

    now = datetime.now(timezone.utc)
    base = now - timedelta(days=days_back)
    out = base.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return out.replace(tzinfo=None)

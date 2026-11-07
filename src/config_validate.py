"""Merge user/LLM JSON with baseline effective config."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

ALLOWED_HMD_COUNTRIES = {
    "AUS", "AUT", "BLR", "BEL", "BGR", "CAN", "CHL", "HRV", "CZE", "DNK", "EST",
    "FIN", "FRATNP", "DEUTNP", "GRC", "HKG", "HUN", "ISL", "IRL", "ISR", "ITA",
    "JPN", "LVA", "LTU", "LUX", "NLD", "NZL_NP", "NOR", "POL", "PRT", "KOR", "RUS",
    "SVK", "SVN", "ESP", "SWE", "CHE", "TWN", "GBR_NP", "USA", "UKR", "GBRTENW",
    "GBRCENW", "GBR_SCO", "GBR_NIR", "NZL_MA", "NZL_NM", "DEUTE", "DEUTW",
}

RUNTIME_CONFIG_KEYS = {
    "_comment",
    "pipeline_id",
    "lock_to_baseline",
    "hmd",
    "ages",
    "fit_period",
    "forecast_max_year",
    "rolling_origins",
    "mse_horizon_max",
    "mcs_seed",
    "mcs_B",
    "mcs_block_l",
    "mcs_alpha",
    "models",
    "stack_metalearners",
    "run_mode",
    "forecast_horizon",
    "forecast_target_year",
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_user_request_to_effective(
    base: Dict[str, Any], user_req: Dict[str, Any]
) -> Dict[str, Any]:
    out = deepcopy(base)
    if "hmd" not in out or not isinstance(out.get("hmd"), dict):
        out["hmd"] = {}
    if isinstance(user_req.get("country"), str) and user_req["country"].strip():
        out["hmd"]["country"] = user_req["country"].strip()
    if isinstance(user_req.get("gender"), str) and user_req["gender"].strip():
        out["hmd"]["sex"] = user_req["gender"].strip()

    for key in ("ages", "fit_period", "models", "stack_metalearners"):
        if key in user_req and user_req[key] is not None:
            out[key] = deepcopy(user_req[key])

    test_period = user_req.get("test_period")
    if isinstance(test_period, dict):
        start = test_period.get("start")
        end = test_period.get("end")
        if start is not None and end is not None:
            out["rolling_origins"] = {"start": start, "end": end}
    return out


def apply_forecast_run_mode(
    effective: Dict[str, Any],
    user_req: Dict[str, Any],
    *,
    hmd_max_year: Optional[int] = None,
) -> Optional[str]:
    if user_req.get("status") == "need_clarification":
        return None

    raw_target = user_req.get("forecast_target_year")
    target: Optional[int]
    if raw_target is None or raw_target == "":
        target = None
    else:
        try:
            target = int(raw_target)
        except (TypeError, ValueError):
            return "forecast_target_year must be an integer calendar year."

    if target is None:
        effective["run_mode"] = "evaluation"
        effective.pop("forecast_horizon", None)
        effective.pop("forecast_target_year", None)
        return None

    if hmd_max_year is not None and target <= int(hmd_max_year):
        return (
            f"forecast_target_year {target} must be after last HMD year ({hmd_max_year})."
        )

    effective["run_mode"] = "both"
    effective["forecast_target_year"] = target
    effective.pop("forecast_horizon", None)
    return None


def keep_runtime_keys_only(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {k: deepcopy(v) for k, v in cfg.items() if k in RUNTIME_CONFIG_KEYS}


def build_effective_config(
    baseline: Dict[str, Any],
    user_req: Dict[str, Any],
    *,
    hmd_max_year: Optional[int] = None,
) -> Dict[str, Any]:
    """Return effective runtime config or raise ValueError."""
    locked = bool(baseline.get("lock_to_baseline", True))
    if locked:
        effective = deepcopy(baseline)
    else:
        effective = apply_user_request_to_effective(baseline, user_req)
    effective = keep_runtime_keys_only(effective)

    if not locked:
        err = apply_forecast_run_mode(
            effective, user_req, hmd_max_year=hmd_max_year
        )
        if err:
            raise ValueError(err)

    country = str(effective.get("hmd", {}).get("country", "")).strip()
    if country and country not in ALLOWED_HMD_COUNTRIES:
        raise ValueError(
            f"country '{country}' is not in the HMD supported list. "
            f"Allowed: {', '.join(sorted(ALLOWED_HMD_COUNTRIES))}"
        )
    return effective

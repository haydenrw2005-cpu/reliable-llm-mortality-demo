"""Stricter validation of LLM/user request JSON before R is launched."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .config_validate import ALLOWED_HMD_COUNTRIES

ALLOWED_MODELS = {"LC", "RH", "APC", "CBD", "M7", "PLAT"}
ALLOWED_GENDERS = {"male", "female"}


def validate_user_request(user_req: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """
    Return (ok, issues, clarifying_questions).
    If status is need_clarification, ok=False and questions come from the model.
    """
    issues: List[str] = []
    questions: List[str] = []

    if user_req.get("status") == "need_clarification":
        missing = user_req.get("missing") or []
        qs = user_req.get("questions") or []
        if isinstance(qs, list):
            questions.extend(str(q) for q in qs)
        if isinstance(missing, list) and missing:
            issues.append("Missing: " + ", ".join(str(m) for m in missing))
        if not questions:
            questions.append("Please provide country, sex, ages, fit/test years, and models.")
        return False, issues, questions

    country = str(user_req.get("country") or "").strip()
    if not country:
        issues.append("country is required.")
        questions.append("Which HMD country/population should we use (e.g. GBRTENW, USA, SWE)?")
    elif country not in ALLOWED_HMD_COUNTRIES:
        issues.append(f"country '{country}' is not in the HMD supported list.")
        questions.append("Please choose a supported HMD country code.")

    gender = str(user_req.get("gender") or "").strip().lower()
    if not gender:
        issues.append("gender is required.")
        questions.append("Should the analysis use male or female mortality?")
    elif gender not in ALLOWED_GENDERS:
        issues.append("gender must be 'male' or 'female'.")
        questions.append("Please specify male or female.")

    ages = user_req.get("ages")
    if isinstance(ages, dict):
        try:
            amin = int(ages.get("min"))
            amax = int(ages.get("max"))
            if amin > amax:
                issues.append("ages.min must be <= ages.max.")
            if amin < 0 or amax > 120:
                issues.append("ages look out of range (expected roughly 0–120).")
        except (TypeError, ValueError):
            issues.append("ages.min and ages.max must be integers.")

    for key, label in (("fit_period", "fit"), ("test_period", "test")):
        period = user_req.get(key)
        if not isinstance(period, dict):
            continue
        try:
            start = int(period.get("start"))
            end = int(period.get("end"))
            if start > end:
                issues.append(f"{label} period start must be <= end.")
            if start < 1800 or end > 2100:
                issues.append(f"{label} period years look invalid.")
        except (TypeError, ValueError):
            issues.append(f"{key} start/end must be integers.")

    fit = user_req.get("fit_period") if isinstance(user_req.get("fit_period"), dict) else None
    test = user_req.get("test_period") if isinstance(user_req.get("test_period"), dict) else None
    if fit and test:
        try:
            if int(test["start"]) <= int(fit["end"]):
                # Allow adjacent/overlapping origins; only hard-fail if test ends before fit starts.
                pass
            if int(test["end"]) < int(fit["start"]):
                issues.append("test_period ends before fit_period starts.")
        except (TypeError, ValueError, KeyError):
            pass

    models = user_req.get("models")
    if models is not None:
        if not isinstance(models, list) or not models:
            issues.append("models must be a non-empty list when provided.")
        else:
            bad = [str(m) for m in models if str(m).upper() not in ALLOWED_MODELS]
            if bad:
                issues.append(
                    "Unknown models: "
                    + ", ".join(bad)
                    + f". Allowed: {', '.join(sorted(ALLOWED_MODELS))}."
                )

    fty = user_req.get("forecast_target_year")
    if fty not in (None, ""):
        try:
            y = int(fty)
            if y < 2000 or y > 2100:
                issues.append("forecast_target_year looks implausible.")
        except (TypeError, ValueError):
            issues.append("forecast_target_year must be an integer calendar year.")

    if issues and not questions:
        questions.append(
            "Please revise the request so country, sex, ages, periods, and models are valid."
        )
    return len(issues) == 0, issues, questions

"""Normalize nuclear refuelling and fuel-cycle configuration."""

def refueling_settings(source_input):
    """Return normalized refuelling settings for one source."""

    if not isinstance(source_input, dict):
        return {"enabled": False}
    raw = source_input.get("refueling")
    if not isinstance(raw, dict) or not raw:
        return {"enabled": False}

    operating_cycle = raw.get("operating_cycle")
    target_burnup = (source_input.get("fuel_cycle", {}) or {}).get(
        "target_burnup")
    basis = "calendar" if operating_cycle is not None else "burnup"
    residence = None
    if basis == "burnup":
        residence = _residence_efpd_from_burnup(source_input)

    fuel_batches = int(raw.get("fuel_batches", 0) or 0)
    return {
        "enabled": True,
        "mode": str(raw.get("mode", "offline")).strip().lower(),
        "unit_capacity": float(source_input.get("unit_capacity", 0.0)),
        "outage_duration": int(raw.get("outage_duration", 0) or 0),
        "operating_cycle": (
            float(operating_cycle)
            if operating_cycle is not None else None),
        "residence_efpd": (
            float(residence) if residence is not None else None),
        "fuel_batches": fuel_batches,
        "basis": basis,
        "schedule": str(raw.get("schedule", "auto")).strip().lower(),}


def _residence_efpd_from_burnup(source_input):
    """Derive residence EFPD from target burnup and core properties."""

    fuel_cycle = source_input.get("fuel_cycle", {}) or {}
    burnup = fuel_cycle.get("target_burnup")
    thermal = fuel_cycle.get("thermal_power")
    mass = fuel_cycle.get("core_fuel_mass")
    if burnup is None or thermal is None or mass is None:
        return None
    burnup = float(burnup)
    thermal = float(thermal)
    mass = float(mass)
    if burnup <= 0.0 or thermal <= 0.0 or mass <= 0.0:
        return None
    return burnup * 1000.0 * mass / thermal


def is_dynamic_efpd_refueling(source_input):
    """Return whether a source needs operational EFPD refuelling."""

    settings = refueling_settings(source_input)
    return bool(
        settings.get("enabled")
        and settings.get("mode") == "offline"
        and settings.get("basis") == "burnup")


def is_online_efpd_refueling(source_input):
    """Return whether a source tracks EFPD with online refuelling."""

    settings = refueling_settings(source_input)
    return bool(
        settings.get("enabled")
        and settings.get("mode") == "online"
        and settings.get("basis") == "burnup")

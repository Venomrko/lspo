def estimate_updates_from_paper(hours: float = 372.9, steps: int = 15_000) -> float:
    return hours * 3600.0 / steps

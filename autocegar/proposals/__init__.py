"""Versioned Auto-CEGAR proposal registry.

One entry per candidate architecture from
``rw_cegar_research_Proposal_1_to_Proposal_5.docx``. Each proposal is a SEPARATE,
self-contained class file (``proposalN.py``) so the versions never bleed into each
other. The plan (week-8 meeting) is to implement + fail-fast test them in order of
rising complexity, ranking on the selected datasets before committing to a full
run.

To add a proposal N:
    1. write ``autocegar/proposals/proposalN.py`` with a ``CNN_RW_CEGAR_PN`` class
       (subclass ``rw.cnn_rw.CNN_RW``, expose ``PROPOSAL`` / ``NAME``);
    2. import it below and fill its ``PROPOSALS[N]`` entry (cls + variant kwargs);
    3. it is then runnable via ``python run_proposal.py --proposal N``.

``variants`` are named preset kwarg overrides (e.g. Proposal 1's 'basic' vs
'selective' confidence). ``cls=None`` marks a proposal that is specified in the
doc but not yet implemented.
"""
from .proposal1 import CNN_RW_CEGAR_P1

PROPOSALS = {
    1: {
        "name": "Residual-Gated RW-CEGAR",
        "cls": CNN_RW_CEGAR_P1,
        "summary": "Robust-z residual wrongness x (basic|tail-quantile) confidence, "
                   "gate amplifies the forecasting-loss gradient. Simplest hybrid.",
        "variants": {
            "basic":     {"conf_mode": "basic"},      # C_t = 1 (always confident)
            "selective": {"conf_mode": "quantile"},   # C_t = residual-tail confidence
        },
        "default_variant": "basic",
    },
    2: {
        "name": "Uncertainty-Aware Residual CEGAR",
        "cls": None,  # TODO: MC-dropout / ensemble confidence (Proposal 2)
        "summary": "Confidence = inverse predictive uncertainty (MC-dropout/ensemble).",
        "variants": {},
        "default_variant": None,
    },
    3: {
        "name": "RW-Correction-Consistency CEGAR",
        "cls": None,  # TODO: correction magnitude + persistence gate (Proposal 3)
        "summary": "Gate from RW correction magnitude x direction-stability over epochs.",
        "variants": {},
        "default_variant": None,
    },
    4: {
        "name": "Dual-Gate Residual-and-Gradient RW-CEGAR",
        "cls": None,  # TODO: residual gate x input-gradient correctability (Proposal 4)
        "summary": "Gate = high residual AND high input-gradient reducibility.",
        "variants": {},
        "default_variant": None,
    },
    5: {
        "name": "Temporal-Persistence Confident-Error CEGAR",
        "cls": None,  # TODO: persistence-smoothed residual gate (Proposal 5)
        "summary": "Gate only residual regions that persist over neighbouring windows.",
        "variants": {},
        "default_variant": None,
    },
}


def get_proposal(n: int):
    """Return the registry entry for proposal ``n`` or raise a clear error."""
    if n not in PROPOSALS:
        raise ValueError(f"Unknown proposal {n}. Available: {sorted(PROPOSALS)}")
    return PROPOSALS[n]


def build_model(n: int, variant=None, **kwargs):
    """Instantiate proposal ``n``'s model with variant presets + explicit kwargs.

    Explicit ``kwargs`` win over the variant presets so the runner can still
    override lam/tau/k/epochs on the command line.
    """
    entry = get_proposal(n)
    cls = entry["cls"]
    if cls is None:
        raise NotImplementedError(
            f"Proposal {n} ({entry['name']}) is not implemented yet. "
            f"Implemented: {[k for k, v in PROPOSALS.items() if v['cls'] is not None]}")
    variant = variant or entry.get("default_variant")
    preset = entry["variants"].get(variant, {}) if variant else {}
    merged = {**preset, **kwargs}
    model = cls(**merged)
    model.VARIANT = variant
    return model


__all__ = ["PROPOSALS", "get_proposal", "build_model", "CNN_RW_CEGAR_P1"]

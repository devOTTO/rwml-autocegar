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
from .proposal2 import CNN_RW_CEGAR_P2
from .proposal3 import CNN_RW_CEGAR_P3
from .proposal4 import CNN_RW_CEGAR_P4
from .proposal5 import CNN_RW_CEGAR_P5

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
        "cls": CNN_RW_CEGAR_P2,
        "summary": "Confidence = inverse predictive uncertainty via MC-dropout; "
                   "robust-z residual wrongness. Cleanest CEGAR analogue.",
        "variants": {
            "mc5":  {"mc_samples": 5},    # 5 MC-dropout passes (default)
            "mc10": {"mc_samples": 10},   # more passes = steadier uncertainty, slower
        },
        "default_variant": "mc5",
    },
    3: {
        "name": "RW-Correction-Consistency CEGAR",
        "cls": CNN_RW_CEGAR_P3,
        "summary": "Gate = correction magnitude x direction-stability over epochs; drives "
                   "gradient amplification AND a preserve (suppress) write-back (keep the "
                   "correction as evidence, not erase). Full docx spec.",
        "variants": {
            "full":          {"gamma": 0.9, "amplify": True},   # docx: amp + preserve write-back
            "preserve_only": {"gamma": 0.9, "amplify": False},  # ablation: write-back only (lam=0)
            "soft":          {"gamma": 0.5, "amplify": True},   # gentler preserve
        },
        "default_variant": "full",
    },
    4: {
        "name": "Dual-Gate Residual-and-Gradient RW-CEGAR",
        "cls": CNN_RW_CEGAR_P4,
        "summary": "Gate = high residual AND high input-gradient reducibility "
                   "(correctability). Extra fwd+bwd w.r.t. the input per batch.",
        "variants": {
            "gradnorm": {"use_benefit": False},   # g_grad from ||d loss/d input||
            "benefit":  {"use_benefit": True},    # g_grad from estimated loss reduction
        },
        "default_variant": "gradnorm",
    },
    5: {
        "name": "Temporal-Persistence Confident-Error CEGAR",
        "cls": CNN_RW_CEGAR_P5,
        "summary": "Gate = residual x temporal persistence (moving-average of the "
                   "residual indicator over +/-h neighbouring windows).",
        "variants": {
            "h5":  {"persist_h": 5},    # persistence window +/-5 (11 steps)
            "h25": {"persist_h": 25},   # wider +/-25 (51 steps)
        },
        "default_variant": "h5",
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


__all__ = ["PROPOSALS", "get_proposal", "build_model",
           "CNN_RW_CEGAR_P1", "CNN_RW_CEGAR_P2", "CNN_RW_CEGAR_P3"]

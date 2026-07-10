# Backup — OLD-config experiment docs (superseded)

These are the P1/P2/P3 results written under the **old configuration**:
- warm-up = **forecaster-only** (no correction during warm-up)
- `correction_init = 'zero'`

That config was later found to be a confound (it deviated from RW-1's neg_x /
no-forecaster-only setup and depressed the proposals). The re-run under the
corrected config (warm-up = plain RW-1 then gate; `correction_init = 'neg_x'`)
supersedes these. Kept for provenance only; see the current
`experiments/proposals/proposalN/proposalN_results.md` for the live results.

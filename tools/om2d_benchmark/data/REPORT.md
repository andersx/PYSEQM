# S/Cl d-orbital small-set benchmark

Closed-shell set: **H2S, SO2, Cl2, SCl2, SF2, H2**.  
Reaction: **H2S + Cl2 → SCl2 + H2**.  
Geometry targets: experiment. Reaction target: B3LYP/def2-SVP (ΔE = **0.310 eV**).

## Does d help? (MNDO vs MNDO/d)

| Method | bond MAE (Å) | angle MAE (°) | reaction ΔE (eV) | ΔE − DFT |
|---|---:|---:|---:|---:|
| MNDO | 0.036 | 5.91 | −0.706 | −1.016 |
| **MNDO_d** | **0.035** | **4.17** | **−0.231** | **−0.540** |

Headline geometry: **SO2 angle** error **12.5° → 2.4°** with d orbitals.

| Molecule | MNDO angle / bond | MNDO/d angle / bond | Exp |
|---|---|---|---|
| SO2 | 106.8° / 1.476 Å | **116.9°** / 1.491 Å | 119.3° / 1.431 Å |
| H2S | 98.0° / 1.299 Å | 99.8° / **1.334 Å** | 92.1° / 1.336 Å |
| SCl2 | 106.3° / 1.971 Å | 105.9° / 1.993 Å | 102.7° / 2.011 Å |
| SF2 | 99.6° / 1.572 Å | 101.4° / 1.618 Å | 98.0° / 1.589 Å |
| Cl2 | — / 1.996 Å | — / 1.936 Å | — / 1.988 Å |

Reaction energy vs DFT also improves with d (error 1.02 → 0.54 eV).

## OM2+d status

- CSV carries **Thiel MNDO/d** `U_dd`, `ζ_d`, `β_d` for S/Cl as **initial** OM2+d seeds (literature transfer — not re-derived from DFT here).
- Opt-in flag: `seqm_parameters["om2_use_d"]=True`.
- **Default is off.** The current 9-AO OM2 path (without OM2 two-electron d integrals) is not geometry-safe (e.g. linear H2S, collapsed Cl2). Fitting d-params against optimized geometries is blocked until that integral stack exists.
- Existing OM2 **sp** rows for S/Cl are incomplete (multipoles/GTO terms missing) and are not production-quality; Cl2 OM2(sp) already collapses.

## Initial OM2+d d-parameters (adopted)

```json
{
  "U_dd":  {"16": -25.095118, "17": -36.6745732},
  "zeta_d": {"16": 1.2314725, "17": 1.25139777},
  "beta_d": {"16": -1.8806695, "17": -1.87778198}
}
```

Source: Thiel MNDO/d (same values already used by PYSEQM `MNDO_d`). Ort/ECP d-factors remain 0.

## How to reproduce

```bash
# DFT refs + full compare (slow)
python tools/om2d_benchmark/run_all.py --skip-fit

# Compare only (reuse tools/om2d_benchmark/data/dft_refs.json)
python tools/om2d_benchmark/run_all.py --compare-only
```

Artifacts: `tools/om2d_benchmark/data/compare_final.json`, `dft_refs.json`, `testset.json`.

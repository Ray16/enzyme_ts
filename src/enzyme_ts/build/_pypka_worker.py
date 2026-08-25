#!/usr/bin/env python
"""pypKa protonation determinant.

pypKa runs a Poisson-Boltzmann/Monte-Carlo pKa calculation and writes an Amber-named,
correctly protonated PDB at the target pH (HID/HIE/HIP, ASH/GLH, LYN, tautomers
resolved) -- replacing tleap's blind pH-7 defaults, which mis-assign buried Asp/Glu
and active-site His. Mechanism-dictated catalytic states are forced via `overrides`.

`titrate()` is import-safe (pypKa is imported lazily inside it), so it runs IN-PROCESS
in the single `ts_finder` env. The same file also runs as a standalone CLI, used only
as a split-env subprocess fallback (an env where pypKa is not importable in-process).
"""
from __future__ import annotations
import os, json


def titrate(pdb, out_pdb, pH=7.0, epsin=15.0, ncpus=8, overrides=None):
    """PB/MC pKa -> Amber-protonated PDB at pH. Returns the states dict; also writes
    <out_pdb>.states.json. `overrides` = {resid: RESNAME} forced after titration."""
    from pypka import Titration
    pdb, out_pdb = os.path.abspath(pdb), os.path.abspath(out_pdb)
    tit = Titration({
        "structure": pdb, "pH": "0,14", "pHstep": 0.25, "epsin": epsin,
        "ionicstr": 0.1, "ncpus": ncpus, "sites_A": "all", "clean_pdb": True,
        "structure_output": (out_pdb, pH, "amber"),
    })
    rows, flags = [], []
    for site in tit:
        resn, resi = site.res_name, site.res_number
        try:
            pk = site.pK
        except Exception:
            pk = None
        rec = {"res": resn, "resi": resi, "pKa": (round(pk, 2) if pk is not None else None)}
        if pk is not None:
            if resn in ("ASP", "GLU") and pk > pH:
                rec["amber"] = {"ASP": "ASH", "GLU": "GLH"}[resn]
                flags.append(f"{resn}{resi}: pKa {pk:.1f} > pH {pH} -> NEUTRAL {rec['amber']}")
            elif resn == "HIS" and pk > pH:
                rec["amber"] = "HIP"
                flags.append(f"HIS{resi}: pKa {pk:.1f} > pH {pH} -> HIP")
            elif resn == "LYS" and pk < pH:
                rec["amber"] = "LYN"
                flags.append(f"LYS{resi}: pKa {pk:.1f} < pH {pH} -> NEUTRAL LYN")
            elif resn in ("TYR", "CYS") and pk < pH:
                flags.append(f"{resn}{resi}: pKa {pk:.1f} < pH {pH} -> deprotonated")
        rows.append(rec)

    applied = []
    ov = {int(k): str(v).upper() for k, v in (overrides or {}).items()}
    if ov and os.path.exists(out_pdb):
        lines = open(out_pdb).read().splitlines(keepends=True)
        for i, l in enumerate(lines):
            if l[:6] in ("ATOM  ", "HETATM"):
                try:
                    ri = int(l[22:26])
                except ValueError:
                    continue
                if ri in ov and l[17:20].strip() in ("HID", "HIE", "HIP", "HIS"):
                    if l[17:20].strip() != ov[ri]:
                        applied.append((ri, l[17:20].strip(), ov[ri]))
                        lines[i] = l[:17] + f"{ov[ri]:>3}" + l[20:]
        open(out_pdb, "w").writelines(lines)

    states = {"pH": pH, "epsin": epsin, "output_pdb": out_pdb, "sites": rows,
              "flags": flags, "overrides_applied": applied}
    json.dump(states, open(out_pdb + ".states.json", "w"), indent=2)
    return states


if __name__ == "__main__":               # split-env subprocess fallback
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pdb"); ap.add_argument("--out", required=True)
    ap.add_argument("--pH", type=float, default=7.0)
    ap.add_argument("--epsin", type=float, default=15.0)
    ap.add_argument("--ncpus", type=int, default=8)
    ap.add_argument("--override", nargs="*", default=[])
    a = ap.parse_args()
    ov = dict(o.split(":") for o in a.override)
    s = titrate(a.pdb, a.out, a.pH, a.epsin, a.ncpus, ov)
    print(f"[pypka] {len(s['sites'])} sites, {len(s['flags'])} non-default; "
          f"overrides {s['overrides_applied']}")
    for f in s["flags"]:
        print("  ! " + f)

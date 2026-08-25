"""Protonate the protein and merge the placed substrate -> real_noep.pdb.

Single-env design: `tleap`/`pdb4amber`/`reduce` are called from PATH and pypKa is
imported in-process (all present in the `ts_finder` env). Each step degrades to a
split-env subprocess only if the tool is genuinely absent in the current env.

Protonation determinant (method=):
  "pypka" (DEFAULT)  PB/MC pKa (pypKa) decides every titratable state (ASH/GLH,
                     HID/HIE/HIP, LYN, tautomers) at the target pH, after `reduce`
                     fixes Asn/Gln/His flips; mechanism-dictated catalytic states are
                     forced via `overrides`. Correct default -- tleap pH-7 defaults
                     mis-assign buried Asp/Glu and active-site His.
  "tleap"            ff14SB pH-7 defaults + manual `overrides` only (fast fallback).
"""
from __future__ import annotations
import os, shutil, subprocess
from .. import config
from . import _pypka_worker


def _extract_chain_heavy(src_pdb, chains, outdir, drop_resn, tsa_resname):
    drop = set(drop_resn or []) | {"HOH"}
    if tsa_resname:
        drop.add(tsa_resname)
    chains = set(chains)
    heavy = os.path.join(outdir, "prot_heavy.pdb")
    with open(heavy, "w") as fh:
        for l in open(src_pdb):
            if l[:6] not in ("ATOM  ", "HETATM"):
                continue
            if l[21] not in chains or l[17:20].strip() in drop or l[16] not in (" ", "A"):
                continue
            if l[76:78].strip() == "H" or l[12:16].strip().startswith("H"):
                continue
            fh.write(l[:16] + " " + l[17:])
        fh.write("END\n")
    return heavy


def _strip_H(src, dst):
    with open(dst, "w") as fh:
        for l in open(src):
            if l[:6] in ("ATOM  ", "HETATM") and not (
                    l[76:78].strip() == "H" or l[12:16].strip().startswith("H")):
                fh.write(l)
        fh.write("END\n")
    return dst


def _reduce_flips(heavy, outdir):
    """Fix Asn/Gln/His amide flips with `reduce`; strip its H (re-added later)."""
    rb = shutil.which(config.reduce_bin()) or config.reduce_bin()
    if not (rb and (os.path.exists(rb) or shutil.which(rb))):
        print("[protein] reduce not found -- skipping flip optimization")
        return heavy
    built = os.path.join(outdir, "reduced.pdb")
    with open(built, "w") as out, open(os.path.join(outdir, "reduce.log"), "w") as err:
        subprocess.run([rb, "-FLIP", "-Quiet", heavy], stdout=out, stderr=err)
    return _strip_H(built, os.path.join(outdir, "prot_flipped.pdb"))


def _pypka(heavy, outdir, pH, overrides, epsin=15.0, ncpus=8):
    out = os.path.join(outdir, "pypka_amber.pdb")
    try:
        import pypka  # noqa: F401  (present in the single ts_finder env)
        states = _pypka_worker.titrate(heavy, out, pH=pH, epsin=epsin, ncpus=ncpus,
                                       overrides=overrides)
        for f in states["flags"]:
            print("  ! " + f)
    except ImportError:
        py = config.pypka_python()
        if not py:
            raise RuntimeError("pypKa not importable and ENZYME_TS_PYPKA_PY not set")
        env = dict(os.environ)
        cmd = [py, os.path.join(os.path.dirname(__file__), "_pypka_worker.py"),
               heavy, "--out", out, "--pH", str(pH)]
        for ri, rn in (overrides or {}).items():
            cmd += ["--override", f"{ri}:{rn}"]
        subprocess.run(cmd, env=env, check=True,
                       stdout=open(os.path.join(outdir, "pypka.log"), "w"),
                       stderr=subprocess.STDOUT)
    if not os.path.exists(out):
        raise RuntimeError(f"pypKa did not write {out} (see {outdir}/pypka.log)")
    return out


def _tleap(in_pdb, outdir, manual_overrides=None):
    if manual_overrides:
        lines = open(in_pdb).readlines()
        for i, l in enumerate(lines):
            if l[:6] == "ATOM  " and int(l[22:26]) in manual_overrides:
                lines[i] = l[:17] + f"{manual_overrides[int(l[22:26])]:>3}" + l[20:]
        in_pdb = os.path.join(outdir, "with_overrides.pdb")
        open(in_pdb, "w").writelines(lines)
    tleap, pdb4amber = config.tleap_bin(), config.pdb4amber_bin()
    base = os.path.basename(in_pdb)
    body = f"""cd {outdir}
{pdb4amber} -i {base} -o amber_in.pdb -y >/dev/null 2>&1 || {pdb4amber} -i {base} -o amber_in.pdb -y
cat > tleap.in <<EOF
source leaprc.protein.ff14SB
prot = loadpdb amber_in.pdb
saveamberparm prot protein.parm7 protein.rst7
savepdb prot protein_amber.pdb
quit
EOF
{tleap} -s -f tleap.in > tleap.log 2>&1 || {{ tail -30 tleap.log; exit 1; }}
"""
    if shutil.which(tleap):                                   # single env: tleap on PATH
        sh = "set -e\n" + body
    else:                                                     # split-env fallback
        env = config.amber_env()
        if not env:
            raise RuntimeError(f"{tleap} not on PATH and ENZYME_TS_AMBER_ENV not set")
        sh = f"set -e\nsource {config.conda_sh()}\nconda activate {env}\n" + body
    subprocess.run(["bash", "-c", sh], check=True)
    return os.path.join(outdir, "protein_amber.pdb")


def prep_protein(src_pdb, chains, outdir, method="pypka", pH=7.0, overrides=None,
                 reduce_flips=True, drop_resn=None, tsa_resname=None):
    heavy = _extract_chain_heavy(src_pdb, chains, outdir, drop_resn, tsa_resname)
    if method == "pypka":
        if reduce_flips:
            heavy = _reduce_flips(heavy, outdir)
        print(f"[protein] pypKa protonation @ pH {pH} (overrides {overrides or {}})")
        protonated = _pypka(heavy, outdir, pH, overrides)
        return _tleap(protonated, outdir)                    # states already in the PDB
    if method == "tleap":
        print(f"[protein] tleap ff14SB defaults (manual overrides {overrides or {}})")
        return _tleap(heavy, outdir, manual_overrides=overrides)
    raise ValueError(f"unknown protonation method {method!r}")


def merge(protein_amber, placed_substrate, outdir):
    prot = [l for l in open(protein_amber) if l[:6] in ("ATOM  ", "TER   ")]
    sub = [l for l in open(placed_substrate) if l[:6] == "HETATM"]
    real = os.path.join(outdir, "real_noep.pdb")
    with open(real, "w") as fh:
        fh.writelines(prot)
        if prot[-1][:3] != "TER":
            fh.write("TER\n")
        fh.writelines(sub)
        fh.write("END\n")
    print(f"[protein] real_noep.pdb ({len(prot)} protein lines + {len(sub)} substrate atoms)")
    return real

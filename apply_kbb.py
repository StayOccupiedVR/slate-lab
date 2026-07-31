"""One-shot: make team + kbb the shipped model; move ERA starter + park to LEGACY.

Run from the slate-lab folder:  python apply_kbb.py
Safe to re-run; it verifies before and after.
"""
import re
import sys

p = "slate_lab/features.py"
s = open(p, encoding="utf-8").read()

if '"kbb":      ["sp_kbb"],\n}' in s and '"starter"' not in s.split("LEGACY")[0].split("GROUPS")[1]:
    print("Already applied. GROUPS is team + kbb.")
    sys.exit(0)

old_groups = '''    "starter":  ["sp_edge", "sp_known"],
    "kbb":      ["sp_kbb"],
    "park":     ["park_run"],
}'''
if old_groups not in s:
    print("ERROR: GROUPS block not in expected form. No changes made.")
    print("Current GROUPS block:")
    print(s[s.index("GROUPS"):s.index("}", s.index("GROUPS")) + 1])
    sys.exit(1)

s = s.replace(old_groups, '''    "kbb":      ["sp_kbb"],
}''', 1)

legacy_anchor = "LEGACY: dict[str, list[str]] = {"
if legacy_anchor not in s:
    print("ERROR: LEGACY block not found. No changes made.")
    sys.exit(1)

s = s.replace(legacy_anchor, legacy_anchor + '''
    # retired 2026-07-30: equal holdout logloss to kbb (0.6827 both) but ERA
    # stabilizes far slower and carries more team-strength collinearity
    # (corr w/ pyth_diff +0.31 vs +0.25); kbb also stronger w/ outcome
    # (+0.103 vs +0.081). Kept for re-testing.
    "starter":  ["sp_edge", "sp_known"],
    # retired 2026-07-30: +0.0001 in the ablation, below the noise floor
    # at n=2123. Real effect may exist; undetectable at this sample size.
    "park":     ["park_run"],''', 1)

open(p, "w", encoding="utf-8").write(s)

# verify by import
sys.path.insert(0, ".")
import importlib
import slate_lab.features as F
importlib.reload(F)
assert list(F.GROUPS) == ["team", "kbb"], F.GROUPS
assert "starter" in F.LEGACY and "park" in F.LEGACY
print("GROUPS -> " + " + ".join(F.GROUPS))
print("LEGACY now holds: " + ", ".join(F.LEGACY))
print("Applied cleanly.")

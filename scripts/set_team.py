"""Stamp the submission eval file with your team ID and USNs, with no re-evaluation.

  python scripts/set_team.py --team T07 --usns "1RV23AI001,1RV23AI002,1RV23AI003,1RV23AI004"

Renames results/eval_P4_<old>.md to results/eval_P4_<TEAM>.md, writes the USNs
(comma separated) on the first line, and updates the team ID in the title.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="team ID, e.g. T07")
    ap.add_argument("--usns", required=True, help="comma-separated USNs of all members")
    a = ap.parse_args()
    team = a.team.strip()
    usns = ",".join(u.strip() for u in a.usns.split(",") if u.strip())
    if not re.fullmatch(r"[A-Za-z0-9_-]+", team):
        raise SystemExit("team ID may only contain letters, digits, - and _")
    if not usns:
        raise SystemExit("give at least one USN")

    res = ROOT / "results"
    files = sorted(res.glob("eval_P4_*.md"))
    if not files:
        raise SystemExit("no results/eval_P4_*.md found - run scripts/evaluate.py first")
    src = files[0]
    lines = src.read_text(encoding="utf-8").splitlines()
    lines[0] = usns
    text = "\n".join(lines) + "\n"
    text = re.sub(r"— Team [A-Za-z0-9_-]+", f"— Team {team}", text, count=1)
    dst = res / f"eval_P4_{team}.md"
    dst.write_text(text, encoding="utf-8")
    if src != dst:
        src.unlink()
    for p in [ROOT / "README.md"]:
        t = p.read_text(encoding="utf-8")
        t = re.sub(r"eval_P4_[A-Za-z0-9_<>-]+\.md", f"eval_P4_{team}.md", t)
        t = t.replace("**Team:** `<T01>`", f"**Team:** `{team}`")
        p.write_text(t, encoding="utf-8")
    print(f"wrote {dst.relative_to(ROOT)}  (first line: {usns})")


if __name__ == "__main__":
    main()

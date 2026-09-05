#!/usr/bin/env python3
"""Write the repository URL, release tag, ORCIDs and Zenodo DOI into the files
that must carry them, so the three copies cannot drift apart.

Those identifiers live in three places that are easy to update inconsistently:

    CITATION.cff                             repository-code, version, doi, ORCIDs
    manuscript/latex/*.tex                   Data and Code Availability statement,
                                             in both the ACS Letter and the JES pair
    .zenodo.json                             related_identifiers (isSupplementTo)

The Zenodo DOI does not exist until the deposit is created, so this runs in two
passes. Reserve the DOI in the Zenodo draft (Zenodo issues it before you
publish), then run once with everything known:

    python scripts/finalise_release.py \\
        --repo https://github.com/you/your-repo \\
        --tag v1.0.0 \\
        --doi 10.5281/zenodo.1234567 \\
        --orcid "Sarkar=0000-0002-1825-0097" \\
        --orcid "Gross=0000-0001-5109-3700"

Pass --check to verify the files are consistent without writing anything; that
is what to run immediately before submitting.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CITATION = ROOT / "CITATION.cff"
LATEX = ROOT / "manuscript/latex"
# Both papers carry the availability statement and must be filled in together:
# the ACS Energy Letters submission and the J. Energy Storage fallback. The
# Highlights block is Elsevier's and exists only in the JES manuscript.
ACS_MANUSCRIPT = LATEX / "Manuscript_ACS_Energy_Letters.tex"
JES_MANUSCRIPT = LATEX / "JES/Manuscript.tex"
MANUSCRIPTS = [ACS_MANUSCRIPT, JES_MANUSCRIPT]
HIGHLIGHTS_MANUSCRIPT = JES_MANUSCRIPT
ZENODO = ROOT / ".zenodo.json"

PLACEHOLDER_PATTERNS = [
    (CITATION, r"OWNER/REPO"),
    (CITATION, r"zenodo\.XXXXXXX"),
    # A. Gross does not supply an ORCID; only an unfilled placeholder is a defect.
    (CITATION, r"0000-0000-0000-0000"),
] + [(m, r"\\todo\{") for m in MANUSCRIPTS]

AVAILABILITY_BLOCK = """The code and derived data are archived at
\\protect\\file{{{repo}}}
(release {tag}), deposited at Zenodo under
DOI~{doi}."""


HIGHLIGHTS = ROOT / "manuscript/submission/highlights.md"


def _manuscript_highlights() -> list[str]:
    """The five bullets as the JES manuscript states them, in plain text."""
    if not HIGHLIGHTS_MANUSCRIPT.is_file():
        return []
    tex = HIGHLIGHTS_MANUSCRIPT.read_text(encoding="utf-8")
    block = re.search(r"\\section\*\{Highlights\}(.*?)\\end\{itemize\}", tex, re.S)
    if not block:
        return []
    items = re.findall(r"\\item\s+(.*?)\s*(?=\n\s*\\item|\Z)", block.group(1), re.S)
    out = []
    for item in items:
        s = " ".join(item.split())
        s = s.replace("\\%", "%").replace("\\,", " ").replace("\\textdegree C", "C")
        out.append(re.sub(r"\s+", " ", s).strip())
    return out


def check() -> int:
    problems = []

    for path in [CITATION, ZENODO, *MANUSCRIPTS]:
        if not path.is_file():
            problems.append(f"missing file: {path.relative_to(ROOT)}")

    # The separate highlights file is uploaded to the journal alongside the PDF.
    # It has drifted from the manuscript before, so compare the two every time.
    if HIGHLIGHTS.is_file():
        if not HIGHLIGHTS_MANUSCRIPT.is_file():
            problems.append(f"{HIGHLIGHTS.relative_to(ROOT)} cannot be checked: "
                            f"{HIGHLIGHTS_MANUSCRIPT.relative_to(ROOT)} is missing")
        stated = _manuscript_highlights()
        listed = [re.sub(r"\s*\[\d+\]\s*$", "", line[2:]).strip()
                  for line in HIGHLIGHTS.read_text(encoding="utf-8").splitlines()
                  if line.startswith("- ")]
        for bullet in stated:
            if bullet not in listed:
                problems.append("highlights.md is out of step with the manuscript: "
                                f"missing {bullet!r}")
        for bullet in stated:
            if len(bullet) > 85:
                problems.append(f"highlight exceeds Elsevier's 85 characters ({len(bullet)}): {bullet!r}")

    for path, pattern in PLACEHOLDER_PATTERNS:
        if not path.is_file():
            problems.append(f"missing file: {path.relative_to(ROOT)}")
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(pattern, line):
                problems.append(f"{path.relative_to(ROOT)}:{number}: unresolved -- {line.strip()[:90]}")
    if problems:
        print("NOT READY TO SUBMIT:")
        for item in problems:
            print("  " + item)
        return 1
    checked = ", ".join(str(p.relative_to(ROOT)) for p in MANUSCRIPTS)
    print(f"release identifiers resolved in CITATION.cff, .zenodo.json and {checked}")
    return 0


def apply(repo: str, tag: str, doi: str, orcids: dict[str, str]) -> int:
    doi = doi.removeprefix("https://doi.org/")

    # --- CITATION.cff --------------------------------------------------------
    text = CITATION.read_text(encoding="utf-8")
    text = re.sub(r'repository-code:.*', f'repository-code: "{repo}"', text)
    text = re.sub(r'#\s*version:.*', f'version: "{tag}"', text)
    text = re.sub(r'#\s*doi:.*', f'doi: "{doi}"', text)
    today = _dt.date.today().isoformat()
    text = re.sub(r'#\s*date-released:.*', f'date-released: {today}', text)
    for family, orcid in orcids.items():
        orcid = orcid.removeprefix("https://orcid.org/")
        # Keep the line break and the indentation: an earlier version let
        # \s* swallow both and welded the orcid key onto the email line.
        text = re.sub(
            rf'(family-names: {re.escape(family)}\b.*?\n)([ \t]*)#?\s*orcid:[^\n]*(\n)',
            rf'\1\2orcid: "https://orcid.org/{orcid}"\3',
            text, count=1, flags=re.S)
    CITATION.write_text(text, encoding="utf-8")

    # --- manuscript availability statements ----------------------------------
    replacement = AVAILABILITY_BLOCK.format(repo=repo, tag=tag, doi=doi)
    for manuscript in MANUSCRIPTS:
        if not manuscript.is_file():
            print(f"note: {manuscript.relative_to(ROOT)} is missing; not updated")
            continue
        text = manuscript.read_text(encoding="utf-8")
        text, count = re.subn(r"\\todo\{BEFORE SUBMISSION:.*?\}\s*\n",
                              lambda _match: replacement + "\n",   # literal: the block has backslashes
                              text, count=1, flags=re.S)
        if count == 0:
            print(f"note: no \\todo block in {manuscript.name} (already filled in?)")
        text = re.sub(r"% PI CRITICAL CHECK BEFORE SUBMISSION:\n"
                      r"% Insert the actual repository URL.*?\n", "", text)
        manuscript.write_text(text, encoding="utf-8")

    # --- .zenodo.json --------------------------------------------------------
    data = json.loads(ZENODO.read_text(encoding="utf-8"))
    related = [r for r in data.get("related_identifiers", [])
               if r.get("relation") != "isSupplementTo"
               or "github" not in r.get("identifier", "")]
    related.append({"identifier": f"{repo}/tree/{tag}",
                    "relation": "isSupplementTo",
                    "resource_type": "software",
                    "scheme": "url"})
    data["related_identifiers"] = related
    data["version"] = tag
    ZENODO.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")

    print(f"repo {repo}\ntag  {tag}\ndoi  {doi}")
    for family, orcid in orcids.items():
        print(f"orcid {family}: {orcid}")
    print("\nRebuild the PDFs (bash manuscript/latex/compile.sh), then rebuild the "
          f"deposit:\n  python scripts/make_release_archive.py --version {tag}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="report unresolved placeholders and exit")
    parser.add_argument("--repo", help="repository URL, e.g. https://github.com/you/repo")
    parser.add_argument("--tag", help="immutable release tag, e.g. v1.0.0")
    parser.add_argument("--doi", help="Zenodo DOI, e.g. 10.5281/zenodo.1234567")
    parser.add_argument("--orcid", action="append", default=[], metavar="FAMILY=ID",
                        help="repeatable, e.g. --orcid Sarkar=0000-0002-1825-0097")
    args = parser.parse_args()

    if args.check:
        return check()
    if not (args.repo and args.tag and args.doi):
        parser.error("--repo, --tag and --doi are all required (or use --check). "
                     "The repo URL and tag are already written into CITATION.cff "
                     "and .zenodo.json; what remains is the reserved Zenodo DOI.")

    orcids = {}
    for item in args.orcid:
        if "=" not in item:
            parser.error(f"--orcid expects FAMILY=ID, got {item!r}")
        family, value = item.split("=", 1)
        orcids[family.strip()] = value.strip()
    return apply(args.repo, args.tag, args.doi, orcids)


if __name__ == "__main__":
    sys.exit(main())

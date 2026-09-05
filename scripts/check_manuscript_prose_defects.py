#!/usr/bin/env python3
"""Prose defects that survive a naive grep.

Two classes, both found in review after they had passed every existing check.

1. Duplicated words across a line break or a stripped comment. A line-bounded,
   comment-blind grep for "the the" returns nothing when the source reads

       ... argues from this screen that the
       % xref: sec:eis_results
       the four candidate circuits ...

   because LaTeX drops the comment line and joins the text. This check strips
   full-line comments first, then looks for a repeated word across whitespace
   including newlines.

2. Section and figure labels detached from their heading. A \\label separated
   from its \\section, \\subsection or \\caption by intervening material still
   resolves today, but silently binds to the wrong counter as soon as a table,
   figure or equation is inserted into the gap.

Both are run against the accepted (clean) sources, since that is what a reader
sees, and against the marked-up sources, since that is what is edited.
"""
from __future__ import annotations
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["manuscript/latex/JES/Manuscript.tex", "manuscript/latex/JES/SM.tex",
         "manuscript/latex/Manuscript_ACS_Energy_Letters.tex",
         "manuscript/latex/SM_ACS_Energy_Letters.tex"]
HEAD = re.compile(r"\\(?:sub)*section\*?\{|\\caption\{|\\paragraph\{")
LABEL = re.compile(r"\\label\{([^}]*)\}")
# words that legitimately repeat in LaTeX or in English
ALLOW = {"had", "that", "very"}


def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in text.split("\n"))


def main() -> int:
    # Fail closed on a path that no longer resolves. This used to `continue`,
    # so when the JES pair moved into a subdirectory the scanner quietly
    # inspected zero files and reported a clean run.
    missing = [rel for rel in FILES if not (ROOT / rel).exists()]
    if missing:
        for rel in missing:
            print(f"  MISSING FILE    {rel}")
        print(f"\n{len(missing)} declared source file(s) not found; nothing was scanned")
        return 1

    bad = 0
    scanned = 0
    for rel in FILES:
        p = ROOT / rel
        scanned += 1
        raw = p.read_text()
        body = strip_comments(raw)
        # 1. duplicated words, across newlines
        for m in re.finditer(r"\b([A-Za-z]{2,})\s+\1\b", body):
            if m.group(1).lower() in ALLOW:
                continue
            line = body[:m.start()].count("\n") + 1
            print(f"  DUPLICATE WORD  {rel}:{line}  '{m.group(0).strip()}'")
            bad += 1
        # 2. labels detached from the heading they name. Only sectioning labels
        #    are checked: a fig:/tab: label legitimately follows a multi-line
        #    caption inside its float, which binds the counter correctly.
        lines = raw.split("\n")
        for i, ln in enumerate(lines):
            m = LABEL.search(ln)
            if not m or not m.group(1).startswith("sec:"):
                continue
            window = lines[max(0, i - 2):i + 1]
            if any(("\\section" in w) or ("\\subsection" in w) for w in window):
                continue
            print(f"  DETACHED LABEL  {rel}:{i+1}  {m.group(1)}  "
                  f"(no sectioning command within 2 lines above)")
            bad += 1
    if not scanned:
        print("\nno files scanned; treating that as a failure")
        return 1
    print(f"\n{bad} prose defect(s) found in {scanned} file(s)" if bad
          else f"\nno prose defects found in {scanned} file(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

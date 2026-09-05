#!/usr/bin/env python3
"""Verify every cross-reference between Manuscript.tex and SM.tex.

The two documents are separate compiles, so a pointer from one into the other
is a hardcoded string with no \\ref safety net. Round 9 inserted three
equations and one subsection into the main text and left four SI pointers
stale for two rounds; nothing in the build caught it, because both documents
compiled cleanly the whole time.

This script resolves each hardcoded pointer against the *other* document's
.aux file, and additionally reports figure panels that are defined in a
caption but cited nowhere -- the defect that left Figure 6a plotted but
undiscussed. Checking captions against rendered figures does not catch either
class.

Run after any structural edit: adding or removing an equation, section,
table, figure, or figure panel. Requires both documents to have been compiled
at least once so the .aux files are current.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# JES pair; see the note in check_results_table_layout.py.
LATEX_DIR = Path(__file__).resolve().parents[1] / "manuscript" / "latex" / "JES"

# \newlabel{key}{{printed number}{page}...}
NEWLABEL = re.compile(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{([0-9]+)\}")
# "Supplementary Section~S3", "Supplementary Table~S3", "Supplementary Figure~S1"
# S[0-9.]+ would swallow a sentence-ending period and look for a float "S1."
MAIN_TO_SI = re.compile(r"Supplementary\s+(Section|Figure|Table)~?\s*(S[0-9]+(?:\.[0-9]+)*)")
# "Section~3.5 of the main text", "Eq.~(8) of the main text", "Table~3 of the main text"
SI_TO_MAIN = re.compile(
    r"(Section|Figure|Table|Eq\.|Equation)~?\s*\(?([0-9]+(?:\.[0-9]+)?)\)?"
)
PREFIX = {"Section": "sec", "Figure": "fig", "Table": "tab",
          "Eq.": "eq", "Equation": "eq"}


# An `% xref: eq:arrhenius` comment on the line after a cross-document pointer
# binds that pointer to a label in the other document, so the printed number is
# checked against the label rather than merely against the set of numbers that
# exist. Without it a stale "Eq.~(8)" resolves happily to whichever equation now
# prints as 8, which is how two SI pointers survived a renumbering twice.
XREF = re.compile(r"%\s*xref:\s*((?:sec|fig|tab|eq):[A-Za-z0-9_:.-]+)")


def strip_comments(text: str) -> str:
    """Drop comment lines, keeping the `% xref:` bindings the checker reads."""
    return re.sub(
        r"(?m)^\s*%(?!\s*xref:).*$",
        "",
        text,
    )


def labels(aux_path: Path) -> dict[str, str]:
    """Map label key -> printed number, from a compiled .aux file."""
    if not aux_path.exists():
        sys.exit(f"missing {aux_path} -- compile the document first")
    return {m.group(1): m.group(2) for m in NEWLABEL.finditer(aux_path.read_text())}


def resolve(number: str, kind: str, target: dict[str, str]) -> list[str]:
    """Label keys in the target document printing as `number` with the right prefix."""
    want = PREFIX[kind]
    return [k for k, v in target.items() if v == number and k.startswith(want + ":")]


SECTION_TITLE = re.compile(
    r"\\(?:sub)*section\*?\{([^}]*)\}\s*(?:%[^\n]*\n\s*)?\\label\{((?:sec):[^}]*)\}"
)


def titles(tex: str) -> dict[str, str]:
    """Map sec: label -> the section title it sits under."""
    return {m.group(2): m.group(1) for m in SECTION_TITLE.finditer(tex)}


def check_pointers(src_name: str, src_text: str, pattern: re.Pattern,
                   target: dict[str, str], target_titles: dict[str, str],
                   report: list[str]) -> tuple[int, int]:
    """Returns (hard failures, pointers needing a human read).

    A pointer that resolves is NOT necessarily correct: a stale number usually
    lands on some other real float. Existence is machine-checkable; aim is not.
    A pointer carrying an `% xref: <label>` binding is checked against that
    label and fails hard on a mismatch, naming the number it should now print.
    Failing that, a citing sentence that names its target is verified by name;
    anything else is printed with its context for a human read.
    """
    failures = review = 0
    for match in pattern.finditer(src_text):
        kind, number = match.group(1), match.group(2)
        hits = resolve(number, kind, target)
        label = f"{kind} {number}"
        context = re.sub(r"\s+", " ", src_text[max(0, match.start() - 55):match.end() + 55]).strip()

        # An explicit binding outranks everything else: it is the only check
        # here that survives a renumbering of the other document.
        binding = XREF.search(src_text, match.end(), match.end() + 160)
        if binding:
            key = binding.group(1)
            if key not in target:
                failures += 1
                report.append(f"  FAIL  {src_name}: {label:<13} bound to '{key}', "
                              f"which is not a label in the other document")
                report.append(f"          ...{context}...")
            elif target[key] != number:
                failures += 1
                report.append(f"  FAIL  {src_name}: {label:<13} is stale -- '{key}' "
                              f"now prints as {target[key]}")
                report.append(f"          ...{context}...")
            else:
                report.append(f"  ok    {src_name}: {label:<13} -> {key} (binding verified)")
            continue

        if not hits:
            failures += 1
            report.append(f"  FAIL  {src_name}: {label:<13} unresolvable -- no such float")
            report.append(f"          ...{context}...")
            continue
        title = target_titles.get(hits[0], "")
        # If the citing sentence quotes the target's title, verify it.
        tail = src_text[match.end():match.end() + 120]
        if title and title.lower()[:24] in re.sub(r"\s+", " ", tail).lower():
            report.append(f"  ok    {src_name}: {label:<13} -> {hits[0]} \"{title}\" (name verified)")
        else:
            review += 1
            named = f' "{title}"' if title else ""
            report.append(f"  ?     {src_name}: {label:<13} -> {hits[0]}{named}")
            report.append(f"          ...{context}...")
    return failures, review


# A panel letter is a lone letter (not part of a word) immediately after the
# \ref, optionally continued as ",d" or ",\,d". Without the (?![a-z]) guard the
# match runs on into the following prose.
PANEL_CITE = re.compile(
    r"\\ref\{(fig:[^}]*)\}((?:[a-z](?![a-z]))(?:\s*,\s*\\?,?\s*[a-z](?![a-z]))*)"
)


def check_panels(name: str, text: str, report: list[str]) -> int:
    """Panels defined in a caption but never cited, or cited but not defined.

    A figure cited as a whole (\\ref with no trailing panel letter) covers all
    of its panels; orphans are only reported for figures whose text citations
    single panels out, which is the case that leaves one panel undiscussed.
    """
    failures = 0
    captions = {}
    for m in re.finditer(r"\\caption\{(.*?)\}\s*\\label\{(fig:[^}]*)\}", text, re.S):
        letters = []
        for group in re.findall(r"\\textbf\{\((.*?)\)\}", m.group(1)):
            letters += re.findall(r"[a-z]", group)
        captions[m.group(2)] = letters

    cited: dict[str, set[str]] = {k: set() for k in captions}
    whole: set[str] = set()
    for m in re.finditer(r"\\ref\{(fig:[^}]*)\}", text):
        key = m.group(1)
        if key not in cited:
            continue
        panels = PANEL_CITE.match(text, m.start())
        if panels:
            cited[key] |= set(re.findall(r"[a-z]", panels.group(2)))
        else:
            whole.add(key)

    for key, defined in captions.items():
        if not defined:
            continue
        used = cited[key]
        unknown = sorted(used - set(defined))
        orphans = [] if (key in whole or not used) else [p for p in defined if p not in used]
        if unknown:
            failures += 1
            report.append(f"  FAIL  {name}: {key} cites panel(s) {unknown} not in its caption")
        if orphans:
            failures += 1
            report.append(f"  FAIL  {name}: {key} singles out panels but never cites {orphans}")
        if not unknown and not orphans:
            how = "cited as a whole" if key in whole and not used else f"panels cited: {sorted(used) or 'whole figure'}"
            report.append(f"  ok    {name}: {key} defines {defined}; {how}")
    return failures


def bib_entries(text: str, bbl: Path | None = None) -> set[str]:
    """Entry keys, from an embedded thebibliography or, if the document uses
    \\bibliography{...}, from the compiled .bbl."""
    keys = set(re.findall(r"\\bibitem\[[^\]]*\]\{([^}]*)\}", text))
    if not keys and bbl is not None and bbl.exists():
        keys = set(re.findall(r"\\bibitem\[[^\]]*\]\{([^}]*)\}", bbl.read_text()))
    return keys


def check_bibliography(name: str, text: str, other_text: str, report: list[str],
                       bbl: Path | None = None, other_bbl: Path | None = None) -> int:
    failures = 0
    entries = bib_entries(text, bbl)
    cited: set[str] = set()
    for group in re.findall(r"\\(?:citep|citealt|cite|citeauthor)\s*\{([^}]*)\}", text):
        cited |= {k.strip() for k in group.split(",")}
    for orphan in sorted(entries - cited):
        failures += 1
        report.append(f"  FAIL  {name}: bibliography entry '{orphan}' is never cited")
    for missing in sorted(cited - entries):
        failures += 1
        report.append(f"  FAIL  {name}: '{missing}' cited but has no bibliography entry")
    if name == "SI":
        other = bib_entries(other_text, other_bbl)
        for only in sorted(entries - other):
            report.append(f"  note  SI: '{only}' is not in the main reference list "
                          f"(fine for Elsevier, a defect under APS rules)")
    if not failures:
        report.append(f"  ok    {name}: {len(entries)} entries, all cited, none missing")
    return failures


def main() -> int:
    # Optional directory argument so the checker can be run against a copy
    # (used to confirm it still fails on the defects it was written for).
    latex_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else LATEX_DIR
    # Report a missing input by name instead of raising a bare FileNotFoundError:
    # the .aux and .bbl files exist only after a compile, and the difference
    # between "not compiled yet" and "wrong directory" should be readable.
    required = ["Manuscript.tex", "SM.tex", "Manuscript.aux", "SM.aux", "Manuscript.bbl"]
    absent = [name for name in required if not (latex_dir / name).exists()]
    if absent:
        print(f"FAIL: {latex_dir} is missing {', '.join(absent)}.")
        print("Compile both documents first (./compile.sh JES/Manuscript JES/SM "
              "leaves the PDFs but removes the auxiliary files, so run pdflatex "
              "directly when this check is needed).")
        return 1

    main_tex = strip_comments((latex_dir / "Manuscript.tex").read_text())
    si_tex = strip_comments((latex_dir / "SM.tex").read_text())
    main_labels = labels(latex_dir / "Manuscript.aux")
    si_labels = labels(latex_dir / "SM.aux")

    report: list[str] = []
    failures = review = 0

    report.append("cross-document pointers  (ok = bound to a label, or names its target;")
    report.append("                          ?  = resolves, but aim needs a human read)")
    f, r = check_pointers("main->SI", main_tex, MAIN_TO_SI, si_labels, titles(si_tex), report)
    failures += f
    review += r
    f, r = check_pointers("SI->main", si_tex, SI_TO_MAIN, main_labels, titles(main_tex), report)
    failures += f
    review += r

    report.append("\nfigure panels")
    failures += check_panels("main", main_tex, report)
    failures += check_panels("SI", si_tex, report)

    report.append("\nbibliography")
    failures += check_bibliography("main", main_tex, si_tex, report,
                                  latex_dir / "Manuscript.bbl", None)
    failures += check_bibliography("SI", si_tex, main_tex, report,
                                  None, latex_dir / "Manuscript.bbl")

    print("\n".join(report))
    checked = failures + review + sum(1 for line in report if line.lstrip().startswith("ok "))
    if not checked:
        print("\nFAIL: no cross-document pointers were examined at all. Either the "
              "documents lost their pointers or this scan matched nothing.")
        return 1
    print(f"\n{'PASS' if not failures else 'FAIL'}: {failures} broken pointer(s); "
          f"{review} resolved pointer(s) to eyeball; {checked} pointer(s) examined.")
    if review:
        print("A stale number normally lands on some other real float, so a clean "
              "exit code does not mean every pointer aims where it should.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

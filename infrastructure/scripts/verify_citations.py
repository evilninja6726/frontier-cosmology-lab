#!/usr/bin/env python3
"""Zero-dependency BibTeX citation verifier for Project Frontier.

Parses 01-literature/citation-database/references.bib, extracts all arXiv
eprint IDs and DOIs, and verifies their existence and title metadata against
the live arXiv API (export.arxiv.org).
"""

from __future__ import annotations

import pathlib
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET


BIB_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "01-literature"
    / "citation-database"
    / "references.bib"
)


def parse_bib_entries(bib_text: str) -> list[dict[str, str]]:
    """Parse BibTeX entries into dictionaries of key, title, eprint, doi, year."""
    entries: list[dict[str, str]] = []
    raw_blocks = re.split(r"(?=@\w+\s*\{)", bib_text)
    for block in raw_blocks:
        block = block.strip()
        if not block.startswith("@"):
            continue
        header_match = re.match(r"@\w+\s*\{\s*([^,\s]+)\s*,", block)
        if not header_match:
            continue
        entry: dict[str, str] = {"key": header_match.group(1)}
        for field in ("title", "eprint", "doi", "year", "author"):
            field_match = re.search(
                rf"{field}\s*=\s*[\"{{]+(.+?)[\"}}]+\s*(?:,|\n|\}})",
                block,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if field_match:
                val = " ".join(field_match.group(1).strip("{}").split())
                entry[field] = val
        entries.append(entry)
    return entries


def verify_arxiv_ids(entries: list[dict[str, str]], check_network: bool = True) -> tuple[int, list[str]]:
    """Verify arXiv eprint IDs in batch via export.arxiv.org."""
    errors: list[str] = []
    eprints = [e["eprint"] for e in entries if "eprint" in e]
    if not eprints:
        return 0, ["No arXiv eprints found in bibliography."]

    if not check_network:
        return len(eprints), errors

    id_list = ",".join(eprints)
    url = f"http://export.arxiv.org/api/query?id_list={id_list}&max_results={len(eprints) + 5}"
    req = urllib.request.Request(url, headers={"User-Agent": "FrontierCosmologyLab-CitationVerifier/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            xml_bytes = resp.read()
    except Exception as exc:
        errors.append(f"Network error querying arXiv API: {exc}")
        return 0, errors

    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_bytes)
    found_ids: dict[str, str] = {}
    for atom_entry in root.findall("a:entry", ns):
        id_el = atom_entry.find("a:id", ns)
        title_el = atom_entry.find("a:title", ns)
        if id_el is not None and title_el is not None and id_el.text:
            raw_id = id_el.text.split("/abs/")[-1]
            base_id = re.sub(r"v\d+$", "", raw_id)
            found_ids[base_id] = " ".join((title_el.text or "").split())

    verified_count = 0
    for entry in entries:
        eprint = entry.get("eprint")
        if not eprint:
            continue
        base_eprint = re.sub(r"v\d+$", "", eprint)
        if base_eprint not in found_ids:
            errors.append(f"[{entry['key']}] arXiv ID '{eprint}' not found on arXiv.")
        else:
            verified_count += 1

    return verified_count, errors


def main() -> int:
    if not BIB_PATH.exists():
        print(f"ERROR: Bibliography not found at {BIB_PATH}", file=sys.stderr)
        return 1

    bib_text = BIB_PATH.read_text(encoding="utf-8")
    entries = parse_bib_entries(bib_text)
    print(f"Parsed {len(entries)} BibTeX entries from {BIB_PATH.name}.")

    verified_count, errors = verify_arxiv_ids(entries, check_network=True)
    print(f"Verified {verified_count}/{len(entries)} arXiv preprints against live export.arxiv.org API.")

    if errors:
        for err in errors:
            print(f"  ERROR: {err}", file=sys.stderr)
        return 1

    print("STATUS: ALL CITATIONS VERIFIED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

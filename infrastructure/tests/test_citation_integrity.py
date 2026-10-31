"""Automated integrity tests for Project Frontier citation database and workspace."""

from __future__ import annotations

import pathlib
import unittest

from infrastructure.scripts.verify_citations import BIB_PATH, parse_bib_entries


class TestCitationIntegrity(unittest.TestCase):
    """Ensure all entries in references.bib have required traceable identifiers."""

    def test_bibliography_exists_and_non_empty(self) -> None:
        self.assertTrue(BIB_PATH.exists(), f"Missing {BIB_PATH}")
        entries = parse_bib_entries(BIB_PATH.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(entries), 20, "Expected at least 20 verified frontier references.")

    def test_every_entry_has_eprint_or_doi_and_year(self) -> None:
        entries = parse_bib_entries(BIB_PATH.read_text(encoding="utf-8"))
        for entry in entries:
            key = entry.get("key", "<unknown>")
            self.assertIn("title", entry, f"Entry {key} missing title")
            self.assertIn("year", entry, f"Entry {key} missing year")
            has_id = ("eprint" in entry) or ("doi" in entry)
            self.assertTrue(has_id, f"Entry {key} missing both arXiv eprint and DOI")

    def test_core_governance_files_exist(self) -> None:
        lab_root = pathlib.Path(__file__).resolve().parents[2]
        required = [
            "README.md",
            "RESEARCH_CHARTER.md",
            "RESEARCH_STATUS.md",
            "ROADMAP.md",
            "08-skills/installed-skills.md",
            "08-skills/evaluated-skills.md",
            "03-projects/pr01-horndeski-crossing-identifiability/proposal/PROJECT_PROPOSAL.md",
            "03-projects/pr01-horndeski-crossing-identifiability/baseline/BENCHMARK_SPECIFICATION.md",
        ]
        for rel in required:
            self.assertTrue((lab_root / rel).is_file(), f"Missing required artifact: {rel}")


if __name__ == "__main__":
    unittest.main()

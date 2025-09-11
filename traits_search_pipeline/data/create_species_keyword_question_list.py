#!/usr/bin/env python3
"""
Expand templated questions by crossing placeholders with species lists.

Given:
  1) A questions file where each line is:
       "[placeholder]" keyword phrase ... | Full question ...
     e.g.
       "[predator_list]" predator larvae nymphs adults | Are [predator_list] larvae, nymphs, or adults when they are predators?

  2) Species list files:
       - predator_list_corrected.txt
       - pest_list_corrected.txt
       - full_species_list_corrected.txt

Produce an output file where each line is:
  species:keyword_phrase_with_species:question_with_species

Usage:
  python expand_questions.py \
    --questions questions.txt \
    --predator predator_list_corrected.txt \
    --pest pest_list_corrected.txt \
    --full full_species_list_corrected.txt \
    --out expanded_questions.txt
"""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path
from typing import Dict, Iterable, List


PLACEHOLDER_PATTERN = re.compile(
    r"\[(predator_list|pest_list|full_species_list)\]"
)


def load_list(path: Path) -> List[str]:
    "Load a newline-delimited list, stripping blanks and comments."
    items: List[str] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            items.append(s)
    return items


def unique_in_order(seq: Iterable[str]) -> List[str]:
    "Return unique items preserving first-seen order."
    seen = set()
    out: List[str] = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def expand_line(
    raw_line: str,
    placeholder_values: Dict[str, List[str]],
) -> List[str]:
    """
    Expand one templated line into concrete lines.

    Args:
        raw_line: One input line, expected to have ' | ' separating keyword phrase and question.
        placeholder_values: Mapping of placeholder key (e.g., 'predator_list') to list of species.

    Returns:
        A list of expanded lines in the format:
          species:keyword_phrase_with_species:question_with_species
    """
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return []

    if " | " not in line:
        # Not a templated Q line; skip
        return []

    left, right = line.split(" | ", 1)
    left = left.strip()
    right = right.strip()

    # Find placeholders present in either side, preserve order of appearance
    placeholders = unique_in_order(
        PLACEHOLDER_PATTERN.findall(left) + PLACEHOLDER_PATTERN.findall(right)
    )
    if not placeholders:
        return []

    # Build the cartesian product of species lists for all placeholders present
    value_lists: List[List[str]] = []
    for ph in placeholders:
        values = placeholder_values.get(ph, [])
        if not values:
            # No species available for this placeholder; no expansions
            return []
        value_lists.append(values)

    expanded_lines: List[str] = []
    for combo in itertools.product(*value_lists):
        # Map placeholders to selected species for this combination
        repl = {f"[{ph}]": species for ph, species in zip(placeholders, combo)}

        # Replace in both keyword phrase and question
        keyword_phrase = left
        question_text = right
        for token, species in repl.items():
            keyword_phrase = keyword_phrase.replace(token, species)
            question_text = question_text.replace(token, species)

        # For the species label on the left of the output line:
        species_label = " & ".join(
            combo
        )  # supports multi-placeholder lines, if any

        expanded_lines.append(
            f"{species_label}:{keyword_phrase}:{question_text}"
        )

    return expanded_lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand templated questions by crossing placeholders with species lists."
    )
    parser.add_argument(
        "--questions", type=Path, required=True, help="Path to questions file"
    )
    parser.add_argument(
        "--predator", type=Path, default=Path("predator_list_corrected.txt")
    )
    parser.add_argument(
        "--pest", type=Path, default=Path("pest_list_corrected.txt")
    )
    parser.add_argument(
        "--full", type=Path, default=Path("full_species_list_corrected.txt")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("expanded_questions.txt")
    )
    args = parser.parse_args()

    predator_species = load_list(args.predator)
    pest_species = load_list(args.pest)
    full_species = load_list(args.full)

    placeholder_values: Dict[str, List[str]] = {
        "predator_list": predator_species,
        "pest_list": pest_species,
        "full_species_list": full_species,
    }

    all_out: List[str] = []
    with args.questions.open("r", encoding="utf-8") as f:
        for raw in f:
            all_out.extend(expand_line(raw, placeholder_values))

    # Write results
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for line in all_out:
            f.write(line + "\n")


if __name__ == "__main__":
    main()

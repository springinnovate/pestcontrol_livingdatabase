import argparse
from pathlib import Path


def load_corrections(path):
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(
                    f'Invalid correction on line {lineno}: {line!r} (expected "original:corrected")'
                )
            original, corrected = line.split(":", 1)
            original = original.strip()
            corrected = corrected.strip()
            if original:
                mapping[original] = corrected
    return mapping


def main():
    parser = argparse.ArgumentParser(
        description="Replace misspelled species names with corrected spellings."
    )
    parser.add_argument(
        "species_file", type=Path, help="Path to file with species names"
    )
    parser.add_argument(
        "corrections_file",
        type=Path,
        help="Path to file with corrections (original:corrected)",
    )
    args = parser.parse_args()

    corrections = load_corrections(args.corrections_file)

    with open(args.species_file, "r", encoding="utf-8") as fin:
        for line in fin:
            raw = line.rstrip("\n")
            name = raw.strip()
            if not name or name.startswith("#"):
                print(raw)
                continue
            fixed = corrections.get(name, name)
            print(fixed)


if __name__ == "__main__":
    main()

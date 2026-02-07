from __future__ import annotations

from pathlib import Path
import csv


def extract_labels(dataset_root: Path, output_csv: Path) -> int:
    rows = []
    for mp4 in dataset_root.rglob("*.mp4"):
        name = mp4.name
        if name.startswith("real_"):
            label = "real"
        elif name.startswith("fake_"):
            label = "not_real"
        else:
            label = mp4.parent.name  # fallback: folder name

        rows.append({
            "filepath": str(mp4),
            "filename": name,
            "label": label,
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "filename", "label"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main() -> None:
    dataset_root = Path("data/sdfvd2.0")
    output_csv = Path("outputs/sdfvd2_filenames_labels.csv")
    total = extract_labels(dataset_root, output_csv)
    print(f"Saved {total} rows to {output_csv}")


if __name__ == "__main__":
    main()

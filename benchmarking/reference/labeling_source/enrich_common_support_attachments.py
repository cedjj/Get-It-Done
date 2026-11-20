"""Join public attachment metadata to the common-support cases without changing labels."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
CASES = OUT / "category_matched_common_support.csv"
ATTACHMENTS = ROOT / "Get_it_done_data/public-attachments-cases.csv"
BASE_URL = "https://getitdone.sandiego.gov/servlet/servlet.FileDownload?file="
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".avif"}


def case_key(value):
    value = value.strip()
    if not re.fullmatch(r"\d+", value):
        raise ValueError(f"Invalid case number: {value!r}")
    return value.lstrip("0") or "0"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Save enriched CSVs and linkage summary")
    args = parser.parse_args()
    fields, cases = read_csv(CASES)
    by_id = {row["ID"]: row for row in cases}
    by_number = {case_key(row["CASENUMBER"]): row for row in cases}
    assert len(by_id) == len(by_number) == len(cases), "Case keys must be unique"
    linked = defaultdict(list)
    seen = {}
    duplicates = 0
    source_rows = 0
    with ATTACHMENTS.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert set(reader.fieldnames) == {"Id", "Name", "ParentId", "Parent.Name"}
        for attachment in reader:
            source_rows += 1
            parent_number = case_key(attachment["Parent.Name"])
            number_match = by_number.get(parent_number)
            id_match = by_id.get(attachment["ParentId"])
            if number_match is None and id_match is None:
                continue
            if number_match is None or id_match is None or number_match["ID"] != id_match["ID"]:
                raise ValueError(f"Parent ID/case number disagree: {attachment}")
            attachment_id = attachment["Id"]
            assert re.fullmatch(r"[A-Za-z0-9]{15}(?:[A-Za-z0-9]{3})?", attachment_id), attachment_id
            if attachment_id in seen:
                assert seen[attachment_id] == attachment, "Conflicting duplicate attachment metadata"
                duplicates += 1
                continue
            seen[attachment_id] = attachment
            linked[number_match["ID"]].append(attachment)

    extensions = Counter(Path(a["Name"]).suffix.lower() for values in linked.values() for a in values)
    groups = {}
    for label in sorted({row["label_name"] for row in cases}):
        subset = [row for row in cases if row["label_name"] == label]
        groups[label] = {
            "cases": len(subset),
            "cases_with_attachments": sum(bool(linked[row["ID"]]) for row in subset),
            "attachments": sum(len(linked[row["ID"]]) for row in subset),
        }
    summary = {
        "case_rows": len(cases),
        "attachment_source_rows": source_rows,
        "cases_with_attachments": sum(bool(linked[row["ID"]]) for row in cases),
        "cases_without_attachment_match": sum(not linked[row["ID"]] for row in cases),
        "unique_matched_attachments": len(seen),
        "duplicate_matched_attachment_rows_removed": duplicates,
        "parent_id_case_number_disagreements": 0,
        "filename_extensions": dict(sorted(extensions.items())),
        "attachments_per_case_distribution": dict(sorted(Counter(len(linked[row["ID"]]) for row in cases).items())),
        "by_label": groups,
        "join_method": "CASENUMBER = Parent.Name after removing leading zeros; require ID = ParentId as independent cross-check",
        "url_template": BASE_URL + "{Id}",
        "image_identification": "Filename extension only; file contents and live URL availability have not been verified",
        "limitations": [
            "No attachment match means absent from this metadata export, not proof that the case has no photos.",
            "Attachment export provides no upload timestamps or before/after role; photos are not certified as available at initial reporting.",
            "Remote images may be compressed or lack original metadata, according to the supplied source note.",
            "CSV contains links, not embedded or downloaded image bytes.",
        ],
    }
    if args.write:
        max_photos = max(sum(Path(a["Name"]).suffix.lower() in IMAGE_EXTENSIONS for a in values) for values in linked.values())
        additions = ["PUBLIC_ATTACHMENT_COUNT", "PUBLIC_ATTACHMENT_IDS_JSON", "PUBLIC_ATTACHMENT_FILENAMES_JSON", "PUBLIC_ATTACHMENT_URLS_JSON", "PUBLIC_PHOTO_COUNT"]
        photo_fields = [f"PUBLIC_PHOTO_URL_{i}" for i in range(1, max_photos + 1)]
        assert not set(fields).intersection(additions + photo_fields)
        enriched = []
        manifest = []
        for case in cases:
            values = linked[case["ID"]]
            photos = [a for a in values if Path(a["Name"]).suffix.lower() in IMAGE_EXTENSIONS]
            row = dict(case)
            row.update({
                "PUBLIC_ATTACHMENT_COUNT": len(values),
                "PUBLIC_ATTACHMENT_IDS_JSON": json.dumps([a["Id"] for a in values]),
                "PUBLIC_ATTACHMENT_FILENAMES_JSON": json.dumps([a["Name"] for a in values]),
                "PUBLIC_ATTACHMENT_URLS_JSON": json.dumps([BASE_URL + a["Id"] for a in values]),
                "PUBLIC_PHOTO_COUNT": len(photos),
            })
            row.update({field: BASE_URL + photos[i]["Id"] if i < len(photos) else "" for i, field in enumerate(photo_fields)})
            enriched.append(row)
            for attachment in values:
                manifest.append({
                    "ID": case["ID"], "CASENUMBER": case["CASENUMBER"],
                    "label": case["label"], "label_name": case["label_name"],
                    **attachment, "PUBLIC_ATTACHMENT_URL": BASE_URL + attachment["Id"],
                    "IS_IMAGE_BY_FILENAME": str(Path(attachment["Name"]).suffix.lower() in IMAGE_EXTENSIONS).lower(),
                })
        enriched_path = OUT / "category_matched_common_support_with_photos.csv"
        manifest_path = OUT / "category_matched_common_support_attachments.csv"
        write_csv(enriched_path, fields + additions + photo_fields, enriched)
        write_csv(manifest_path, ["ID", "CASENUMBER", "label", "label_name", "Id", "Name", "ParentId", "Parent.Name", "PUBLIC_ATTACHMENT_URL", "IS_IMAGE_BY_FILENAME"], manifest)
        saved_fields, saved = read_csv(enriched_path)
        assert saved_fields == fields + additions + photo_fields
        assert [{k: row[k] for k in fields} for row in saved] == cases, "Original case data must remain identical and in order"
        assert sum(int(row["PUBLIC_ATTACHMENT_COUNT"]) for row in saved) == len(manifest) == len(seen)
        assert sum(int(row["PUBLIC_PHOTO_COUNT"]) for row in saved) == sum(extensions[e] for e in IMAGE_EXTENSIONS)
        for row in saved:
            urls = [row[field] for field in photo_fields if row[field]]
            assert len(urls) == int(row["PUBLIC_PHOTO_COUNT"])
            assert set(urls).issubset(json.loads(row["PUBLIC_ATTACHMENT_URLS_JSON"]))
        summary["sources"] = {str(path.relative_to(ROOT)): {"sha256": sha256(path)} for path in (CASES, ATTACHMENTS)}
        summary["outputs"] = [str(path.relative_to(ROOT)) for path in (enriched_path, manifest_path)]
        summary["validation"] = "Readback passed: original fields, row order, labels, attachment totals, photo totals and URL alignment preserved"
        (OUT / "common_support_photo_linkage_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

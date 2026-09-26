#!/usr/bin/env python3
"""
package_submission.py — Creates the final submission zip package for ML Challenge 2026.

According to competition guidelines, the zip archive must contain:
  <team_name>_submission.zip
  ├── output/
  │   ├── matching_results.tsv
  │   └── candidate_pairs.tsv
  ├── code/
  │   └── business_entity_resolution/
  │       ├── src/
  │       ├── README.md
  │       └── requirements.txt
  └── Documentation_template.md

Usage:
    python package_submission.py [--team-name TEAM_NAME] [--output-dir .]
"""

import os
import sys
import argparse
import zipfile
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def create_submission_zip(base_dir: str, team_name: str, zip_output_dir: str) -> str:
    zip_filename = f"{team_name}_submission.zip"
    zip_path = os.path.join(zip_output_dir, zip_filename)

    logger.info("Packaging submission archive -> %s", zip_path)

    # Required files to verify before packing
    required_files = [
        os.path.join(base_dir, "output", "matching_results.tsv"),
        os.path.join(base_dir, "output", "candidate_pairs.tsv"),
        os.path.join(base_dir, "Documentation_template.md"),
        os.path.join(base_dir, "code", "business_entity_resolution", "README.md"),
        os.path.join(base_dir, "code", "business_entity_resolution", "requirements.txt"),
    ]

    for p in required_files:
        if not os.path.isfile(p):
            logger.error("Missing required submission file: %s", p)
            sys.exit(1)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # 1. Output files
        for fname in ["matching_results.tsv", "candidate_pairs.tsv"]:
            fpath = os.path.join(base_dir, "output", fname)
            arcname = os.path.join("output", fname)
            logger.info("Adding %s (%.1f MB)...", arcname, os.path.getsize(fpath) / (1024 * 1024))
            zf.write(fpath, arcname=arcname)

        # 2. Documentation template
        doc_path = os.path.join(base_dir, "Documentation_template.md")
        logger.info("Adding Documentation_template.md...")
        zf.write(doc_path, arcname="Documentation_template.md")

        # 3. Code directory (all source code, README, requirements.txt, models)
        code_root = os.path.join(base_dir, "code", "business_entity_resolution")
        for root, dirs, files in os.walk(code_root):
            # Exclude __pycache__ and temporary files
            dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
            for f in files:
                if f.endswith((".pyc", ".pyo", ".tmp")):
                    continue
                file_full = os.path.join(root, f)
                rel_path = os.path.relpath(file_full, base_dir)
                logger.info("Adding %s...", rel_path)
                zf.write(file_full, arcname=rel_path)

    logger.info("Successfully packaged %s (Total size: %.1f MB)", zip_path, os.path.getsize(zip_path) / (1024 * 1024))
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Package ML Challenge 2026 submission.")
    parser.add_argument("--base-dir", default=".", help="Base project directory (student_resource)")
    parser.add_argument("--team-name", default="EntityResolvers", help="Team name for zip archive name")
    parser.add_argument("--output-dir", default=".", help="Destination directory for zip file")
    args = parser.parse_args()

    create_submission_zip(args.base_dir, args.team_name, args.output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Worker that polls Redis queue and processes generation tasks."""

import json
import os
import sys
import time
import redis
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")


def main():
    r = redis.from_url(REDIS_URL)
    print(f"Worker started, polling Redis at {REDIS_URL}")

    while True:
        task_key = r.brpoplpush("queue:todo", "queue:processing", timeout=10)
        if not task_key:
            continue

        print(f"Processing {task_key}")

        task = r.hgetall(task_key)
        if not task:
            r.lrem("queue:processing", 0, task_key)
            continue

        job_id = task.get(b"job_id", b"").decode()
        asset_id = task.get(b"asset_id", b"").decode()
        task_type = task.get(b"task_type", b"").decode()

        r.hset(task_key, "started_at", datetime.utcnow().isoformat())

        try:
            from vplib import VPLib
            from pipeline import Pipeline

            job_dir = Path(__file__).parent / "pipeline" / job_id
            manifest_path = job_dir / "asset_manifest.json"

            with open(manifest_path) as f:
                assets = json.load(f)

            asset = next((a for a in assets if a["asset_id"] == asset_id), None)
            if not asset:
                raise Exception(f"Asset {asset_id} not found")

            asset_type = asset["type"]
            name = asset.get("name", asset_id).lower().replace(" ", "_")

            context = (job_dir / "context.txt").read_text().strip() if (job_dir / "context.txt").exists() else ""
            year_hint = "1980s"
            if context:
                try:
                    p = Pipeline()
                    year_hint = p._derive_year_hint(context)
                except:
                    pass

            vplib = VPLib()

            if asset_type == "character":
                current_version = asset.get("current_version", 1)
                new_version = current_version + 1
                asset["current_version"] = new_version

                sheet_data = vplib.generate_character_sheet(
                    asset_id,
                    asset.get("name", asset_id),
                    asset.get("visual_description", ""),
                    year_hint,
                    asset.get("full_prompt", ""),
                )

                version_dir = job_dir / "assets" / "characters" / name / f"v{new_version}"
                version_dir.mkdir(parents=True, exist_ok=True)

                meta = {
                    "version": new_version,
                    "asset_id": asset_id,
                    "name": asset.get("name"),
                    "generated_at": datetime.utcnow().isoformat(),
                    "angles": {}
                }

                for angle, data in sheet_data.items():
                    data["image"].save(version_dir / f"{angle}.png")
                    meta["angles"][angle] = {"filename": f"{angle}.png", "prompt": data.get("prompt")}

                with open(version_dir / "meta.json", "w") as f:
                    json.dump(meta, f, indent=2)

                asset["has_gen"] = True

            else:
                prompt = asset.get("full_prompt") or f"{year_hint}, {asset.get('visual_description', '')}"
                image = vplib.generate_image(prompt)

                current_version = asset.get("current_version", 1)
                new_version = current_version + 1
                version_dir = job_dir / "assets" / asset_type + "s" / name / f"v{new_version}"
                version_dir.mkdir(parents=True, exist_ok=True)

                filename = "background.png" if asset_type == "background" else "object.png"
                image.save(version_dir / filename)

                meta = {
                    "version": new_version,
                    "asset_id": asset_id,
                    "prompt": prompt,
                    "generated_at": datetime.utcnow().isoformat()
                }
                with open(version_dir / "meta.json", "w") as f:
                    json.dump(meta, f, indent=2)

                asset["current_version"] = new_version
                asset["has_gen"] = True
                asset["source"] = "generated"

            with open(manifest_path, "w") as f:
                json.dump(assets, f, indent=2)

            r.hset(task_key, "state", "done")
            r.hset(task_key, "completed_at", datetime.utcnow().isoformat())
            print(f"Done {task_key}")

        except Exception as e:
            print(f"Error {task_key}: {e}")
            r.hset(task_key, "state", "failed")
            r.hset(task_key, "error", str(e))

        finally:
            r.lrem("queue:processing", 0, task_key)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Render loop for video generation using WanGPSession."""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "Wan2GP"))

from shared.api import init, WanGPSession


class RenderLoop:
    def __init__(self, config_path: str | Path | None = None):
        config_path = Path(config_path) if config_path else Path(__file__).parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        wan2gp_cfg = self.config.get("wan2gp", {})
        self.wan2gp_path = Path(wan2gp_cfg.get("path", 
                                 Path(__file__).parent.parent / "Wan2GP"))
        self.video_model = wan2gp_cfg.get("video_model", "ltx2_22B_distilled")
        self.image_model = wan2gp_cfg.get("image_model", "flux2_klein_9b")
        self.edit_model = wan2gp_cfg.get("edit_model", "flux2_klein_9b")
        
        self.output_dir = Path(__file__).parent / "renders"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("render_loop")
        self.session: WanGPSession | None = None

    def init_session(self) -> WanGPSession:
        if self.session is None:
            self.logger.info(f"Initializing WanGPSession with root: {self.wan2gp_path}")
            self.session = init(
                root=str(self.wan2gp_path),
                output_dir=str(self.output_dir),
                console_output=True,
            )
        return self.session

    def render_scene_package(self, scene_package_path: str | Path, max_retries: int = 3) -> dict[str, Any]:
        scene_package_path = Path(scene_package_path)
        job_id = scene_package_path.parent.parent.name

        with open(scene_package_path) as f:
            package = json.load(f)

        if package.get("status") == "rendered":
            self.logger.info(f"Scene {package['shot_id']} already rendered, skipping")
            return package

        retry_count = package.get("retry_count", 0)
        current_take = retry_count + 1

        session = self.init_session()

        settings = {
            "model_type": self.video_model,
            "prompt": package["full_prompt"],
            "resolution": "1280x704",
            "num_inference_steps": 8,
            "video_length": 97,
            "duration_seconds": package.get("duration_seconds", 4),
            "force_fps": 24,
        }

        self.logger.info(f"Rendering {package['shot_id']} (take {current_take}) with model {self.video_model}")

        try:
            job = session.submit_task(settings)

            for event in job.events.iter(timeout=0.2):
                if event.kind == "progress":
                    progress = event.data
                    self.logger.info(f"Progress: {progress.phase} - {progress.progress}%")
                elif event.kind == "preview":
                    if event.data.image is not None:
                        event.data.image.save(f"pipeline/{job_id}/preview_{package['shot_id']}.png")
                elif event.kind == "stream":
                    self.logger.debug(f"[{event.data.stream}] {event.data.text}")

            result = job.result()

            if result.success and result.generated_files:
                output_path = Path(result.generated_files[0])

                renders_dir = scene_package_path.parent.parent / "renders"
                renders_dir.mkdir(parents=True, exist_ok=True)

                final_path = renders_dir / f"{package['shot_id']}_take_{current_take}.mp4"
                if output_path.exists() and output_path != final_path:
                    import shutil
                    shutil.copy(output_path, final_path)

                package["status"] = "rendered"
                package["rendered_file"] = str(final_path)
                package["retry_count"] = retry_count
                package["last_error"] = None
            else:
                errors = [e.message for e in result.errors]
                self.logger.error(f"Render failed: {errors}")

                if retry_count < max_retries:
                    package["retry_count"] = retry_count + 1
                    package["status"] = "pending"
                    package["last_error"] = "; ".join(errors)
                else:
                    package["status"] = "failed"
                    package["last_error"] = "; ".join(errors)

        except Exception as e:
            self.logger.error(f"Render exception: {e}")
            if retry_count < max_retries:
                package["retry_count"] = retry_count + 1
                package["status"] = "pending"
                package["last_error"] = str(e)
            else:
                package["status"] = "failed"
                package["last_error"] = str(e)

        with open(scene_package_path, "w") as f:
            json.dump(package, f, indent=2)

        return package

    def render_job(self, job_id: str) -> dict[str, Any]:
        job_dir = Path(f"pipeline/{job_id}")
        scene_packages_dir = job_dir / "scene_packages"

        if not scene_packages_dir.exists():
            raise FileNotFoundError(f"No scene packages found for job {job_id}")

        results = {}
        for scene_pkg_file in sorted(scene_packages_dir.glob("shot_*.json")):
            self.logger.info(f"Rendering {scene_pkg_file.name}")
            result = self.render_scene_package(scene_pkg_file)
            results[result["shot_id"]] = result

        return results

    def mutate_prompt_for_retry(self, previous_prompt: str, timing_breakdown: list[dict]) -> str:
        return f"{previous_prompt}\n\nRephrase the timing breakdown only. Keep all asset references and harness identical."


def main():
    if len(sys.argv) < 2:
        print("Usage: python render.py <job_id>")
        sys.exit(1)

    job_id = sys.argv[1]
    render_loop = RenderLoop()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    results = render_loop.render_job(job_id)
    print(f"\nRender complete for job {job_id}")
    for shot_id, result in results.items():
        print(f"  {shot_id}: {result['status']}")


if __name__ == "__main__":
    main()

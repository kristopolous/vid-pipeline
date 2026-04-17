#!/usr/bin/env python3
"""FastAPI server for video harness - serves UI and render endpoints."""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from Wan2GP.shared.api import WanGPSession

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent / "Wan2GP"))

try:
    from shared.api import init, WanGPSession
    WAN2GP_AVAILABLE = True
except ImportError:
    WAN2GP_AVAILABLE = False
    WanGPSession = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("server")


WAN2GP_ROOT = Path(__file__).parent.parent / "Wan2GP"
RENDER_SESSION: WanGPSession | None = None
CONFIG_PATH = Path(__file__).parent / "config.json"

WAN2GP_URL = None
WAN2GP_CONFIG = {}


def load_config():
    global WAN2GP_URL, WAN2GP_CONFIG
    
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            WAN2GP_CONFIG = json.load(f)
        
        wan2gp_cfg = WAN2GP_CONFIG.get("wan2gp", {})
        WAN2GP_URL = wan2gp_cfg.get("url")
        
        if WAN2GP_URL:
            logger.info(f"Using Wan2GP at: {WAN2GP_URL}")
        else:
            logger.info("wan2gp.url not set in config - remote mode disabled")
    else:
        WAN2GP_CONFIG = {
            "text": {
                "model": "llama.cpp",
                "url": "http://localhost:11434",
                "key": None
            },
            "wan2gp": {
                "url": "",
                "video": "ltx-2",
                "audio": "ltx-2",
                "edit": "flux-2",
                "image": "flux-2"
            }
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(WAN2GP_CONFIG, f, indent=4)
        logger.info(f"Config file not found - created default at {CONFIG_PATH}")
        logger.info("Please edit config.json to set your wan2gp.url and other settings")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global RENDER_SESSION
    load_config()
    
    if WAN2GP_URL and WAN2GP_AVAILABLE:
        logger.info("Remote Wan2GP mode - not initializing local session")
    elif WAN2GP_AVAILABLE and os.environ.get("WGP_INITIALIZED", "false").lower() == "true":
        try:
            RENDER_SESSION = init(
                root=str(WAN2GP_ROOT),
                output_dir=str(Path(__file__).parent / "renders"),
                console_output=False,
            )
            logger.info("WanGPSession initialized and kept resident")
        except Exception as e:
            logger.warning(f"Could not initialize WanGPSession: {e}")
    
    yield
    
    if RENDER_SESSION is not None:
        RENDER_SESSION.close()
        logger.info("WanGPSession closed")


app = FastAPI(title="Video Harness", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline_static_path = Path(__file__).parent / "pipeline"
if pipeline_static_path.exists():
    app.mount("/pipeline", StaticFiles(directory=pipeline_static_path), name="pipeline")


class SubmitJobRequest(BaseModel):
    scene: str | None = None
    style: str | None = None
    objects: str | None = None
    characters: str | None = None
    backgrounds: str | None = None
    context: str | None = None


class RenderRequest(BaseModel):
    job_id: str
    shot_id: str | None = None


class UpdateAssetRequest(BaseModel):
    job_id: str
    asset_id: str
    description: str | None = None


class UpdateHarnessRequest(BaseModel):
    job_id: str
    harness: dict[str, Any]


def get_job_dir(job_id: str) -> Path:
    return Path(__file__).parent / "pipeline" / job_id


@app.get("/")
async def root():
    return FileResponse("review.html")


@app.get("/api/jobs")
async def list_jobs():
    pipeline_dir = Path(__file__).parent / "pipeline"
    if not pipeline_dir.exists():
        return []

    jobs = []
    for job_path in sorted(pipeline_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if job_path.is_dir():
            meta_path = job_path / "meta-info.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    jobs.append(json.load(f))
            else:
                jobs.append({"job_id": job_path.name, "stage": "unknown"})
    return jobs


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job_dir = get_job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    meta_path = job_dir / "meta-info.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {"job_id": job_id, "stage": "unknown"}

    response = {"job_id": job_id, "meta": meta}

    asset_manifest_path = job_dir / "asset_manifest.json"
    if asset_manifest_path.exists():
        with open(asset_manifest_path) as f:
            asset_manifest = json.load(f)
        for asset in asset_manifest:
            asset_type = asset.get("type", "")
            if asset_type in ["character", "object", "background"]:
                subdir = asset_type + "s"
                gen_path = job_dir / "assets" / subdir / "gen" / f"{asset['asset_id']}.png"
                web_path = job_dir / "assets" / subdir / "web" / f"{asset['asset_id']}.png"
                asset["has_gen"] = gen_path.exists()
                asset["has_web"] = web_path.exists()
        response["asset_manifest"] = asset_manifest
    else:
        asset_manifest = []

    for key, filename in [
        ("script", "script.txt"),
        ("style", "style.txt"),
        ("context", "context.txt"),
        ("action_items", "action_items.json"),
        ("shots", "shots.json"),
        ("harness", "harness.json"),
    ]:
        if key == "asset_manifest":
            continue
        file_path = job_dir / filename
        if file_path.exists():
            if filename.endswith(".json"):
                with open(file_path) as f:
                    response[key] = json.load(f)
            else:
                with open(file_path) as f:
                    response[key] = f.read()

    scene_packages = {}
    scene_dir = job_dir / "scene_packages"
    if scene_dir.exists():
        for sp_file in scene_dir.glob("*.json"):
            with open(sp_file) as f:
                sp = json.load(f)
                scene_packages[sp["shot_id"]] = sp
    response["scene_packages"] = scene_packages

    renders = {}
    renders_dir = job_dir / "renders"
    if renders_dir.exists():
        for render_file in renders_dir.glob("*.mp4"):
            renders[render_file.stem] = str(render_file)
    response["renders"] = renders

    return response


@app.post("/api/jobs")
async def submit_job(request: SubmitJobRequest):
    from pipeline import Pipeline

    pipeline = Pipeline()

    with open(Path(__file__).parent / "pipeline" / "temp_scene.txt", "w") as f:
        f.write(request.scene or "")

    with open(Path(__file__).parent / "pipeline" / "temp_style.txt", "w") as f:
        f.write(request.style or "")

    with open(Path(__file__).parent / "pipeline" / "temp_objects.txt", "w") as f:
        f.write(request.objects or "")

    with open(Path(__file__).parent / "pipeline" / "temp_characters.txt", "w") as f:
        f.write(request.characters or "")

    with open(Path(__file__).parent / "pipeline" / "temp_backgrounds.txt", "w") as f:
        f.write(request.backgrounds or "")

    with open(Path(__file__).parent / "pipeline" / "temp_context.txt", "w") as f:
        f.write(request.context or "")

    pipeline.load_input_files(
        scene_file=Path(__file__).parent / "pipeline" / "temp_scene.txt",
        style_file=Path(__file__).parent / "pipeline" / "temp_style.txt",
        objects_file=Path(__file__).parent / "pipeline" / "temp_objects.txt",
        characters_file=Path(__file__).parent / "pipeline" / "temp_characters.txt",
        backgrounds_file=Path(__file__).parent / "pipeline" / "temp_backgrounds.txt",
        context_file=Path(__file__).parent / "pipeline" / "temp_context.txt",
    )

    job_id = pipeline.run_full_pipeline()

    for f in ["temp_scene.txt", "temp_style.txt", "temp_objects.txt", "temp_characters.txt", "temp_backgrounds.txt", "temp_context.txt"]:
        temp_path = Path(__file__).parent / "pipeline" / f
        if temp_path.exists():
            temp_path.unlink()

    return {"job_id": job_id}


@app.post("/api/jobs/{job_id}/run-track1")
async def run_track1(job_id: str):
    from pipeline import Pipeline

    job_dir = get_job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    pipeline = Pipeline()
    success = pipeline.run_track1(job_id)

    if not success:
        raise HTTPException(status_code=500, detail="Track 1 failed")

    return {"status": "ok", "job_id": job_id}


@app.post("/api/jobs/{job_id}/build-scenes")
async def build_scenes(job_id: str):
    from pipeline import Pipeline

    job_dir = get_job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    pipeline = Pipeline()
    success = pipeline.build_scene_packages(job_id)

    if not success:
        raise HTTPException(status_code=500, detail="Scene building failed")

    return {"status": "ok", "job_id": job_id}


@app.post("/api/render")
async def render_scene(request: RenderRequest):
    if not WAN2GP_AVAILABLE:
        raise HTTPException(status_code=503, detail="Wan2GP not available")

    job_dir = get_job_dir(request.job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    scene_dir = job_dir / "scene_packages"
    if request.shot_id:
        scene_files = [scene_dir / f"{request.shot_id}.json"]
    else:
        scene_files = list(scene_dir.glob("*.json"))

    results = []
    for scene_file in scene_files:
        with open(scene_file) as f:
            package = json.load(f)

        if package.get("status") == "rendered":
            results.append({"shot_id": package["shot_id"], "status": "already_rendered"})
            continue

        retry_count = package.get("retry_count", 0)

        settings = {
            "model_type": "ltx2_22B_distilled",
            "prompt": package["full_prompt"],
            "resolution": "1280x704",
            "num_inference_steps": 8,
            "video_length": 97,
            "duration_seconds": package.get("duration_seconds", 4),
            "force_fps": 24,
        }

        try:
            session = RENDER_SESSION or init(
                root=str(WAN2GP_ROOT),
                output_dir=str(job_dir / "renders"),
                console_output=False,
            )

            job = session.submit_task(settings)

            for event in job.events.iter(timeout=0.2):
                if event.kind == "progress":
                    logger.info(f"Progress: {event.data.phase} - {event.data.progress}%")

            result = job.result()

            if result.success and result.generated_files:
                renders_dir = job_dir / "renders"
                renders_dir.mkdir(parents=True, exist_ok=True)

                output_path = Path(result.generated_files[0])
                final_path = renders_dir / f"{package['shot_id']}_take_{retry_count + 1}.mp4"

                if output_path.exists() and output_path != final_path:
                    import shutil
                    shutil.copy(output_path, final_path)

                package["status"] = "rendered"
                package["rendered_file"] = str(final_path)
                package["last_error"] = None
            else:
                errors = [e.message for e in result.errors]
                package["status"] = "failed"
                package["last_error"] = "; ".join(errors)

        except Exception as e:
            logger.error(f"Render error: {e}")
            package["status"] = "failed"
            package["last_error"] = str(e)

        with open(scene_file, "w") as f:
            json.dump(package, f, indent=2)

        results.append({"shot_id": package["shot_id"], "status": package["status"]})

    return {"results": results}


@app.post("/api/jobs/{job_id}/assets/{asset_id}")
async def update_asset(job_id: str, asset_id: str, request: UpdateAssetRequest):
    job_dir = get_job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    manifest_path = job_dir / "asset_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Asset manifest not found")

    with open(manifest_path) as f:
        assets = json.load(f)

    for asset in assets:
        if asset["asset_id"] == asset_id:
            if request.description:
                asset["visual_description"] = request.description
            break
    else:
        raise HTTPException(status_code=404, detail="Asset not found")

    with open(manifest_path, "w") as f:
        json.dump(assets, f, indent=2)

    for sp_file in (job_dir / "scene_packages").glob("shot_*.json"):
        with open(sp_file) as f:
            sp = json.load(f)

        for a in sp.get("assets", []):
            if a["asset_id"] == asset_id:
                sp["status"] = "stale"
                break

        with open(sp_file, "w") as f:
            json.dump(sp, f, indent=2)

    return {"status": "ok", "asset_id": asset_id}


@app.post("/api/jobs/{job_id}/assets/{asset_id}/regenerate")
async def regenerate_asset(job_id: str, asset_id: str):
    job_dir = get_job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    manifest_path = job_dir / "asset_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Asset manifest not found")

    with open(manifest_path) as f:
        assets = json.load(f)

    asset = None
    for a in assets:
        if a["asset_id"] == asset_id:
            asset = a
            break

    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    asset_type = asset["type"]
    subdir = asset_type + "s"

    gen_path = job_dir / "assets" / subdir / "gen" / f"{asset_id}.png"
    web_path = job_dir / "assets" / subdir / "web" / f"{asset_id}.png"

    if gen_path.exists():
        gen_path.unlink()
    if web_path.exists():
        web_path.unlink()

    asset["has_gen"] = False
    asset["has_web"] = False
    asset["needs_regen"] = True

    with open(manifest_path, "w") as f:
        json.dump(assets, f, indent=2)

    for sp_file in (job_dir / "scene_packages").glob("shot_*.json"):
        with open(sp_file) as f:
            sp = json.load(f)
        for a in sp.get("assets", []):
            if a["asset_id"] == asset_id:
                sp["status"] = "stale"
                break
        with open(sp_file, "w") as f:
            json.dump(sp, f, indent=2)

    return {"status": "ok", "asset_id": asset_id, "message": "Asset marked for regeneration"}


@app.post("/api/jobs/{job_id}/assets/{asset_id}/upload")
async def upload_asset_image(job_id: str, asset_id: str, file: UploadFile):
    job_dir = get_job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    manifest_path = job_dir / "asset_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Asset manifest not found")

    with open(manifest_path) as f:
        assets = json.load(f)

    asset = None
    for a in assets:
        if a["asset_id"] == asset_id:
            asset = a
            break

    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    asset_type = asset["type"]
    subdir = asset_type + "s"
    gen_path = job_dir / "assets" / subdir / "gen" / f"{asset_id}.png"

    gen_path.parent.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    with open(gen_path, "wb") as f:
        f.write(content)

    asset["has_gen"] = True
    asset["has_web"] = False
    asset["source"] = "uploaded"

    with open(manifest_path, "w") as f:
        json.dump(assets, f, indent=2)

    for sp_file in (job_dir / "scene_packages").glob("shot_*.json"):
        with open(sp_file) as f:
            sp = json.load(f)
        for a in sp.get("assets", []):
            if a["asset_id"] == asset_id:
                sp["status"] = "stale"
                break
        with open(sp_file, "w") as f:
            json.dump(sp, f, indent=2)

    return {"status": "ok", "asset_id": asset_id, "message": "Image uploaded successfully"}


@app.post("/api/jobs/{job_id}/harness")
async def update_harness(job_id: str, request: UpdateHarnessRequest):
    job_dir = get_job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    harness_path = job_dir / "harness.json"
    with open(harness_path, "w") as f:
        json.dump(request.harness, f, indent=2)

    for sp_file in (job_dir / "scene_packages").glob("shot_*.json"):
        with open(sp_file) as f:
            sp = json.load(f)
        sp["status"] = "stale"
        sp["harness"] = request.harness
        with open(sp_file, "w") as f:
            json.dump(sp, f, indent=2)

    return {"status": "ok"}


@app.get("/api/config")
async def get_config():
    load_config()
    return WAN2GP_CONFIG


@app.post("/api/config")
async def update_config(request: dict):
    global WAN2GP_CONFIG, WAN2GP_URL
    WAN2GP_CONFIG = request
    wan2gp_cfg = WAN2GP_CONFIG.get("wan2gp", {})
    WAN2GP_URL = wan2gp_cfg.get("url")
    with open(CONFIG_PATH, "w") as f:
        json.dump(WAN2GP_CONFIG, f, indent=4)
    logger.info(f"Config updated. Wan2GP URL: {WAN2GP_URL}")
    return {"status": "ok", "wan2gp_url": WAN2GP_URL}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
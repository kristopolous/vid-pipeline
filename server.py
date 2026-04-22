#!/usr/bin/env python3
"""FastAPI server for video harness - serves UI and render endpoints."""

from __future__ import annotations

import json
import logging
import os
import subprocess
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

from vplib import VPLib
vplib = VPLib()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("server")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
_queue_redis = None


def get_redis():
    global _queue_redis
    if _queue_redis is None:
        try:
            import redis
            _queue_redis = redis.from_url(REDIS_URL)
            _queue_redis.ping()
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            _queue_redis = False
    return _queue_redis if _queue_redis else None


def enqueue_task(job_id: str, asset_id: str, task_type: str):
    """Add task to Redis queue."""
    r = get_redis()
    if not r:
        return

    from datetime import datetime
    task_key = f"task:{job_id}:{asset_id}"
    r.hset(task_key, mapping={
        "state": "todo",
        "task_type": task_type,
        "created_at": datetime.utcnow().isoformat(),
        "job_id": job_id,
        "asset_id": asset_id
    })
    r.lpush("queue:todo", task_key)
    logger.info(f"Enqueued {task_key}")


def add_text_label(input_path: Path, output_path: Path, label: str) -> None:
    """Add label above image using ImageMagick."""
    label_clean = label.strip()
    cmd = [
        "convert",
        "-size", "800x50",
        "-background", "black",
        "-fill", "white",
        "-font", "DejaVu-Sans",
        "-pointsize", "32",
        "-gravity", "center",
        f"label:{label_clean}",
        str(input_path),
        "-append",
        str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logger.warning(f"ImageMagick montage failed: {e}")
        if input_path != output_path:
            input_path.rename(output_path)


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
    description: str | None = None
    full_prompt: str | None = None
    caption: str | None = None


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

        try:
            prompt = package["full_prompt"]
            resolution = package.get("resolution", "1280x704")
            width, height = map(int, resolution.split("x"))
            duration = int(package.get("duration_seconds", 4))
            num_frames = duration * 24
            negative = package.get("harness", {}).get("negative_prompt", "anime, cartoon, low quality, distorted")

            logger.info(f"Rendering shot {package['shot_id']} with LTX-2: {num_frames} frames, {resolution}")

            frames = vplib.render_video(
                prompt=prompt,
                negative_prompt=negative,
                width=width,
                height=height,
                num_frames=num_frames,
            )

            if frames:
                renders_dir = job_dir / "renders"
                renders_dir.mkdir(parents=True, exist_ok=True)
                final_path = renders_dir / f"{package['shot_id']}_take_{retry_count + 1}.mp4"

                import numpy as np
                import tempfile

                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                    tmp_path = tmp.name

                try:
                    from moviepy import ImageSequenceClip
                    clip = ImageSequenceClip(list(frames), fps=24)
                    clip.write_videofile(tmp_path, codec='libx264', audio=False, logger=None)
                except ImportError:
                    import cv2
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(tmp_path, fourcc, 24, (width, height))
                    for frame in frames:
                        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                        out.write(frame_bgr)
                    out.release()

                import shutil
                shutil.copy(tmp_path, final_path)
                Path(tmp_path).unlink()

                package["status"] = "rendered"
                package["rendered_file"] = str(final_path)
                package["last_error"] = None
            else:
                package["status"] = "failed"
                package["last_error"] = "No frames generated"

        except Exception as e:
            logger.error(f"Render error: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
            if request.description is not None:
                asset["visual_description"] = request.description
            if request.full_prompt is not None:
                asset["full_prompt"] = request.full_prompt
            if request.caption is not None:
                asset["caption"] = request.caption
            break
    else:
        raise HTTPException(status_code=404, detail="Asset not found")

    with open(manifest_path, "w") as f:
        json.dump(assets, f, indent=2)

    for sp_file in (job_dir / "scene_packages").glob("*.json"):
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

    if asset_type == "character":
        current_version = asset.get("current_version", 1)
        new_version = current_version + 1
        asset["current_version"] = new_version
        asset["preferred_version"] = asset.get("preferred_version", current_version)

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

    return {"status": "ok", "asset_id": asset_id, "message": "Asset marked for regeneration", "version": asset.get("current_version", 1)}


@app.get("/pipeline/{job_id}/assets/{full_path:path}")
async def serve_asset_image(job_id: str, full_path: str):
    """Serve asset image from versioned directory."""
    import time

    job_dir = Path(__file__).parent / "pipeline" / job_id
    asset_path = job_dir / "assets" / full_path

    timeout = 300
    start = time.time()
    while time.time() - start < timeout:
        if asset_path.exists() and asset_path.stat().st_size > 100:
            return FileResponse(asset_path)
        time.sleep(0.5)

    loading_path = Path(__file__).parent / "loading.png"
    if loading_path.exists():
        return FileResponse(loading_path)

    return Response(status_code=404, content="Image not found")


@app.post("/api/jobs/{job_id}/assets/{asset_id}/regen-now")
async def regenerate_asset_now(job_id: str, asset_id: str):
    """Actually regenerate an asset now with proper versioning."""
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
    name = asset.get("name", asset_id).lower().replace(" ", "_")

    context_path = job_dir / "context.txt"
    context = context_path.read_text().strip() if context_path.exists() else ""
    year_hint = "1980s"
    if context:
        try:
            from pipeline import Pipeline
            p = Pipeline()
            year_hint = p._derive_year_hint(context)
        except:
            pass

    logger.info(f"Generating {asset_id} ({asset_type})")

    try:
        if asset_type == "character":
            current_version = asset.get("current_version", 1)
            new_version = current_version + 1
            asset["current_version"] = new_version
            asset["preferred_version"] = asset.get("preferred_version", current_version)

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
                "description": asset.get("visual_description"),
                "year_hint": year_hint,
                "full_prompt": asset.get("full_prompt"),
                "generated_at": __import__("datetime").datetime.now().isoformat(),
                "angles": {}
            }

            for angle, data in sheet_data.items():
                image = data["image"]
                png_path = version_dir / f"{angle}.png"
                image.save(png_path)
                meta["angles"][angle] = {
                    "filename": f"{angle}.png",
                    "prompt": data.get("prompt"),
                    "model": data.get("model")
                }
                logger.info(f"Saved {png_path}")

            meta_path = version_dir / "meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            asset["character_sheet"] = {"version": new_version, "path": f"characters/{name}/v{new_version}"}
            asset["has_gen"] = True
            asset["source"] = "generated"

        elif asset_type == "background":
            current_version = asset.get("current_version", 1)
            new_version = current_version + 1

            version_dir = job_dir / "assets" / "backgrounds" / name / f"v{new_version}"
            version_dir.mkdir(parents=True, exist_ok=True)

            prompt = asset.get("full_prompt") or f"{year_hint}, {asset.get('visual_description', '')}"
            image = vplib.generate_image(prompt)

            png_path = version_dir / "background.png"
            image.save(png_path)

            meta = {
                "version": new_version,
                "asset_id": asset_id,
                "name": asset.get("name"),
                "description": asset.get("visual_description"),
                "prompt": prompt,
                "generated_at": __import__("datetime").datetime.now().isoformat()
            }
            with open(version_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            asset["current_version"] = new_version
            asset["preferred_version"] = asset.get("preferred_version", current_version)
            asset["has_gen"] = True
            asset["source"] = "generated"

        else:
            current_version = asset.get("current_version", 1)
            new_version = current_version + 1

            version_dir = job_dir / "assets" / "objects" / name / f"v{new_version}"
            version_dir.mkdir(parents=True, exist_ok=True)

            prompt = asset.get("full_prompt") or f"{year_hint}, {asset.get('visual_description', '')}"
            image = vplib.generate_image(prompt)

            png_path = version_dir / "object.png"
            image.save(png_path)

            meta = {
                "version": new_version,
                "asset_id": asset_id,
                "name": asset.get("name"),
                "description": asset.get("visual_description"),
                "prompt": prompt,
                "generated_at": __import__("datetime").datetime.now().isoformat()
            }
            with open(version_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            asset["current_version"] = new_version
            asset["preferred_version"] = asset.get("preferred_version", current_version)
            asset["has_gen"] = True
            asset["source"] = "generated"

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        asset["has_gen"] = False
        asset["last_error"] = str(e)

    with open(manifest_path, "w") as f:
        json.dump(assets, f, indent=2)

    return {"status": "ok", "asset_id": asset_id, "generated": asset.get("has_gen", False), "message": "Task enqueued"}


@app.get("/api/queue/status")
async def get_queue_status():
    """Get current queue status."""
    r = get_redis()
    if not r:
        return {"queue": [], "error": "Redis not available"}

    jobs = []
    for key in r.lrange("queue:todo", 0, -1):
        data = r.hgetall(key)
        if data:
            jobs.append(dict(data))

    processing = []
    for key in r.lrange("queue:processing", 0, -1):
        data = r.hgetall(key)
        if data:
            processing.append(dict(data))

    return {"todo": jobs, "processing": processing}


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
    name = asset.get("name", asset_id).lower().replace(" ", "_")
    current_version = asset.get("current_version", 1)

    if asset_type == "character":
        version_dir = job_dir / "assets" / "characters" / name / f"v{current_version}"
    elif asset_type == "background":
        version_dir = job_dir / "assets" / "backgrounds" / name / f"v{current_version}"
    else:
        version_dir = job_dir / "assets" / "objects" / name / f"v{current_version}"

    version_dir.mkdir(parents=True, exist_ok=True)

    content = await file.read()
    filename = "headshot.png" if asset_type == "character" else "image.png"
    img_path = version_dir / filename
    with open(img_path, "wb") as f:
        f.write(content)

    meta = {
        "version": current_version,
        "asset_id": asset_id,
        "source": "uploaded",
        "uploaded_at": __import__("datetime").datetime.now().isoformat()
    }
    with open(version_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    asset["has_gen"] = True
    asset["has_web"] = False
    asset["source"] = "uploaded"

    with open(manifest_path, "w") as f:
        json.dump(assets, f, indent=2)

    return {"status": "ok", "asset_id": asset_id, "message": f"Uploaded to {version_dir}"}


@app.post("/api/jobs/{job_id}/assets/{asset_id}/url")
async def set_asset_url(job_id: str, asset_id: str, request: dict):
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

    url = request.get("url", "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    asset_type = asset["type"]
    subdir = asset_type + "s"
    gen_path = job_dir / "assets" / subdir / "gen" / f"{asset_id}.png"
    gen_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import requests
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/png,image/jpeg,*/*;q=0.8",
            "Referer": url,
        }
        img_response = requests.get(url, timeout=15, headers=headers)
        if img_response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image: {img_response.status_code}")

        from PIL import Image
        from io import BytesIO
        img = Image.open(BytesIO(img_response.content)).convert("RGB")
        img.save(gen_path)

        if asset.get("name"):
            add_text_label(gen_path, gen_path, asset["name"])

        asset["has_gen"] = True
        asset["has_web"] = False
        asset["source"] = "url"

        with open(manifest_path, "w") as f:
            json.dump(assets, f, indent=2)

        for sp_file in (job_dir / "scene_packages").glob("*.json"):
            with open(sp_file) as f:
                sp = json.load(f)
            for sp_asset in sp.get("assets", []):
                if sp_asset["asset_id"] == asset_id:
                    sp["status"] = "stale"
                    break
            with open(sp_file, "w") as f:
                json.dump(sp, f, indent=2)

        return {"status": "ok", "asset_id": asset_id, "message": "Image URL set successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/jobs/{job_id}/scene_packages/{shot_id}")
async def update_scene_package(job_id: str, shot_id: str, request: dict):
    job_dir = get_job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    scene_file = job_dir / "scene_packages" / f"{shot_id}.json"
    if not scene_file.exists():
        raise HTTPException(status_code=404, detail="Scene package not found")

    with open(scene_file, "w") as f:
        json.dump(request, f, indent=2)

    return {"status": "ok", "shot_id": shot_id}


@app.post("/api/jobs/{job_id}/scene_packages/{shot_id}/composite")
async def regenerate_composite(job_id: str, shot_id: str):
    job_dir = get_job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    scene_file = job_dir / "scene_packages" / f"{shot_id}.json"
    if not scene_file.exists():
        raise HTTPException(status_code=404, detail="Scene package not found")

    with open(scene_file) as f:
        package = json.load(f)

    assets_path = job_dir / "asset_manifest.json"
    if not assets_path.exists():
        raise HTTPException(status_code=400, detail="No asset manifest")

    assets = json.loads(assets_path.read_text())

    for asset in assets:
        asset_type = asset.get("type", "")
        if asset_type in ["character", "object", "background"]:
            subdir = asset_type + "s"
            gen_path = job_dir / "assets" / subdir / "gen" / f"{asset['asset_id']}.png"
            web_path = job_dir / "assets" / subdir / "web" / f"{asset['asset_id']}.png"
            asset["has_gen"] = gen_path.exists()
            asset["has_web"] = web_path.exists()

    try:
        composite_path = vplib.composite_scene_image(job_dir, package, assets)
    except Exception as e:
        logger.error(f"Composite failed: {e}")
        raise HTTPException(status_code=500, detail=f"Composite failed: {e}")

    if composite_path:
        rel_path = composite_path.relative_to(job_dir)
        package["keyframe_image"] = str(rel_path)
        logger.info(f"Saving composite: {rel_path}")
        with open(scene_file, "w") as f:
            json.dump(package, f, indent=2)
        return {"status": "ok", "keyframe_image": str(rel_path)}
    else:
        raise HTTPException(status_code=500, detail="Failed to generate composite - no assets found")


@app.get("/api/jobs/{job_id}/scene_packages/{shot_id}/download")
async def download_shot_zip(job_id: str, shot_id: str):
    import zipfile
    import io

    job_dir = get_job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    scene_file = job_dir / "scene_packages" / f"{shot_id}.json"
    if not scene_file.exists():
        raise HTTPException(status_code=404, detail="Scene not found")

    with open(scene_file) as f:
        package = json.load(f)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        zf.writestr(f"shot_{shot_id}.json", json.dumps(package, indent=2))

        if package.get("keyframe_image"):
            kf_path = job_dir / package["keyframe_image"]
            if kf_path.exists():
                zf.writestr(f"keyframe.png", kf_path.read_bytes())

        for asset_ref in package.get("assets", []):
            asset_id = asset_ref["asset_id"]
            for atype in ["characters", "objects", "backgrounds"]:
                for skind in ["gen", "web"]:
                    img_path = job_dir / "assets" / atype / skind / f"{asset_id}.png"
                    if img_path.exists():
                        zf.writestr(f"assets/{asset_id}_{skind}.png", img_path.read_bytes())
                        break

    buf.seek(0)
    from fastapi.responses import Response
    return Response(content=buf.read(), media_type="application/zip", headers={"Content-Disposition": f"attachment; filename=shot_{shot_id}.zip"})


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


@app.post("/api/jobs/{job_id}/assets/{asset_id}/select-version")
async def select_preferred_version(job_id: str, asset_id: str, request: dict):
    job_dir = get_job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    manifest_path = job_dir / "asset_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Asset manifest not found")

    with open(manifest_path) as f:
        assets = json.load(f)

    version = request.get("version")
    if version is None:
        raise HTTPException(status_code=400, detail="Version required")

    for asset in assets:
        if asset["asset_id"] == asset_id:
            asset["preferred_version"] = version
            break
    else:
        raise HTTPException(status_code=404, detail="Asset not found")

    with open(manifest_path, "w") as f:
        json.dump(assets, f, indent=2)

    return {"status": "ok", "asset_id": asset_id, "preferred_version": version}


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
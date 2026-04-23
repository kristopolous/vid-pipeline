#!/usr/bin/env python3
"""Worker that subscribes to Redis pub/sub for task notifications.

Architecture:
- Subscribes to "queue:new_task" Redis pub/sub channel
- On message: processes task, updates Redis hash for observability
- Multiple workers can run in parallel picking up different jobs
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import redis

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("worker")

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")

# Load config for Brave API and remote APIs
CONFIG_PATH = Path(__file__).parent / "config.json"
config = {}
if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as f:
        config = json.load(f)


def process_character(job_dir: Path, asset: dict, vplib) -> dict:
    """Generate character sheet with proper headshot→img2img flow.
    
    Returns dict with:
    - version_dir: Path to version directory
    - meta: metadata dict
    """
    asset_id = asset["asset_id"]
    name = asset.get("name", asset_id).lower().replace(" ", "_")
    description = asset.get("visual_description", "")
    full_prompt = asset.get("full_prompt", "")
    year_hint = asset.get("year_hint", "1980s")
    
    current_version = asset.get("current_version", 1)
    new_version = current_version + 1
    
    version_dir = job_dir / "assets" / "characters" / name / f"v{new_version}"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    # STEP 1: Generate HEADSHOT first - this is the identity source
    headshot_prompt = full_prompt or f"{year_hint}, realistic full color portrait photograph, {description}"
    logger.info(f"[{asset_id}] Generating HEADSHOT: {headshot_prompt[:60]}...")
    
    headshot = vplib.generate_image(headshot_prompt)
    if not headshot:
        # Fallback to web search
        logger.warning(f"[{asset_id}] Generation failed, trying web search")
        headshot = vplib.generate_image_fallback(f"{year_hint} {description}")
    
    if headshot:
        headshot.save(version_dir / "headshot.png")
        source = "flux2-klein-9B"
    else:
        logger.error(f"[{asset_id}] Could not generate headshot at all!")
        return None
    
    # STEP 2: Generate left/right/full using headshot as ref with img2img
    angles = {
        "left": f"obtain the left profile view of this character facing left",
        "right": f"obtain the right profile view of this character facing right", 
        "full": f"obtain the full body shot of this character standing, neutral background",
    }
    
    meta = {
        "version": new_version,
        "asset_id": asset_id,
        "name": asset.get("name"),
        "description": description,
        "year_hint": year_hint,
        "generated_at": datetime.utcnow().isoformat(),
        "source": source,
        "headshot": {
            "filename": "headshot.png",
            "prompt": headshot_prompt,
            "source": source
        },
        "angles": {}
    }
    
    for angle_name, angle_prompt in angles.items():
        logger.info(f"[{asset_id}] Generating {angle_name} from headshot...")
        
        # Use headshot as ref_image for img2img
        img = vplib.generate_image(angle_prompt, ref_image=headshot)
        
        if img:
            img.save(version_dir / f"{angle_name}.png")
            meta["angles"][angle_name] = {
                "filename": f"{angle_name}.png",
                "prompt": angle_prompt,
                "ref": "headshot.png",
                "source": "img2img"
            }
            logger.info(f"[{asset_id}] Saved {angle_name}.png")
        else:
            # If img2img fails, just copy headshot as fallback
            logger.warning(f"[{asset_id}] {angle_name} generation failed, using headshot")
            meta["angles"][angle_name] = {
                "filename": f"{angle_name}.png",
                "prompt": angle_prompt,
                "source": "fallback_headshot",
                "note": "generation failed, used headshot"
            }
    
    return {"version_dir": version_dir, "meta": meta}


def process_background(job_dir: Path, asset: dict, vplib) -> dict:
    """Generate background image."""
    asset_id = asset["asset_id"]
    name = asset.get("name", asset_id).lower().replace(" ", "_")
    year_hint = asset.get("year_hint", "1980s")
    
    current_version = asset.get("current_version", 1)
    new_version = current_version + 1
    
    version_dir = job_dir / "assets" / "backgrounds" / name / f"v{new_version}"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    prompt = asset.get("full_prompt") or f"{year_hint}, {asset.get('visual_description', '')}"
    logger.info(f"[{asset_id}] Generating background: {prompt[:60]}...")
    
    image = vplib.generate_image(prompt)
    if not image:
        image = vplib.generate_image_fallback(prompt)
    
    if image:
        image.save(version_dir / "background.png")
        source = "generated"
    else:
        source = "none"
        logger.error(f"[{asset_id}] Could not generate background!")
    
    meta = {
        "version": new_version,
        "asset_id": asset_id,
        "name": asset.get("name"),
        "prompt": prompt,
        "generated_at": datetime.utcnow().isoformat(),
        "source": source
    }
    
    return {"version_dir": version_dir, "meta": meta}


def process_object(job_dir: Path, asset: dict, vplib) -> dict:
    """Generate object image."""
    asset_id = asset["asset_id"]
    name = asset.get("name", asset_id).lower().replace(" ", "_")
    year_hint = asset.get("year_hint", "1980s")
    
    current_version = asset.get("current_version", 1)
    new_version = current_version + 1
    
    version_dir = job_dir / "assets" / "objects" / name / f"v{new_version}"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    prompt = asset.get("full_prompt") or f"{year_hint}, {asset.get('visual_description', '')}"
    logger.info(f"[{asset_id}] Generating object: {prompt[:60]}...")
    
    image = vplib.generate_image(prompt)
    if not image:
        image = vplib.generate_image_fallback(prompt)
    
    if image:
        image.save(version_dir / "object.png")
        source = "generated"
    else:
        source = "none"
        logger.error(f"[{asset_id}] Could not generate object!")
    
    meta = {
        "version": new_version,
        "asset_id": asset_id,
        "name": asset.get("name"),
        "prompt": prompt,
        "generated_at": datetime.utcnow().isoformat(),
        "source": source
    }
    
    return {"version_dir": version_dir, "meta": meta}


def main():
    """Main worker loop - subscribes to Redis and processes jobs."""
    r = redis.from_url(REDIS_URL)
    pubsub = r.pubsub()
    pubsub.psubscribe("queue:new_task")
    print(f"Worker started, subscribed to queue:new_task")
    
    # Keep worker running
    while True:
        try:
            # Wait for message with timeout so we can check periodically
            message = pubsub.get_message(timeout=5)
            
            if message and message["type"] == "pmessage":
                task_key = message["data"].decode()
                print(f"Received task: {task_key}")
                
                task = r.hgetall(task_key)
                if not task:
                    print(f"No task data for {task_key}")
                    continue
                
                job_id = task.get(b"job_id", b"").decode()
                asset_id = task.get(b"asset_id", b"").decode()
                task_type = task.get(b"task_type", b"").decode()
                
                # Update task status to processing
                r.hset(task_key, "started_at", datetime.utcnow().isoformat())
                r.hset(task_key, "state", "processing")
                
                try:
                    from vplib import VPLib
                    
                    job_dir = Path(__file__).parent / "pipeline" / job_id
                    manifest_path = job_dir / "asset_manifest.json"
                    
                    if not manifest_path.exists():
                        raise Exception(f"Manifest not found: {manifest_path}")
                    
                    with open(manifest_path) as f:
                        assets = json.load(f)
                    
                    asset = next((a for a in assets if a["asset_id"] == asset_id), None)
                    if not asset:
                        raise Exception(f"Asset {asset_id} not found")
                    
                    asset_type = asset["type"]
                    
                    # Derive year_hint from context
                    context = (job_dir / "context.txt").read_text().strip() if (job_dir / "context.txt").exists() else ""
                    asset["year_hint"] = "1980s"  # TODO: use Pipeline._derive_year_hint when needed
                    
                    vplib = VPLib(config)
                    
                    # Process based on type
                    if asset_type == "character":
                        result = process_character(job_dir, asset, vplib)
                    elif asset_type == "background":
                        result = process_background(job_dir, asset, vplib)
                    else:
                        result = process_object(job_dir, asset, vplib)
                    
                    if result:
                        # Write meta.json
                        with open(result["version_dir"] / "meta.json", "w") as f:
                            json.dump(result["meta"], f, indent=2)
                        
                        # Update manifest
                        asset["current_version"] = result["meta"]["version"]
                        asset["has_gen"] = result["meta"].get("source") != "none"
                        
                        with open(manifest_path, "w") as f:
                            json.dump(assets, f, indent=2)
                        
                        r.hset(task_key, "state", "done")
                        r.hset(task_key, "completed_at", datetime.utcnow().isoformat())
                        print(f"Done {task_key}")
                    else:
                        r.hset(task_key, "state", "failed")
                        r.hset(task_key, "error", "Generation returned no result")
                        
                except Exception as e:
                    import traceback
                    print(f"Error {task_key}: {e}")
                    print(traceback.format_exc())
                    r.hset(task_key, "state", "failed")
                    r.hset(task_key, "error", str(e))
            else:
                # No message, just continue loop - this keeps us subscribed
                time.sleep(0.1)
                
        except Exception as e:
            print(f"Worker loop error: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()
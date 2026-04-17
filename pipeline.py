#!/usr/bin/env python3
"""Video Harness Pipeline - Converts prose to scene packages for video generation."""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import requests
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image

CONFIG_PATH = Path(__file__).parent / "config.json"


@dataclass
class Character:
    asset_id: str
    name: str
    description: str


@dataclass
class Object:
    asset_id: str
    name: str
    description: str
    scene_description: str = ""


@dataclass
class Background:
    asset_id: str
    name: str
    description: str


class Pipeline:
    def __init__(self, config_path: Path | None = None):
        self.config = self._load_config(config_path or CONFIG_PATH)
        self.text_config = self.config.get("text", {})
        self.wan2gp_config = self.config.get("wan2gp", {})
        self._setup_logging()
        self.characters: list[Character] = []
        self.objects: list[Object] = []
        self.backgrounds: list[Background] = []
        self.style = ""
        self.context = ""
        self.scene = ""

    def _load_config(self, path: Path) -> dict:
        if path.exists():
            with open(path) as f:
                return json.load(f)
        
        default_config = {
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
        
        with open(path, "w") as f:
            json.dump(default_config, f, indent=4)
        
        self.logger.info(f"Config file not found - created default at {path}")
        self.logger.info("Please edit config.json to set your settings")
        
        return default_config

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger("pipeline")

    def _llm_call(self, system_prompt: str, user_prompt: str) -> str:
        url = self.text_config.get("url", "http://localhost:11434")
        model = self.text_config.get("model", "llama.cpp")
        api_key = self.text_config.get("key")

        full_url = f"{url}/v1/chat/completions"
        self.logger.info(f"LLM Request: URL={full_url}, model={model}")

        if OpenAI is None:
            raise RuntimeError("OpenAI library not installed. Run: pip install openai")

        try:
            client = OpenAI(
                base_url=f"{url}/v1",
                api_key=api_key or "not-needed",
            )

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                stream=False,
            )

            content = response.choices[0].message.content or ""
            
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            return content

        except Exception as e:
            self.logger.error(f"LLM call failed: URL={full_url}, model={model}, error={type(e).__name__}: {e}")
            raise

    def _log_pass(self, job_id: str, pass_name: str, input_hash: str, output: str, output_hash: str):
        log_dir = Path(f"pipeline/{job_id}/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "debug.log"

        with open(log_file, "a") as f:
            f.write(f"{datetime.now().isoformat()} | {pass_name} | input_hash={input_hash} | output_hash={output_hash}\n")
            f.write(f"Output:\n{output}\n")
            f.write("-" * 80 + "\n")

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _parse_character_line(self, line: str) -> Character | None:
        match = re.match(r"(.+?)\s+is\s+(.+)", line.strip())
        if match:
            name = match.group(1).strip()
            description = match.group(2).strip()
            asset_id = f"char_{name.lower().replace(' ', '_')}"
            return Character(asset_id=asset_id, name=name, description=description)
        return None

    def _parse_object_line(self, line: str) -> Object | None:
        match = re.match(r"(.+?)\s+is\s+(.+)", line.strip())
        if match:
            name = match.group(1).strip()
            rest = match.group(2).strip()
            if " | " in rest:
                scene_desc, obj_desc = rest.split(" | ", 1)
                scene_desc = scene_desc.strip()
                obj_desc = obj_desc.strip()
            else:
                scene_desc = ""
                obj_desc = rest
            asset_id = f"obj_{name.lower().replace(' ', '_')}"
            return Object(asset_id=asset_id, name=name, description=obj_desc, scene_description=scene_desc)
        return None

    def _parse_background_line(self, line: str) -> Background | None:
        match = re.match(r"(.+?)\s+is\s+(.+)", line.strip())
        if match:
            name = match.group(1).strip()
            description = match.group(2).strip()
            asset_id = f"bg_{name.lower().replace(' ', '_')}"
            return Background(asset_id=asset_id, name=name, description=description)
        return None

    def load_input_files(
        self,
        scene_file: Path | None = None,
        style_file: Path | None = None,
        objects_file: Path | None = None,
        characters_file: Path | None = None,
        backgrounds_file: Path | None = None,
        context_file: Path | None = None
    ) -> None:
        if scene_file and scene_file.exists():
            self.scene = scene_file.read_text().strip()

        if style_file and style_file.exists():
            self.style = style_file.read_text().strip()

        if context_file and context_file.exists():
            self.context = context_file.read_text().strip()

        if characters_file and characters_file.exists():
            for line in characters_file.read_text().splitlines():
                line = line.strip()
                if line:
                    char = self._parse_character_line(line)
                    if char:
                        self.characters.append(char)

        if objects_file and objects_file.exists():
            for line in objects_file.read_text().splitlines():
                line = line.strip()
                if line:
                    obj = self._parse_object_line(line)
                    if obj:
                        self.objects.append(obj)

        if backgrounds_file and backgrounds_file.exists():
            for line in backgrounds_file.read_text().splitlines():
                line = line.strip()
                if line:
                    bg = self._parse_background_line(line)
                    if bg:
                        self.backgrounds.append(bg)

    def create_job(self) -> str:
        job_id = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        job_dir = Path(f"pipeline/{job_id}")
        job_dir.mkdir(parents=True, exist_ok=True)

        if self.scene:
            (job_dir / "raw_input.txt").write_text(self.scene)

        if self.style:
            (job_dir / "style.txt").write_text(self.style)

        if self.context:
            (job_dir / "context.txt").write_text(self.context)

        if not self.characters:
            self._deduce_characters(job_id)
        else:
            chars_txt = "\n".join(f"{c.name} is {c.description}" for c in self.characters)
            (job_dir / "characters.txt").write_text(chars_txt)

        if not self.objects:
            self._deduce_objects(job_id)
        else:
            objs_txt = "\n".join(f"{o.name} is {o.description}" for o in self.objects)
            (job_dir / "objects.txt").write_text(objs_txt)

        if not self.backgrounds:
            self._deduce_backgrounds(job_id)
        else:
            bgs_txt = "\n".join(f"{b.name} is {b.description}" for b in self.backgrounds)
            (job_dir / "backgrounds.txt").write_text(bgs_txt)

        meta = {
            "job_id": job_id,
            "start_date": datetime.now().isoformat(),
            "stage": "created",
            "num_characters": len(self.characters),
            "num_objects": len(self.objects),
            "num_backgrounds": len(self.backgrounds),
        }
        with open(job_dir / "meta-info.json", "w") as f:
            json.dump(meta, f, indent=2)

        self._save_asset_manifest(job_id)
        self._save_harness(job_id)

        return job_id

    def _deduce_characters(self, job_id: str) -> None:
        job_dir = Path(f"pipeline/{job_id}")
        system_prompt = """Extract distinct characters from this scene. For each character provide:
- Name (use uppercase if not explicitly named, or infer a simple name)
- Visual description (what they look like, their state/emotion if relevant)

Format: One per line as "Name is description"

Be concise - just the key characters who drive the narrative."""

        characters_text = self._llm_call(system_prompt, self.scene)
        (job_dir / "characters.txt").write_text(characters_text)
        self.logger.info(f"{job_id}: Deduced characters:\n{characters_text}")

        for line in characters_text.splitlines():
            line = line.strip()
            if line:
                char = self._parse_character_line(line)
                if char:
                    self.characters.append(char)

    def _deduce_objects(self, job_id: str) -> None:
        job_dir = Path(f"pipeline/{job_id}")
        system_prompt = """Extract notable objects/props from this scene that are visually important for video generation.

For each object provide TWO things separated by " | ":
1. Scene context: where/how this object appears in the scene (e.g., "on a table at RadioShack" or "being held by a child")
2. Object description: what the object actually is (e.g., "Compaq portable computer, 1980s style")

Format: "object name | scene context | object description"
Example: "compaq portable | sitting on a table at RadioShack | 1980s beige portable computer with handle"

Focus on: electronics, furniture, vehicles, clothing, tools - things that appear in the scene."""

        objects_text = self._llm_call(system_prompt, self.scene)
        (job_dir / "objects.txt").write_text(objects_text)
        self.logger.info(f"{job_id}: Deduced objects:\n{objects_text}")

        for line in objects_text.splitlines():
            line = line.strip()
            if line:
                obj = self._parse_object_line(line)
                if obj:
                    self.objects.append(obj)

    def _deduce_backgrounds(self, job_id: str) -> None:
        job_dir = Path(f"pipeline/{job_id}")
        system_prompt = """Extract the distinct locations/backgrounds from this scene.
For each provide:
- Name (simple, descriptive)
- Visual description (setting, time of day if mentioned, atmosphere)

Format: One per line as "name is description"

Focus on: indoor/outdoor settings, specific locations mentioned, environments where actions occur."""

        backgrounds_text = self._llm_call(system_prompt, self.scene)
        (job_dir / "backgrounds.txt").write_text(backgrounds_text)
        self.logger.info(f"{job_id}: Deduced backgrounds:\n{backgrounds_text}")

        for line in backgrounds_text.splitlines():
            line = line.strip()
            if line:
                bg = self._parse_background_line(line)
                if bg:
                    self.backgrounds.append(bg)

    def _save_asset_manifest(self, job_id: str) -> None:
        job_dir = Path(f"pipeline/{job_id}")

        assets = []
        for char in self.characters:
            assets.append({
                "asset_id": char.asset_id,
                "type": "character",
                "name": char.name,
                "visual_description": char.description,
                "first_appears_in": ""
            })

        for obj in self.objects:
            assets.append({
                "asset_id": obj.asset_id,
                "type": "object",
                "name": obj.name,
                "visual_description": obj.description,
                "scene_description": obj.scene_description,
                "first_appears_in": ""
            })

        for bg in self.backgrounds:
            assets.append({
                "asset_id": bg.asset_id,
                "type": "background",
                "name": bg.name,
                "visual_description": bg.description,
                "first_appears_in": ""
            })

        with open(job_dir / "asset_manifest.json", "w") as f:
            json.dump(assets, f, indent=2)

        self.logger.info(f"{job_id}: Asset manifest created with {len(assets)} entities")

    def _save_harness(self, job_id: str) -> None:
        job_dir = Path(f"pipeline/{job_id}")

        harness = {
            "film_stock": "35mm film grain",
            "color_palette": ["cinematic"],
            "lighting_style": "cinematic",
            "aspect_ratio": "16:9",
            "era": "modern",
            "mood": "neutral",
            "negative_prompt": "anime, cartoon, low quality, distorted",
            "style_description": self.style,
            "context_description": self.context
        }

        with open(job_dir / "harness.json", "w") as f:
            json.dump(harness, f, indent=2)

        self.logger.info(f"{job_id}: Harness created")

    def run_track2(self, job_id: str) -> bool:
        self.logger.info(f"Running Track 2 for {job_id}")
        job_dir = Path(f"pipeline/{job_id}")

        try:
            assets_dir = job_dir / "assets"
            for subdir in ["characters", "objects", "backgrounds", "voices"]:
                (assets_dir / subdir).mkdir(parents=True, exist_ok=True)

            assets = json.loads((job_dir / "asset_manifest.json").read_text())
            if not assets:
                self.logger.info(f"{job_id}: No assets to generate")
                return True

            context = ""
            context_path = job_dir / "context.txt"
            if context_path.exists():
                context = context_path.read_text().strip()

            style_directive = self._derive_style_directive(context)

            self.logger.info(f"{job_id}: Generating assets")
            self.logger.info(f"{job_id}: Style directive: {style_directive}")

            brave_api_key = self.config.get("brave-api-key")

            for asset in assets:
                asset_id = asset["asset_id"]
                asset_type = asset["type"]
                description = asset.get("visual_description", asset.get("description", ""))

                if not description:
                    self.logger.warning(f"{job_id}: No description for {asset_id}, skipping")
                    continue

                subdir = asset_type + "s"
                out_path = assets_dir / subdir / f"{asset_id}.png"

                try:
                    if asset_type == "object":
                        scene_desc = asset.get("scene_description", "")
                        self.logger.info(f"{job_id}: Searching web for object '{asset_id}'")
                        api_key = brave_api_key or ""
                        image = self._search_object_image(
                            asset["name"],
                            scene_desc,
                            style_directive,
                            api_key
                        )
                        if image:
                            image.save(out_path)
                            self.logger.info(f"{job_id}: Saved (web) {out_path}")
                        else:
                            self.logger.warning(f"{job_id}: No web image found for {asset_id}, skipping")
                    else:
                        prompt = f"{style_directive}. {description}"
                        self.logger.info(f"{job_id}: Generating {asset_type} '{asset_id}': {prompt[:80]}...")
                        image = self._generate_image(prompt)
                        if image:
                            image.save(out_path)
                            self.logger.info(f"{job_id}: Saved (generated) {out_path}")

                except Exception as e:
                    self.logger.error(f"{job_id}: Failed to generate {asset_id}: {e}")
                    continue

            self.logger.info(f"{job_id}: Track 2 complete (assets generated)")
            return True

        except Exception as e:
            self.logger.error(f"{job_id}: Track 2 failed: {e}")
            return False

    def _search_object_image(self, name: str, scene_desc: str, style: str, api_key: str) -> Image.Image | None:
        if not api_key:
            self.logger.warning("No Brave API key configured")
            return None

        search_query = f"{name}"
        if scene_desc:
            search_query = f"{scene_desc} {name}"

        headers = {
            "X-Subscription-Token": api_key,
            "Accept": "application/json",
        }
        params = {
            "q": search_query,
            "count": 1,
        }

        try:
            response = requests.get(
                "https://api.search.brave.com/res/v1/images/search",
                headers=headers,
                params=params,
                timeout=15,
            )
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                if results and len(results) > 0:
                    image_url = results[0].get("thumbnail", {}).get("url")
                    if not image_url:
                        image_url = results[0].get("url")
                    self.logger.info(f"Found image: {image_url}")
                    img_response = requests.get(image_url, timeout=15)
                    if img_response.status_code == 200:
                        img = Image.open(BytesIO(img_response.content)).convert("RGB")
                        img = img.resize((512, 512), Image.Resampling.LANCZOS)
                        return img
        except Exception as e:
            self.logger.warning(f"Image search failed: {e}")
        return None

    def _generate_image(self, prompt: str) -> Image.Image | None:
        from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
        import torch

        pipe = Flux2KleinPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-9B",
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()

        result = pipe(
            prompt=prompt,
            height=512,
            width=512,
            guidance_scale=3.5,
            num_inference_steps=4,
        )
        image = result.images[0]

        del pipe
        torch.cuda.empty_cache()
        return image

    def _derive_style_directive(self, context: str) -> str:
        if not context:
            return "modern high quality"
        prompt = f"""Given this context: "{context[:500]}"
Describe the visual style/time period for image generation. Be declarative and concise, under 10 words. Example: "1980s suburban mall aesthetic with warm tones"
"""
        result = self._llm_call(prompt, "").strip()
        return result if result else "modern high quality"

    def run_track1(self, job_id: str) -> bool:
        self.logger.info(f"Running Track 1 for {job_id}")
        job_dir = Path(f"pipeline/{job_id}")

        meta = json.loads((job_dir / "meta-info.json").read_text())
        meta["stage"] = "track1_running"
        (job_dir / "meta-info.json").write_text(json.dumps(meta, indent=2))

        try:
            combined_input = self.scene
            if self.context:
                combined_input = f"Context: {self.context}\n\nScene: {self.scene}"

            system_prompt = """You are a script normalizer. Convert raw prose into a clean shooting script.
Rules:
- Present tense only
- Visual language only (describe what can be seen/heard on screen)
- No internal monologue or thoughts
- Each line is a single action or dialogue beat
- Include camera directions when implied by the prose
Output ONLY the normalized script, no explanations."""

            script = self._llm_call(system_prompt, combined_input)
            input_hash = self._hash(combined_input)
            output_hash = self._hash(script)
            self._log_pass(job_id, "script_normalization", input_hash, script, output_hash)

            (job_dir / "script.txt").write_text(script)
            self.logger.info(f"{job_id}: Script normalization complete")

            system_prompt = """You are an action itemizer. Break the script into atomic action items.
Each action item must be:
- A single observable action (no "continue" or "then")
- Under 10 seconds duration
- Has clear characters_involved, objects_involved, location

IMPORTANT: Output ONLY a valid JSON array, nothing else. Example:
[{"id": "act_01", "description": "ORION approaches a middle-aged woman at the demo table", "duration_estimate_seconds": 5.0, "characters_involved": ["ORION", "MIDDLE-AGED WOMAN"], "objects_involved": ["demo table", "computer"], "location": "INT. RADIO SHACK"}]"""

            action_items_json = self._llm_call(system_prompt, script)
            input_hash = self._hash(script)

            self.logger.info(f"{job_id}: action_itemization raw response length={len(action_items_json)}")
            self.logger.info(f"{job_id}: action_itemization raw response preview={action_items_json[:500]}")

            try:
                action_items = json.loads(action_items_json)
                if not isinstance(action_items, list):
                    self.logger.error(f"{job_id}: action_itemization returned {type(action_items)} instead of list")
                    action_items = []
            except json.JSONDecodeError as e:
                self.logger.error(f"{job_id}: JSON parse failed: {e}")
                self.logger.error(f"{job_id}: Raw response was: {action_items_json[:1000]}")
                action_items = []

            with open(job_dir / "action_items.json", "w") as f:
                json.dump(action_items, f, indent=2)

            self._log_pass(job_id, "action_itemization", input_hash, json.dumps(action_items), self._hash(json.dumps(action_items)))
            self.logger.info(f"{job_id}: Action itemization complete - {len(action_items)} items")

            if len(action_items) == 0:
                self.logger.error(f"{job_id}: No action items generated! Cannot proceed to shot grouping.")
                return False

            system_prompt = """You are a shot grouper. Group action items into continuous shots.
A new shot occurs when:
- Camera angle changes
- Location changes
- Time cuts

Each shot needs: shot_id, action_item_ids[], total_duration_seconds, shot_type (wide/medium/close/extreme)

IMPORTANT: Output ONLY a valid JSON array, nothing else."""

            shots_json = self._llm_call(system_prompt, json.dumps(action_items, indent=2))
            input_hash = self._hash(json.dumps(action_items))

            self.logger.info(f"{job_id}: shot_grouping raw response length={len(shots_json)}")
            self.logger.info(f"{job_id}: shot_grouping raw response preview={shots_json[:500]}")

            try:
                shots = json.loads(shots_json)
                if not isinstance(shots, list):
                    self.logger.error(f"{job_id}: shot_grouping returned {type(shots)} instead of list")
                    shots = []
            except json.JSONDecodeError as e:
                self.logger.error(f"{job_id}: JSON parse failed: {e}")
                self.logger.error(f"{job_id}: Raw response was: {shots_json[:1000]}")
                shots = []

            with open(job_dir / "shots.json", "w") as f:
                json.dump(shots, f, indent=2)

            self._log_pass(job_id, "shot_grouping", input_hash, json.dumps(shots), self._hash(json.dumps(shots)))
            self.logger.info(f"{job_id}: Shot grouping complete - {len(shots)} shots")

            total_duration = sum(s.get("total_duration_seconds", 0) for s in shots)
            self.logger.info(f"{job_id}: Total estimated duration: {total_duration}s")

            if shots and all(s.get("total_duration_seconds", 0) <= 10 for s in shots):
                self.logger.info(f"{job_id}: All shots under 10 seconds - PASS")
            else:
                self.logger.warning(f"{job_id}: Some shots exceed 10 seconds - may need calibration")

            meta["stage"] = "track1_complete"
            (job_dir / "meta-info.json").write_text(json.dumps(meta, indent=2))

            return True

        except Exception as e:
            self.logger.error(f"{job_id}: Track 1 failed: {e}")
            meta["stage"] = "track1_failed"
            meta["error"] = str(e)
            (job_dir / "meta-info.json").write_text(json.dumps(meta, indent=2))
            return False

    def build_scene_packages(self, job_id: str) -> bool:
        self.logger.info(f"Building scene packages for {job_id}")
        job_dir = Path(f"pipeline/{job_id}")

        try:
            shots = json.loads((job_dir / "shots.json").read_text())
            assets = json.loads((job_dir / "asset_manifest.json").read_text())
            harness = json.loads((job_dir / "harness.json").read_text())

            scene_packages_dir = job_dir / "scene_packages"
            scene_packages_dir.mkdir(parents=True, exist_ok=True)

            assets_dir = job_dir / "assets"
            for subdir in ["characters", "objects", "backgrounds", "voices"]:
                (assets_dir / subdir).mkdir(parents=True, exist_ok=True)

            for shot in shots:
                shot_id = shot["shot_id"]
                action_item_ids = shot["action_item_ids"]

                action_items = []
                shots_data = json.loads((job_dir / "shots.json").read_text())
                action_items_data = json.loads((job_dir / "action_items.json").read_text())
                for shot_data in shots_data:
                    if shot_data["shot_id"] == shot_id:
                        for aid in shot_data["action_item_ids"]:
                            for item in action_items_data:
                                if item["id"] == aid:
                                    action_items.append(item)
                                    break
                        break

                characters_in_shot = set()
                objects_in_shot = set()
                for ai in action_items:
                    for c in ai.get("characters_involved", []):
                        characters_in_shot.add(c.lower())
                    for o in ai.get("objects_involved", []):
                        objects_in_shot.add(o.lower())

                selected_assets = []
                for asset in assets:
                    asset_name_lower = asset["name"].lower()
                    if asset["type"] == "character" and asset_name_lower in characters_in_shot:
                        if len(selected_assets) < 5:
                            selected_assets.append({
                                "asset_id": asset["asset_id"],
                                "role": "primary_character" if len(selected_assets) == 0 else "secondary_character",
                                "file": ""
                            })
                    elif asset["type"] == "object" and asset_name_lower in objects_in_shot:
                        if len(selected_assets) < 5:
                            selected_assets.append({
                                "asset_id": asset["asset_id"],
                                "role": "object",
                                "file": ""
                            })
                    elif asset["type"] == "background":
                        if len(selected_assets) < 5:
                            selected_assets.append({
                                "asset_id": asset["asset_id"],
                                "role": "background",
                                "file": ""
                            })

                timing_breakdown = []
                cumulative_time = 0
                for ai in action_items:
                    duration = ai.get("duration_estimate_seconds", 3)
                    timing_breakdown.append({
                        "start": f"0:{cumulative_time:02d}",
                        "end": f"0:{cumulative_time + int(duration):02d}",
                        "action": ai["description"]
                    })
                    cumulative_time += int(duration)

                style_desc = harness.get("style_description", "")
                context_desc = harness.get("context_description", "")
                harness_str = f"Style: {style_desc}. " if style_desc else ""
                harness_str += f"Context: {context_desc}. " if context_desc else ""
                harness_str += f"Film stock: {harness['film_stock']}. Color palette: {', '.join(harness['color_palette'])}. Lighting: {harness['lighting_style']}. Era: {harness['era']}. Mood: {harness['mood']}."

                shot_type = shot.get("shot_type", "medium")
                timing_str = " ".join([f"{tb['start']}-{tb['end']}: {tb['action']}." for tb in timing_breakdown])
                full_prompt = f"{harness_str} {shot_type} shot. {timing_str}"

                package = {
                    "shot_id": shot_id,
                    "duration_seconds": shot.get("total_duration_seconds", 8),
                    "harness": harness,
                    "assets": selected_assets,
                    "timing_breakdown": timing_breakdown,
                    "full_prompt": full_prompt,
                    "shot_type": shot_type,
                    "retry_count": 0,
                    "status": "pending"
                }

                with open(scene_packages_dir / f"{shot_id}.json", "w") as f:
                    json.dump(package, f, indent=2)

            self.logger.info(f"{job_id}: Built {len(shots)} scene packages")

            meta = json.loads((job_dir / "meta-info.json").read_text())
            meta["stage"] = "scene_packages_built"
            (job_dir / "meta-info.json").write_text(json.dumps(meta, indent=2))

            return True

        except Exception as e:
            self.logger.error(f"{job_id}: Scene package building failed: {e}")
            return False

    def run_full_pipeline(self) -> str:
        job_id = self.create_job()

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_t1 = executor.submit(self.run_track1, job_id)
            future_t2 = executor.submit(self.run_track2, job_id)

            track1_success = future_t1.result()
            track2_success = future_t2.result()

        if not track1_success:
            raise RuntimeError(f"Track 1 failed for job {job_id}")

        if not track2_success:
            raise RuntimeError(f"Track 2 failed for job {job_id}")

        if not self.build_scene_packages(job_id):
            raise RuntimeError(f"Scene package building failed for job {job_id}")

        meta = json.loads((Path(f"pipeline/{job_id}") / "meta-info.json").read_text())
        meta["stage"] = "pipeline_complete"
        meta["complete_date"] = datetime.now().isoformat()
        (Path(f"pipeline/{job_id}") / "meta-info.json").write_text(json.dumps(meta, indent=2))

        self.logger.info(f"Pipeline complete for job {job_id}")
        return job_id


def parse_args():
    parser = argparse.ArgumentParser(description="Video Harness Pipeline")
    parser.add_argument("--scene", type=Path, help="Scene/prose input file")
    parser.add_argument("--style", type=Path, help="Style description file")
    parser.add_argument("--objects", type=Path, help="Objects description file")
    parser.add_argument("--characters", type=Path, help="Characters description file")
    parser.add_argument("--backgrounds", type=Path, help="Backgrounds description file")
    parser.add_argument("--context", type=Path, help="Context/setting description file")
    return parser.parse_args()


def main():
    args = parse_args()

    if not any([args.scene, args.style, args.objects, args.characters, args.backgrounds, args.context]):
        print("Usage: python pipeline.py --scene <scene.txt> [--style <style.txt>] [--objects <objects.txt>] [--characters <characters.txt>] [--backgrounds <backgrounds.txt>] [--context <context.txt>]")
        sys.exit(1)

    pipeline = Pipeline()
    pipeline.load_input_files(
        scene_file=args.scene,
        style_file=args.style,
        objects_file=args.objects,
        characters_file=args.characters,
        backgrounds_file=args.backgrounds,
        context_file=args.context
    )

    job_id = pipeline.run_full_pipeline()
    print(f"Pipeline complete. Job ID: {job_id}")


if __name__ == "__main__":
    main()
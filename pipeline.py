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

from PIL import Image

OpenAI = None

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
        global OpenAI
        if OpenAI is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise RuntimeError("OpenAI library not installed. Run: pip install openai")

        url = self.text_config.get("url", "http://localhost:11434")
        model = self.text_config.get("model", "llama.cpp")
        api_key = self.text_config.get("key")

        full_url = f"{url}/v1/chat/completions"
        self.logger.info(f"LLM Request: URL={full_url}, model={model}")

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
            asset_id = f"char_{re.sub(r'[^a-z0-9_]', '', name.lower().replace(' ', '_'))}"
            return Character(asset_id=asset_id, name=name, description=description)
        return None

    def _parse_object_line(self, line: str) -> Object | None:
        line = line.strip()
        if not line:
            return None

        name = ""
        obj_desc = ""
        scene_desc = ""

        if ":" in line:
            name, rest = line.split(":", 1)
            name = name.strip()
            obj_desc = rest.strip()
        elif "-" in line:
            name, rest = line.split("-", 1)
            name = name.strip()
            obj_desc = rest.strip()
        elif " | " in line:
            parts = line.split(" | ")
            if len(parts) >= 3:
                name = parts[0].strip()
                scene_desc = parts[1].strip()
                obj_desc = parts[2].strip()
            else:
                name = parts[0].strip()
                obj_desc = parts[1].strip()
        elif " is " in line:
            name, rest = line.split(" is ", 1)
            name = name.strip()
            obj_desc = rest.strip()
        else:
            return None

        if not name or not obj_desc:
            return None

        asset_id = f"obj_{re.sub(r'[^a-z0-9_]', '', name.lower().replace(' ', '_'))}"
        return Object(asset_id=asset_id, name=name, description=obj_desc, scene_description=scene_desc)

    def _parse_background_line(self, line: str) -> Background | None:
        match = re.match(r"(.+?)\s+is\s+(.+)", line.strip())
        if match:
            name = match.group(1).strip()
            description = match.group(2).strip()
            asset_id = f"bg_{re.sub(r'[^a-z0-9_]', '', name.lower().replace(' ', '_'))}"
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
            content = objects_file.read_text().strip()
            if content:
                for line in content.splitlines():
                    line = line.strip()
                    if line:
                        obj = self._parse_object_line(line)
                        if obj:
                            self.objects.append(obj)

        if backgrounds_file and backgrounds_file.exists():
            content = backgrounds_file.read_text().strip()
            if content:
                for line in content.splitlines():
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

For each object provide:
1. Object name (simple, brief - what it IS)
2. Visual description (what it looks like, specific details that identify it)

Format: "name: description" - one object per line
Example: "compaq portable: 1980s beige portable computer, 30 pounds with handle"
Example: "AMC Eagle: 1980s four-wheel-drive station wagon, rust-colored"
Example: "mouse: early 1980s mechanical computer mouse, beige with cord"

Be specific - describe it like you're telling someone what to google for a reference image.
Focus on: electronics, furniture, vehicles, clothing, tools - things that appear in the scene.
Do NOT include scene context (where it is in the scene) - just what the object IS."""

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
                for kind in ["gen", "web"]:
                    (assets_dir / subdir / kind).mkdir(exist_ok=True)

            assets = json.loads((job_dir / "asset_manifest.json").read_text())
            if not assets:
                self.logger.info(f"{job_id}: No assets to generate")
                return True

            context = ""
            context_path = job_dir / "context.txt"
            if context_path.exists():
                context = context_path.read_text().strip()

            year_hint = self._derive_year_hint(context)
            style_directive = self._derive_style_directive(context)

            self.logger.info(f"{job_id}: Generating assets")
            self.logger.info(f"{job_id}: Year hint: {year_hint}")
            self.logger.info(f"{job_id}: Style directive: {style_directive}")

            brave_api_key = self.config.get("brave-api-key")

            for asset in assets:
                asset_id = asset["asset_id"]
                asset_type = asset["type"]
                full_prompt = asset.get("full_prompt", "").strip()
                description = asset.get("visual_description", asset.get("description", ""))

                if not full_prompt and not description:
                    self.logger.warning(f"{job_id}: No description for {asset_id}, skipping")
                    continue

                subdir = asset_type + "s"

                try:
                    if asset_type == "object":
                        self.logger.info(f"{job_id}: Searching web for object '{asset_id}'")
                        api_key = brave_api_key or ""
                        disambiguated = self._disambiguate_object_name(asset["name"], context)
                        self.logger.info(f"{job_id}: Disambiguated '{asset['name']}' -> '{disambiguated}'")
                        image = self._search_object_image(
                            disambiguated,
                            year_hint,
                            api_key
                        )
                        if image:
                            web_path = assets_dir / subdir / "web" / f"{asset_id}.png"
                            image.save(web_path)
                            self._montage_text_label(image, web_path, asset["name"])
                            self.logger.info(f"{job_id}: Saved (web) {web_path}")
                        else:
                            self.logger.info(f"{job_id}: No web image found for {asset_id}, skipping")
                    elif asset_type == "character":
                        prompt = full_prompt or f"{year_hint}, realistic full color portrait photograph, {description}"
                        self.logger.info(f"{job_id}: Generating character '{asset_id}': {prompt[:80]}...")
                        image = self._generate_image(prompt)
                        if image:
                            gen_path = assets_dir / subdir / "gen" / f"{asset_id}.png"
                            image.save(gen_path)
                            self._montage_text_label(image, gen_path, asset["name"])
                            self.logger.info(f"{job_id}: Saved (gen) {gen_path}")
                    else:
                        prompt = full_prompt or f"{year_hint}, {description}"
                        self.logger.info(f"{job_id}: Generating {asset_type} '{asset_id}': {prompt[:80]}...")
                        image = self._generate_image(prompt)
                        if image:
                            gen_path = assets_dir / subdir / "gen" / f"{asset_id}.png"
                            image.save(gen_path)
                            self._montage_text_label(image, gen_path, asset["name"])
                            self.logger.info(f"{job_id}: Saved (gen) {gen_path}")

                except Exception as e:
                    self.logger.error(f"{job_id}: Failed to generate {asset_id}: {e}")
                    continue

            self.logger.info(f"{job_id}: Track 2 complete (assets generated)")
            return True

        except Exception as e:
            self.logger.error(f"{job_id}: Track 2 failed: {e}")
            return False

    def _derive_year_hint(self, context: str) -> str:
        if not context:
            return "1980s"
        prompt = f"""Given this context: "{context[:500]}"
Extract only the time period/year range for image generation. Be very specific with years if mentioned.
Just output the year or decade, nothing else. Example outputs: "1982", "late 1970s", "early 1980s"
"""
        result = self._llm_call(prompt, "").strip()
        return result if result else "1980s"

    def _disambiguate_object_name(self, name: str, context: str) -> str:
        prompt = f"""Given this object name: "{name}"
And this context: "{context[:500]}"

What would this prompt generate? Would it be ambiguous?

If "mouse" could mean the animal or a computer mouse, "keyboard" could mean musical or computer, etc.

If ambiguous, suggest 2-3 specific adjectives to disambiguate. Output ONLY the suggested phrase, nothing else.
Example output for "mouse" in 1982 context: "1982 computer mouse"
Example output for "keyboard" in 1982 context: "computer keyboard"
Example output for " Jaguar" in 1982 context: "1980s Jaguar car"
If not ambiguous, output just the name as-is."""
        result = self._llm_call(prompt, "").strip()
        return result if result else name

    def _search_object_image(self, name: str, year_hint: str, api_key: str) -> Image.Image | None:
        if not api_key:
            self.logger.warning("No Brave API key configured")
            return None

        search_query = f"{name}"

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
                    thumbnail = results[0].get("thumbnail", {})
                    image_url = thumbnail.get("src") if isinstance(thumbnail, dict) else None
                    if not image_url:
                        image_url = results[0].get("url")
                    self.logger.info(f"Found image for '{search_query}': {image_url}")
                    img_response = requests.get(image_url, timeout=15)
                    if img_response.status_code == 200:
                        img = Image.open(BytesIO(img_response.content)).convert("RGB")
                        return img
        except Exception as e:
            self.logger.warning(f"Image search failed: {e}")
        return None

    def _generate_image(self, prompt: str, ref_image: Image.Image | None = None) -> Image.Image | None:
        from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
        import torch

        pipe = Flux2KleinPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-9B",
            torch_dtype=torch.bfloat16,
        )
        pipe.enable_model_cpu_offload()

        self.logger.info(f"=== FLUX CALL. Prompt: {prompt[:80]}, ref_image: {ref_image is not None}")
        if ref_image:
            self.logger.info(f"Ref image size: {ref_image.size}, mode: {ref_image.mode}")
            ref_image = ref_image.resize((512, 512), Image.Resampling.LANCZOS)
            self.logger.info(f"Ref image resized to: {ref_image.size}")

        kwargs = {
            "prompt": prompt,
            "height": 512,
            "width": 512,
            "num_inference_steps": 4,
        }
        if ref_image:
            kwargs["image"] = ref_image

        result = pipe(**kwargs)
        image = result.images[0]

        del pipe
        del result
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return image

    def _montage_text_label(self, image: Image.Image, output_path: Path, label: str) -> None:
        import subprocess

        label_clean = label.strip()
        temp_path = output_path.with_suffix(".tmp.png")
        image.save(temp_path)

        cmd = [
            "convert",
            "-size", f"{image.width}x50",
            "-background", "black",
            "-fill", "white",
            "-font", "DejaVu-Sans",
            "-pointsize", "28",
            "-gravity", "center",
            f"label:{label_clean}",
            temp_path,
            "-append",
            str(output_path)
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            temp_path.unlink()
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"ImageMagick montage failed: {e}")
            if temp_path.exists():
                temp_path.rename(output_path)

    def _composite_scene_image(self, job_dir: Path, package: dict, assets: list[dict]) -> Path | None:
        try:
            from PIL import Image, ImageDraw, ImageFont

            assets_in_shot = package.get("assets", [])
            self.logger.info(f"=== COMPOSITE START. Assets in shot: {len(assets_in_shot)}, job_dir: {job_dir}")

            if not assets_in_shot:
                self.logger.warning("No assets_in_shot")
                return None

            composite_prompt = package.get("composite_prompt", "")
            self.logger.info(f"Composite prompt: {composite_prompt[:100]}")
            bg_height = 300
            char_height = 150
            obj_height = 100
            caption_height = 40
            total_width = 1024
            label_height = 50

            total_height = bg_height + char_height + obj_height + caption_height + label_height
            composite = Image.new("RGB", (total_width, total_height), color="black")
            draw = ImageDraw.Draw(composite)

            current_y = 0

            captions_used = []
            
            bg_info = None
            chars_objs = []
            
            for asset_ref in assets_in_shot:
                if asset_ref.get("role") == "background":
                    for asset in assets:
                        if asset["asset_id"] == asset_ref["asset_id"]:
                            name = asset.get("name", "background")
                            desc = asset.get("visual_description", "")
                            bg_info = f"{name}: {desc}" if desc else name
                            break
                else:
                    for asset in assets:
                        if asset["asset_id"] == asset_ref["asset_id"]:
                            role = asset_ref.get("role", "")
                            name = asset.get("name", "object")
                            desc = asset.get("visual_description", "")
                            chars_objs.append((role, f"{name} - {desc}" if desc else name))
                            break
            
            flux_parts = []
            if bg_info:
                flux_parts.append(f"Background: {bg_info}")
            for role, name in chars_objs:
                flux_parts.append(f"{name}")
            
            auto_composite_prompt = ", ".join(flux_parts)
            
            package["composite_prompt"] = package.get("composite_prompt") or auto_composite_prompt

            for asset_ref in assets_in_shot:
                if asset_ref.get("role") == "background":
                    asset_id = asset_ref["asset_id"]
                    for asset in assets:
                        if asset["asset_id"] == asset_id:
                            img = self._load_asset_image(job_dir, asset, "background")
                            if img:
                                img = img.resize((total_width, bg_height), Image.Resampling.LANCZOS)
                                composite.paste(img, (0, current_y))
                                current_y += bg_height
                                cap = asset.get("name", "")
                                if cap:
                                    captions_used.append(f"[{cap}]")
                            break

            for asset_ref in assets_in_shot:
                if asset_ref.get("role") in ("object", "primary_character", "secondary_character"):
                    asset_id = asset_ref["asset_id"]
                    asset_type = "object" if asset_ref.get("role") == "object" else "character"
                    for asset in assets:
                        if asset["asset_id"] == asset_id:
                            img = self._load_asset_image(job_dir, asset, asset_type)
                            if img:
                                target_h = char_height if asset_type == "character" else obj_height
                                ratio = target_h / img.height
                                new_w = int(img.width * ratio)
                                img = img.resize((new_w, target_h), Image.Resampling.LANCZOS)
                                x_offset = (total_width - img.width) // 2
                                composite.paste(img, (x_offset, current_y))
                                current_y += target_h
                                cap = asset.get("name", "")
                                if cap:
                                    captions_used.append(f"[{cap}]")
                            break

            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
            except:
                font = ImageFont.load_default()

            caption_text = " ".join(captions_used) if captions_used else f"Shot {package['shot_id']}"
            caption_text = f"[ {caption_text} ]"

            draw.rectangle([(0, current_y), (total_width, current_y + caption_height)], fill="black")
            bbox = draw.textbbox((0, 0), caption_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            text_x = (total_width - text_w) // 2
            text_y = current_y + (caption_height - text_h) // 2
            draw.text((text_x, text_y), caption_text, fill="white", font=font)
            current_y += caption_height

            if composite_prompt:
                draw.rectangle([(0, current_y), (total_width, current_y + label_height)], fill="#1a1a1a")
                small_font = ImageFont.load_default()
                words = composite_prompt.split()
                lines = []
                current_line = ""
                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    if draw.textlength(test_line, font=small_font) < total_width - 20:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)

                for i, line in enumerate(lines[:2]):
                    draw.text((10, current_y + i * 16), line, fill="#aaa", font=small_font)
                current_y += label_height

            # Don't add text labels - FLUX will generate clean composite
            # label_text intentionally removed

            renders_dir = job_dir / "renders"
            renders_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info("Generating composite with FLUX using reference...")
            result = self._generate_image(composite_prompt, ref_image=composite)
            output_path = renders_dir / f"shot_{package['shot_id']}_keyframe.png"
            if result:
                result.save(output_path)
                self.logger.info(f"Saved FLUX composite: {output_path}")
            else:
                composite.save(output_path)
                self.logger.info(f"Saved collage fallback: {output_path}")
            return output_path

        except Exception as e:
            self.logger.warning(f"Composite image failed: {e}")
            return None

    def _load_asset_image(self, job_dir: Path, asset: dict, asset_type: str) -> Image.Image | None:
        asset_id = asset["asset_id"]
        subdir = asset_type + "s"
        gen_path = job_dir / "assets" / subdir / "gen" / f"{asset_id}.png"
        web_path = job_dir / "assets" / subdir / "web" / f"{asset_id}.png"

        if asset.get("has_gen") and gen_path.exists():
            return Image.open(gen_path).convert("RGB")
        elif asset.get("has_web") and web_path.exists():
            return Image.open(web_path).convert("RGB")
        return None

    def _derive_style_directive(self, context: str) -> str:
        if not context:
            return "high quality"
        prompt = f"""Given this context: "{context[:500]}"
Describe the visual style and aesthetic for image generation. Be declarative and concise, under 10 words.
Example: "suburban mall aesthetic with warm tones" or "corporate office with fluorescent lighting"
Do NOT include years or time periods - just the visual style/aesthetic.
"""
        result = self._llm_call(prompt, "").strip()
        return result if result else "high quality"

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

                composite_path = self._composite_scene_image(job_dir, package, assets)
                if composite_path:
                    package["keyframe_image"] = str(composite_path.relative_to(job_dir))

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
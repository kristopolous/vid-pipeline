#!/usr/bin/env python3
"""VPLib - Shared video pipeline generation library."""

import io
import logging
import os
import requests
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger("vplib")


class VPLib:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.logger = logging.getLogger("vplib")

    def generate_image(self, prompt: str, ref_image: "Image.Image | None" = None) -> "Image.Image | None":
        """Generate an image - tries multiple backends in order."""
        
        # Try Flux pipeline for actual generation
        try:
            return self._generate_flux_image(prompt, ref_image)
        except Exception as e:
            self.logger.warning(f"Flux generation failed: {e}")
        
        # Try remote API  
        wan2gp_url = self.config.get("wan2gp", {}).get("url") or self.config.get("url")
        if wan2gp_url:
            img = self._generate_remote_image(prompt, ref_image)
            if img:
                return img
        
        # Final fallback: web search
        self.logger.info(f"Trying web search fallback for: {prompt[:50]}")
        return self.generate_image_fallback(prompt)
    
    def _generate_flux_image(self, prompt: str, ref_image: "Image.Image | None" = None) -> "Image.Image | None":
        """Use local Flux2Klein pipeline for image generation."""
        import torch
        from diffusers.pipelines import Flux2KleinPipeline
        
        self.logger.info(f"Flux2Klein: {prompt[:60]}...")
        
        if not hasattr(self, "_flux_pipe"):
            self._flux_pipe = Flux2KleinPipeline.from_pretrained(
                "black-forest-labs/FLUX.2-klein-9B",
                torch_dtype=torch.bfloat16,
            )
            self._flux_pipe.enable_model_cpu_offload()
        
        kwargs = {
            "prompt": prompt,
            "num_inference_steps": 4,
        }
        
        if ref_image:
            ref = ref_image.resize((512, 512))
            kwargs["image"] = ref
        
        result = self._flux_pipe(**kwargs)
        return result.images[0]
    
    def _generate_remote_image(self, prompt: str, ref_image: "Image.Image | None" = None) -> "Image.Image | None":
        """Use remote wan2gp API for image generation."""
        import requests
        
        wan2gp_url = self.config.get("wan2gp", {}).get("url")
        if not wan2gp_url:
            return None
        
        api_url = f"{wan2gp_url}/v1/image/generate"
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "prompt": prompt,
            "model": self.config.get("wan2gp", {}).get("image", "flux-2"),
            "num_inference_steps": 4,
        }
        
        try:
            response = requests.post(api_url, json=payload, headers=headers, timeout=60)
            if response.status_code == 200:
                from PIL import Image
                import numpy as np
                data = response.json()
                image_data = data.get("image")
                if image_data:
                    # Could be base64 encoded
                    import base64
                    img_bytes = base64.b64decode(image_data)
                    return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            self.logger.error(f"Remote image generation failed: {e}")
        return None
    
    def generate_image_fallback(self, prompt: str) -> "Image.Image | None":
        """Fallback: use Brave web search to get an image."""
        
        api_key = self.config.get("brave-api-key")
        if not api_key:
            self.logger.warning("No Brave API key configured")
            return None
        
        # Search for an image matching the prompt
        params = {"q": prompt, "count": 1}
        headers = {"X-Subscription-Token": api_key, "Accept": "application/json"}
        
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
                if results:
                    thumb = results[0].get("thumbnail", {})
                    image_url = thumb.get("src") if isinstance(thumb, dict) else None
                    if not image_url:
                        image_url = results[0].get("url")
                    
                    # Download the image
                    img_response = requests.get(image_url, timeout=15)
                    if img_response.status_code == 200:
                        from PIL import Image
                        img = Image.open(io.BytesIO(img_response.content)).convert("RGB")
                        self.logger.info(f"Web search found image for: {prompt[:50]}")
                        return img
        except Exception as e:
            self.logger.warning(f"Web search failed: {e}")
        return None

    def generate_character_sheet(
            self,
            asset_id: str,
            name: str,
            description: str,
            year_hint: str = "1980s",
            full_prompt: str = "",
    ) -> dict:
        base_prompt = full_prompt or f"{year_hint}, realistic full color portrait photograph, {description}"
        angles = {
            "headshot": base_prompt,
            "left": f"{base_prompt}, facing left, left profile view, neutral gray background",
            "right": f"{base_prompt}, facing right, right profile view, neutral gray background",
            "full": f"{base_prompt}, full body shot, standing, neutral background, from distance",
        }

        results = {}
        for angle_name, prompt in angles.items():
            self.logger.info(f"Generating {angle_name}: {prompt[:60]}...")
            try:
                image = self.generate_image(prompt)
                if image:
                    results[angle_name] = {
                        "image": image,
                        "prompt": prompt,
                        "angle": angle_name,
                        "model": "flux2-klein-9B"
                    }
                    self.logger.info(f"Generated {angle_name}")
            except Exception as e:
                self.logger.error(f"Failed to generate {angle_name}: {e}")
        return results

    def search_object_image(
            self, name: str, year_hint: str, api_key: str
    ) -> "Image.Image | None":
        import io
        import requests
        from PIL import Image

        if not api_key:
            self.logger.warning("No Brave API key configured")
            return None

        params = {"q": name, "count": 1}
        headers = {"X-Subscription-Token": api_key, "Accept": "application/json"}

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
                if results:
                    thumb = results[0].get("thumbnail", {})
                    image_url = thumb.get("src") if isinstance(thumb, dict) else None
                    if not image_url:
                        image_url = results[0].get("url")
                    img_response = requests.get(image_url, timeout=15)
                    if img_response.status_code == 200:
                        return Image.open(io.BytesIO(img_response.content)).convert("RGB")
        except Exception as e:
            self.logger.warning(f"Image search failed: {e}")
        return None

    def add_text_label(
            self, image: "Image.Image", output_path: Path, label: str
    ) -> None:
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
            str(temp_path),
            "-append",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            temp_path.unlink()
        except subprocess.CalledProcessError:
            if temp_path.exists():
                temp_path.rename(output_path)

    def load_asset_image(
            self, job_dir: Path, asset: dict, asset_type: str
    ) -> "Image.Image | None":
        from PIL import Image

        asset_id = asset["asset_id"]
        subdir = asset_type + "s"
        gen_path = job_dir / "assets" / subdir / "gen" / f"{asset_id}.png"
        web_path = job_dir / "assets" / subdir / "web" / f"{asset_id}.png"

        if asset.get("has_gen") and gen_path.exists():
            return Image.open(gen_path).convert("RGB")
        elif asset.get("has_web") and web_path.exists():
            return Image.open(web_path).convert("RGB")
        return None

    def composite_scene_image(
            self, job_dir: Path, package: dict, assets: list[dict]
    ) -> Path | None:
        from PIL import Image, ImageDraw, ImageFont

        try:
            assets_in_shot = package.get("assets", [])
            if not assets_in_shot:
                self.logger.warning("No assets in shot")
                return None

            composite_prompt = package.get("composite_prompt", "")

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
                            name = asset.get("name", "object")
                            desc = asset.get("visual_description", "")
                            chars_objs.append((asset_ref.get("role", ""), f"{name} - {desc}" if desc else name))
                            break

            flux_parts = []
            if bg_info:
                flux_parts.append(f"Background: {bg_info}")
            for _, name in chars_objs:
                flux_parts.append(name)

            auto_composite_prompt = ", ".join(flux_parts)
            package["composite_prompt"] = package.get("composite_prompt") or auto_composite_prompt

            for asset_ref in assets_in_shot:
                if asset_ref.get("role") == "background":
                    asset_id = asset_ref["asset_id"]
                    for asset in assets:
                        if asset["asset_id"] == asset_id:
                            img = self.load_asset_image(job_dir, asset, "background")
                            if img:
                                img = img.resize((total_width, bg_height))
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
                            img = self.load_asset_image(job_dir, asset, asset_type)
                            if img:
                                target_h = char_height if asset_type == "character" else obj_height
                                ratio = target_h / img.height
                                new_w = int(img.width * ratio)
                                img = img.resize((new_w, target_h))
                                x_offset = (total_width - img.width) // 2
                                composite.paste(img, (x_offset, current_y))
                                current_y += target_h
                                cap = asset.get("name", "")
                                if cap:
                                    captions_used.append(f"[{cap}]")
                            break

            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28
                )
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

            renders_dir = job_dir / "renders"
            renders_dir.mkdir(parents=True, exist_ok=True)

            result = self.generate_image(composite_prompt, ref_image=composite)
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

    def render_video(
            self,
            prompt: str,
            negative_prompt: str = "anime, cartoon, low quality, distorted",
            width: int = 1280,
            height: int = 704,
            num_frames: int = 96,
            num_inference_steps: int = 8,
            guidance_scale: float = 3.5,
    ) -> "Image.Image | None":
        from diffusers import LTXVideoPipeline
        import torch

        try:
            pipeline = LTXVideoPipeline.from_pretrained(
                "Lightricks/LTX-2",
                torch_dtype=torch.bfloat16,
            )
            pipeline.enable_sequential_cpu_offload()
            pipeline.enable_vae_spatial_tiling()

            self.logger.info(f"Rendering: {num_frames} frames, {width}x{height}")

            output = pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )

            return output.frames[0] if hasattr(output, "frames") else None

        except Exception as e:
            self.logger.error(f"Render failed: {e}")
            return None
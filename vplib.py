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
            output_path: "Path | str | None" = None,
            first_frame: "Image.Image | Path | str | None" = None,
            seed: int = 0,
    ) -> "list[Image.Image] | Path | None":
        """Render a video with the configured backend.

        Backend is selected via ``config["wan2gp"]["video"]``:

        - ``ltx-2`` (default): diffusers ``LTXVideoPipeline``. Returns a list of
          PIL frames (decode later with moviepy/cv2).
        - ``minimax-h3``: ComfyUI running the quantized MiniMax-H3 GGUF model.
          Returns the path of the finished mp4 (video + native audio).

        ``output_path`` and ``first_frame`` only apply to the ComfyUI backend.
        """
        video_backend = self.config.get("wan2gp", {}).get("video", "ltx-2")
        if video_backend in ("minimax-h3", "minimax_h3", "minimax"):
            return self.render_video_minimax_h3(
                prompt=prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                first_frame=first_frame,
                seed=seed,
                output_path=output_path,
            )
        return self._render_video_ltx(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

    def _render_video_ltx(
            self,
            prompt: str,
            negative_prompt: str = "anime, cartoon, low quality, distorted",
            width: int = 1280,
            height: int = 704,
            num_frames: int = 96,
            num_inference_steps: int = 8,
            guidance_scale: float = 3.5,
    ) -> "list[Image.Image] | None":
        from diffusers import LTXVideoPipeline
        import torch

        try:
            if not hasattr(self, "_ltx_pipe"):
                self.logger.info("Loading LTX-2 pipeline (kept resident)")
                self._ltx_pipe = LTXVideoPipeline.from_pretrained(
                    "Lightricks/LTX-2",
                    torch_dtype=torch.bfloat16,
                )
                self._ltx_pipe.enable_sequential_cpu_offload()
                self._ltx_pipe.enable_vae_spatial_tiling()
            pipeline = self._ltx_pipe

            self.logger.info(f"Rendering (LTX-2): {num_frames} frames, {width}x{height}")

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

    # ------------------------------------------------------------------
    # ComfyUI backend (quantized MiniMax-H3)
    # ------------------------------------------------------------------

    @staticmethod
    def _snap_frame_count(num_frames: int) -> int:
        """Snap a frame count up to MiniMax-H3's 17k+5 grid (ComfyUI alignment)."""
        n = max(int(num_frames), 5)
        while n % 17 != 5:
            n += 1
        return n

    def _comfyui_url(self) -> str:
        url = self.config.get("comfyui", {}).get("url") or ""
        return url.rstrip("/")

    def _comfyui_build_workflow(
            self,
            prompt: str,
            width: int,
            height: int,
            length: int,
            first_frame_name: "str | None" = None,
            seed: int = 0,
    ) -> dict:
        """Build a ComfyUI API-format workflow for MiniMax-H3 T2V/FL2V.

        Mirrors the official Comfy-Org 'MiniMax H3 T2V' template but uses the
        GGUF loaders from ComfyUI-GGUF so Abiray/MiniMax-H3-GGUF checkpoints work.
        """
        cfg = self.config.get("comfyui", {}) or {}
        unet = cfg.get("unet", "MiniMax-H3-FL2VA-Q4_K_M.gguf")
        text_encoder = cfg.get("text_encoder", "qwen3vl_32b_minimax_h3-Q4_K_M.gguf")
        video_vae = cfg.get("video_vae", "minimax_h3_video_vae_fp16.safetensors")
        audio_vae = cfg.get("audio_vae", "minimax_h3_audio_vae_fp32.safetensors")
        steps = int(cfg.get("steps", 25))
        scheduler = cfg.get("scheduler", "simple")
        sampler = cfg.get("sampler", "res_multistep")
        weight_dtype = cfg.get("weight_dtype", "default")

        def node(class_type: str, inputs: dict) -> dict:
            return {"class_type": class_type, "inputs": inputs}

        wf: dict = {}

        if unet.lower().endswith(".gguf"):
            wf["unet"] = node("UnetLoaderGGUF", {"unet_name": unet})
        else:
            wf["unet"] = node("UNETLoader", {"unet_name": unet, "weight_dtype": weight_dtype})

        if text_encoder.lower().endswith(".gguf"):
            wf["clip"] = node("CLIPLoaderGGUF", {"clip_name": text_encoder, "type": "minimax"})
        else:
            wf["clip"] = node("CLIPLoader", {"clip_name": text_encoder, "type": "minimax", "device": "default"})

        wf["vae_video"] = node("VAELoader", {"vae_name": video_vae})
        wf["vae_audio"] = node("VAELoader", {"vae_name": audio_vae})

        if first_frame_name:
            wf["first_frame"] = node("LoadImage", {"image": first_frame_name})

        mini_inputs: dict = {
            "clip": ["clip", 0],
            "vae": ["vae_video", 0],
            "prompt": prompt,
            "width": width,
            "height": height,
            "length": length,
        }
        if first_frame_name:
            mini_inputs["first_frame"] = ["first_frame", 0]
        wf["mini"] = node("MiniMaxH3ImageToVideo", mini_inputs)

        wf["noise"] = node("RandomNoise", {"noise_seed": seed, "noise_mode": "randomize"})
        wf["sched"] = node(
            "BasicScheduler",
            {"scheduler": scheduler, "steps": steps, "denoise": 1.0, "model": ["unet", 0]},
        )
        wf["ks"] = node("KSamplerSelect", {"sampler_name": sampler})
        wf["guider"] = node("BasicGuider", {"model": ["unet", 0], "conditioning": ["mini", 0]})
        wf["sampler"] = node(
            "SamplerCustomAdvanced",
            {
                "noise": ["noise", 0],
                "guider": ["guider", 0],
                "sampler": ["ks", 0],
                "sigmas": ["sched", 0],
                "latent_image": ["mini", 1],
            },
        )
        wf["decode"] = node("VAEDecode", {"samples": ["sampler", 0], "vae": ["vae_video", 0]})
        wf["decode_audio"] = node("VAEDecodeAudio", {"samples": ["sampler", 0], "vae": ["vae_audio", 0]})
        wf["create"] = node(
            "CreateVideo",
            {"fps": 24.0, "images": ["decode", 0], "audio": ["decode_audio", 0]},
        )
        wf["save"] = node(
            "SaveVideo",
            {"filename_prefix": "vid_harness", "format": "auto", "video": ["create", 0]},
        )
        return wf

    def _comfyui_upload_image(self, image_path: Path, name: str) -> str:
        """Upload an image to ComfyUI's input dir; returns the filename to reference."""
        import requests

        url = self._comfyui_url()
        if not url:
            raise RuntimeError("comfyui.url not configured in config.json")
        with open(image_path, "rb") as f:
            resp = requests.post(
                f"{url}/upload/image",
                files={"image": (name, f, "image/png")},
                data={"type": "input", "overwrite": "true"},
                timeout=60,
            )
        resp.raise_for_status()
        return resp.json()["name"]

    def _comfyui_submit(self, workflow: dict, client_id: str) -> str:
        import requests

        url = self._comfyui_url()
        if not url:
            raise RuntimeError("comfyui.url not configured in config.json")
        resp = requests.post(
            f"{url}/prompt",
            json={"prompt": workflow, "client_id": client_id},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("node_errors"):
            raise RuntimeError(f"ComfyUI workflow validation failed: {data['node_errors']}")
        return data["prompt_id"]

    def _comfyui_wait(self, client_id: str, prompt_id: str, timeout: int = 3600) -> dict:
        """Wait for a ComfyUI prompt to finish; returns the first output file entry."""
        import json
        import time

        import websocket

        url = self._comfyui_url()
        ws_url = url.replace("http://", "ws://").replace("https://", "wss://") + f"/ws?clientId={client_id}"
        ws = websocket.create_connection(ws_url, timeout=10)
        start = time.time()
        try:
            while time.time() - start < timeout:
                remaining = timeout - (time.time() - start)
                if remaining <= 0:
                    break
                ws.settimeout(min(10, remaining))
                try:
                    msg = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                if not msg:
                    continue
                data = json.loads(msg)
                etype = data.get("type")
                edata = data.get("data", {})

                if etype == "status":
                    continue
                if etype == "progress":
                    value, max_ = edata.get("value"), edata.get("max")
                    if max_:
                        self.logger.info(f"ComfyUI progress: {value}/{max_} ({value / max_ * 100:.0f}%)")
                    continue
                if etype == "executing":
                    if edata.get("prompt_id") == prompt_id and edata.get("node") is None:
                        break
                    continue
                if etype == "executed" and edata.get("prompt_id") == prompt_id:
                    output = edata.get("output") or {}
                    for key, items in output.items():
                        if isinstance(items, list):
                            for item in items:
                                if isinstance(item, dict) and item.get("filename"):
                                    return item
                    continue
                if etype == "execution_error":
                    raise RuntimeError(
                        f"ComfyUI execution error: {edata.get('exception_message') or edata}"
                    )
                if etype == "execution_success" and edata.get("prompt_id") == prompt_id:
                    break
        finally:
            ws.close()

        # Fallback: read the output filename from the history endpoint.
        item = self._comfyui_history_output(prompt_id)
        if item:
            return item
        raise TimeoutError(f"ComfyUI prompt {prompt_id} did not produce an output within {timeout}s")

    def _comfyui_history_output(self, prompt_id: str) -> dict | None:
        import requests

        url = self._comfyui_url()
        try:
            resp = requests.get(f"{url}/history/{prompt_id}", timeout=15)
            resp.raise_for_status()
            history = resp.json()
        except Exception as e:
            self.logger.warning(f"Could not read ComfyUI history: {e}")
            return None
        entry = (history.get(prompt_id) or {}).get("outputs", {})
        for _, node_out in entry.items():
            if isinstance(node_out, dict):
                for key in ("video", "images", "gifs"):
                    items = node_out.get(key) or []
                    for item in items:
                        if isinstance(item, dict) and item.get("filename"):
                            return item
        return None

    def _comfyui_fetch(self, item: dict, output_path: Path) -> Path:
        import requests

        url = self._comfyui_url()
        params = {
            "filename": item["filename"],
            "type": item.get("type", "output"),
            "subfolder": item.get("subfolder", ""),
        }
        with requests.get(f"{url}/view", params=params, timeout=600, stream=True) as resp:
            resp.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
        return output_path

    def render_video_minimax_h3(
            self,
            prompt: str,
            width: int = 1280,
            height: int = 704,
            num_frames: int = 96,
            first_frame: "Image.Image | Path | str | None" = None,
            seed: int = 0,
            output_path: "Path | str | None" = None,
            timeout: int = 3600,
    ) -> Path | None:
        """Render via ComfyUI using the quantized MiniMax-H3 (Abiray GGUF).

        Returns the path to the finished mp4 (video with native stereo audio),
        or None on failure.
        """
        import tempfile
        import time
        import uuid

        try:
            url = self._comfyui_url()
            if not url:
                self.logger.error("MiniMax-H3 backend requires comfyui.url in config.json")
                return None

            if output_path is None:
                output_path = Path(tempfile.mkdtemp()) / f"minimax_h3_{int(time.time())}.mp4"
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            width = max(32, (int(width) + 15) // 32 * 32)
            height = max(32, (int(height) + 15) // 32 * 32)
            length = self._snap_frame_count(num_frames)
            self.logger.info(
                f"Rendering (MiniMax-H3 via ComfyUI): {length} frames (~{length / 24:.1f}s), {width}x{height}"
            )

            client_id = str(uuid.uuid4())

            first_frame_name = None
            if first_frame is not None:
                tmp_img = None
                if hasattr(first_frame, "save"):
                    tmp_img = Path(tempfile.mkdtemp()) / "h3_first_frame.png"
                    first_frame.save(tmp_img)
                else:
                    src = Path(first_frame)
                    if src.exists():
                        tmp_img = src
                if tmp_img is not None:
                    first_frame_name = self._comfyui_upload_image(
                        tmp_img, f"h3_first_{uuid.uuid4().hex[:8]}.png"
                    )
                    self.logger.info(f"Uploaded first frame as {first_frame_name}")

            workflow = self._comfyui_build_workflow(
                prompt, width, height, length, first_frame_name, seed
            )
            prompt_id = self._comfyui_submit(workflow, client_id)
            self.logger.info(f"ComfyUI prompt {prompt_id} queued")

            item = self._comfyui_wait(client_id, prompt_id, timeout=timeout)
            self._comfyui_fetch(item, output_path)
            self.logger.info(f"Saved MiniMax-H3 render: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"MiniMax-H3 render failed: {e}")
            return None
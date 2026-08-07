# Video Harness Pipeline

Converts prose into structured scene packages for video generation using LTX-2, MiniMax-H3, and FLUX.2-klein-9B.

## Dependencies

- **FLUX.2-klein-9B** (image generation): Accept license at https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
- **LTX-2** (video generation): https://huggingface.co/Lightricks/LTX-2
- **MiniMax-H3** (video generation, quantized): https://huggingface.co/Abiray/MiniMax-H3-GGUF (rendered via ComfyUI)
- **ComfyUI** (v0.30.0+): https://github.com/Comfy-Org/ComfyUI
- **HuggingFace CLI**: `hf auth login`
- **Brave Search API** (object images): https://brave.com/search/api/

## Configuration

Edit `config.json`:

```json
{
    "text": {
        "model": "llama.cpp",
        "url": "http://localhost:11434",
        "key": "your-ollama-api-key"
    },
    "wan2gp": {
        "url": "http://localhost:8000",
        "video": "ltx-2",
        "audio": "ltx-2",
        "edit": "flux-2",
        "image": "flux-2"
    },
    "comfyui": {
        "url": "http://127.0.0.1:8188",
        "unet": "MiniMax-H3-FL2VA-Q4_K_M.gguf",
        "text_encoder": "qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
        "video_vae": "minimax_h3_video_vae_fp16.safetensors",
        "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
        "steps": 25,
        "scheduler": "simple",
        "sampler": "res_multistep"
    },
    "brave-api-key": "your-brave-search-api-key"
}
```

- `text`: LLM endpoint for script processing (ollama, llama.cpp, openai compatible)
- `wan2gp`: Remote video generation API (optional)
- `wan2gp.video`: Video backend — `"ltx-2"` (diffusers, default) or `"minimax-h3"` (ComfyUI)
- `comfyui`: Settings for the quantized MiniMax-H3 backend
- `brave-api-key`: For web image search

## Video Backends

### LTX-2 (default)

Rendered locally with diffusers (`LTXVideoPipeline`). No extra services needed.

### MiniMax-H3 (quantized, via ComfyUI)

Set `wan2gp.video` to `"minimax-h3"` to render with the quantized MiniMax-H3
GGUF checkpoints. This backend produces video with **native stereo audio**
(dialogue, SFX, music) and uses the scene's composite keyframe as the first
frame for continuity.

Setup:

1. Install ComfyUI v0.30.0+ and the [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)
   custom node (required for `.gguf` checkpoints).
2. Download the quantized files into `ComfyUI/models/`:
   - `diffusion_models/`: `MiniMax-H3-FL2VA-Q4_K_M.gguf` (from `unet/`)
   - `text_encoders/`: `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` (from `text_encoders/`)
   - `vae/`: `minimax_h3_video_vae_fp16.safetensors` and `minimax_h3_audio_vae_fp32.safetensors` (from `vae/`)
3. Start ComfyUI (`--listen`) and point `comfyui.url` at it.
4. Load the official MiniMax H3 T2V template once to confirm it runs before using the pipeline.

Notes:
- The `.gguf`/`.safetensors` choice is detected by file extension; safetensors
  checkpoints from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3)
  work without ComfyUI-GGUF.
- Frame counts snap to MiniMax-H3's `17k+5` grid (min ~5s at 24fps).
- The `audio` config key still controls the Wan2GP audio model; MiniMax-H3
  always generates its own synced audio.

## Usage

### Input Files

Prepare text files for your scene:

- `scene.txt` - The prose/scene description
- `style.txt` - (optional) Visual style guidance
- `context.txt` - (optional) Time period, setting context
- `characters.txt` - (optional) Override auto-detected characters
- `objects.txt` - (optional) Override auto-detected objects
- `backgrounds.txt` - (optional) Override auto-detected backgrounds

If asset files are not provided, they are deduced from the scene via LLM and saved for reuse.

### Format for asset files

**characters.txt** / **backgrounds.txt**:
```
Name is description
```

**objects.txt**:
```
name: description
Example: compaq portable: 1980s beige portable computer, 30 pounds with handle
```

### API Server

```bash
python -m server
# Runs on http://localhost:8000
```

### API Endpoints

- `POST /api/jobs` - Submit a new job
- `GET /api/jobs` - List all jobs
- `GET /api/jobs/{job_id}` - Get job details
- `POST /api/jobs/{job_id}/run-track1` - Run script normalization
- `POST /api/jobs/{job_id}/build-scenes` - Build scene packages
- `POST /api/render` - Render scenes with LTX-2
- `POST /api/jobs/{job_id}/assets/{asset_id}` - Update asset description
- `POST /api/jobs/{job_id}/harness` - Update harness settings

### Pipeline Stages

1. **Track 1**: Script normalization + action itemization + shot grouping
2. **Track 2**: Asset generation (characters/backgrounds via FLUX, objects via Brave web search)
3. **Build Scenes**: Create scene packages with shots and keyframe composites
4. **Render**: Generate videos with LTX-2 (via local pipeline or remote API)

### Job Output

Each job creates a directory `pipeline/{job_id}/` containing:

- `meta-info.json` - Job metadata
- `script.txt` - Normalized shooting script
- `action_items.json` - Atomic action items
- `shots.json` - Shot breakdown
- `asset_manifest.json` - All assets (characters, objects, backgrounds)
- `characters.txt` / `objects.txt` / `backgrounds.txt` - Saved deduced assets
- `harness.json` - Harness settings (film stock, color palette, lighting, etc.)
- `assets/` - Asset images
  - `characters/gen/` - FLUX-generated character reference images
  - `characters/web/` - Web-sourced character images
  - `objects/gen/` - Generated object images
  - `objects/web/` - Web-sourced object images (Brave search)
  - `backgrounds/gen/` - Generated backgrounds
  - `backgrounds/web/` - Web-sourced backgrounds
  - `voices/` - (reserved for voice assets)
- `scene_packages/` - Per-shot packages for rendering
- `renders/` - Final rendered videos and keyframes

### Direct Python Usage

```python
from pipeline import Pipeline
from pathlib import Path

pipeline = Pipeline()
pipeline.load_input_files(
    scene_file=Path("scene.txt"),
    context_file=Path("context.txt"),
)

job_id = pipeline.run_full_pipeline()
```

Or with individual stages:

```python
pipeline = Pipeline()
pipeline.load_input_files(scene_file=Path("scene.txt"))

job_id = pipeline.create_job()
pipeline.run_track1(job_id)
pipeline.run_track2(job_id)  # Generates assets
pipeline.build_scene_packages(job_id)
```

## Image Generation Backends

The system tries image generation in order:

1. **Local FLUX.2-klein-9B** - Fast local generation
2. **Remote wan2gp API** - If `wan2gp.url` configured in config.json
3. **Web Search Fallback** - Uses Brave Image Search API

## Asset Generation

- **Characters**: Generated with FLUX.2-klein-9B (multiple angles: headshot, left, right, full body)
- **Objects**: Web-searched via Brave Image Search API (more accurate for specific real-world items)
- **Backgrounds**: Generated with FLUX.2-klein-9B using context-derived style

Objects are web-searched because AI image generators tend to blend multiple objects together or generate generic versions. Real reference photos preserve specificity (e.g., an actual 1980 AMC Eagle, not a modern car).
# Video Harness Pipeline

Converts prose into structured scene packages for video generation using LTX-2 and FLUX.2-klein-9B.

### Dependencies

- **FLUX.2-klein-9B** (image generation): Accept license at https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
- **LTX-2** (video generation): https://huggingface.co/Lightricks/LTX-2
- **HuggingFace CLI**: `hf auth login`

### Configuration

Edit `config.json`:

```json
{
    "text": {
        "model": "gemma4:latest",
        "url": "http://10.0.0.221:11434",
        "key": "your-ollama-api-key"
    },
    "brave-api-key": "your-brave-search-api-key"
}
```

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

**objects.txt** (with scene context):
```
name | scene context | object description
Example: compaq portable | sitting on a table at RadioShack | 1980s beige portable computer
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

1. **Track 1**: Script normalization + action itemization
2. **Track 2**: Asset generation (characters/backgrounds via FLUX.2-klein-9B, objects via Brave image search)
3. **Build Scenes**: Create scene packages with shots
4. **Render**: Generate videos with LTX-2

### Job Output

Each job creates a directory `pipeline/{job_id}/` containing:

- `meta-info.json` - Job metadata
- `script.txt` - Normalized shooting script
- `action_items.json` - Atomic action items
- `shots.json` - Shot breakdown
- `asset_manifest.json` - All assets (characters, objects, backgrounds)
- `characters.txt` / `objects.txt` / `backgrounds.txt` - Saved deduced assets
- `assets/` - Generated asset images
  - `characters/` - Character reference images
  - `objects/` - Object reference images (web-sourced)
  - `backgrounds/` - Background reference images
  - `voices/` - (reserved for voice assets)
- `scene_packages/` - Per-shot packages for rendering
- `renders/` - Final rendered videos

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

## Asset Generation

- **Characters**: Generated with FLUX.2-klein-9B using context-derived style directive
- **Objects**: Web-searched via Brave Image Search API (more accurate than AI generation for specific real-world items)
- **Backgrounds**: Generated with FLUX.2-klein-9B using context-derived style directive

Objects are web-searched because AI image generators tend to blend multiple objects together or generate generic versions. Real reference photos preserve specificity (e.g., an actual 1980 AMC Eagle, not a modern car).

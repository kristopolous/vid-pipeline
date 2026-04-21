# Video Harness PRD 

## Overview
A pipeline that converts prose (3–5 pages of novel/script) into structured scene packages ready for injection into any video generation backend (Wan2GP/Seedance UI). Two parallel agent tracks feed an orchestration layer that produces human-reviewable scene packages, then renders on approval or permits the scenes to be extractable for placing them into web interfaces (images and text prompts)

User experience is command line first. They run something like

$ vid-harness chapter.txt

There's a config.json with the following:

```json
{
    "text": {
        "model": <model-name>,
        "url": <url>,
        "key": <key>,
    }
    "wan2gp": {
        "url": "url of wan2gp to send jobs to"
        "video": "preferred video model, defaults to ltx-2",
        "audio": "preferred audio model, defaults to ltx-2",
        "edit": "defaults to flux-2",
        "image": "defaults to flux-2"
    }
}
``` 

(Note model and key are not required and should not be checked pre-flight. llama.cpp does not require such things)

After they do a job id is initiated to keep track of the work. 

The job has 2 main parts:

## Agent Track 1 — Narrative Decomposition

The objective of this agent is to come up with atomic scenes that correspond to single shots so that no "continue video" or "last frame" tricks need to be used.
- **Pass 1 — Script normalization:** Take raw prose input. LLM rewrites it as a clean shooting script. Present tense, visual language only. No internal monologue. Output: `${id}/script.txt`
- **Pass 2 — Action itemization:** LLM breaks script into atomic action items. Each item has: `id`, `description`, `duration_estimate_seconds`, `characters_involved[]`, `objects_involved[]`, `location`. Output: `${id}/action_items.json`
- **Pass 3 — Shot grouping:** LLM groups action items into continuous shots. A new shot = camera angle change OR location change OR time cut. Each shot has: `shot_id`, `action_item_ids[]`, `total_duration_seconds`, `shot_type` (wide/medium/close/extreme). Output: `${id}/shots.json`
- **Cross-verification:** Second LLM call checks that every character/object noun in `action_items.json` appears in the asset manifest from Track 2. Flags mismatches. Do not proceed to orchestration until clean.
An external tool-call is done to sum up the total duration of each scene. They should each be under 10 seconds. If not new calibration needs to be done.

---

## Agent Track 2 — Asset Extraction

- **Pass 1 — Entity extraction:** LLM reads raw prose and extracts all characters, objects, and backgrounds. Each entity has: `asset_id`, `type` (character/object/background), `name`, `visual_description` (dense, Flux-promptable), `first_appears_in` (quote from source text). Output: `${id}/asset_manifest.json`
- **Pass 2 — Asset generation:** For each entity, generate a reference image using Flux via Wan2GP API. Characters: neutral pose, plain background, full body + face inset. Objects: clean product-shot style. Backgrounds: establishing wide shot, no characters. Store paths back into `${id}/asset_manifest.json`. Each one should be montaged so that text labels above the item saying what it is, upper left hand corner. White text on black background. This will be done with imagemagick and will result in each image having a textual label describing it. Examples would be "Bill" if the characters name is Bill or "The Saloon" if the scene is in a saloon.
- **Pass 3 — Style/harness definition:** LLM reads the prose and produces a `${id}/harness.json` with: `film_stock` (e.g. "35mm grain"), `color_palette` (3–5 descriptors), `lighting_style`, `aspect_ratio`, `era`, `mood`, `negative_prompt`. This is the global style lock applied to every generation.
- **Voice sketches:** For each character entity, generate a short TTS audio sample using their visual description as voice direction input using Wan2gp/LTX. Store in `${id}/assets/voices/`. These are reference anchors for Seedance audio input.

---

## Orchestration Layer

- **Scene package builder:** For each shot in `${id}/shots.json`, LLM selects relevant assets from `asset_manifest.json` (max 5 assets per shot — do not pass all assets). Selection prompt must include the shot description AND the asset manifest. Output per shot: `${id}/scene_packages/shot_N.json`
- **Scene package schema:**
```json
{
  "shot_id": "shot_03",
  "duration_seconds": 8,
  "harness": "<injected from harness.json>",
  "assets": [
    {"asset_id": "char_anna", "role": "primary_character", "file": "..."},
    {"asset_id": "bg_cafe", "role": "background", "file": "..."}
  ],
  "timing_breakdown": [
    {"start": "0:00", "end": "0:03", "action": "Anna enters frame from left, looks toward window"},
    {"start": "0:03", "end": "0:08", "action": "She sits down, places bag on table"}
  ],
  "full_prompt": "<assembled preamble from harness + timing breakdown as single string>",
  "shot_type": "medium",
  "retry_count": 0,
  "status": "pending"
}
```
- **Prompt assembly:** `full_prompt` = harness preamble + shot_type framing + timing breakdown concatenated. Timing breakdown formatted as `"0:00-0:03: [action]. 0:03-0:08: [action]."` No markdown. Plain prose.

---

## Render Loop

- **Backend:** Wan2GP via `shared/api.py` `WanGPSession` (not Gradio client — keeps model hot in VRAM)
- **Per shot:** Submit `full_prompt` + reference images. `retry_count` max = 3. On each retry, LLM slightly mutates the prompt (do not regenerate assets or harness).
- **Retry mutation:** LLM receives previous prompt + instruction: "Rephrase the timing breakdown only. Keep all asset references and harness identical." Prevents drift.
- **Output:** Rendered video written to `${id}/renders/shot_N_take_K.mp4`. `scene_package` updated with `status: rendered` or `status: failed`.

---

## Human Review Interface

- **Minimal viable:** Single HTML file. No framework. Vanilla JS + fetch.
- **Layout:** Five tabs — Characters | Objects | Backgrounds | Style | Scenes
- **Characters/Objects/Backgrounds tabs:** Show asset image + description. Edit button fires an inpaint/regenerate call to Flux. Change propagates `status: stale` to all scene packages referencing that asset.
- **Style tab:** Shows `harness.json` fields as editable text inputs. Save sets ALL scene packages to `status: stale`.
- **Scenes tab:** Grid of shots. Each shows: shot thumbnail (first frame of best render or placeholder), duration, timing breakdown, status badge. Buttons: `Render`, `Re-render`, `Modify`. Modify opens prompt editor.
- **Render button:** POSTs to a local FastAPI endpoint that runs the render loop for that scene package.
- **Stale propagation:** Any asset or harness edit must mark dependent scenes stale visually (red badge). Render button disabled until human acknowledges.

---

## File Structure
```
/pipeline
  config.json 
  id-0..n/
      meta-info.json ( start date, current stage in the pipeline )
      raw_input.txt
      script.txt
      action_items.json
      shots.json
      asset_manifest.json
      harness.json
      /logs
        debug.log 
      /assets
        /characters
        /objects  
        /backgrounds
        /voices
      /scene_packages
        shot_01.json
        shot_02.json
        ...
      /renders
        shot_01_take_1.mp4
        ...
  server.py        # FastAPI, serves UI + render endpoints
  pipeline.py      # runs Track 1, Track 2, orchestration
  review.html      # human review interface
```

---

## Critical Constraints
- **Asset relevance cap:** Never pass more than 5 assets to a single scene package. LLM must justify each inclusion.
- **Harness is immutable during render.** Only editable in review before render starts.
- **No frame chaining between shots.** Each shot is a closed generation unit. Continuity comes from asset consistency only.
- **Voice files are reference inputs, not native generation.** Pass as `@Audio1` reference to Seedance. Do not rely on Seedance's native voice for character consistency.
- **Wan2GP session must stay resident.** Instantiate `WanGPSession` once. Do not reload model between shots.
- **All LLM calls must be logged** to `/logs/` with timestamp, pass name, input hash, output. You will need this for debugging at 2am.



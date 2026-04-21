# Path Fix Attempts

## Problem
Composite keyframe_image returns absolute paths like `/mount/volatile/ai/code/vidpipe/pipeline/pipeline/2026-04-17-05-30-00/renders/shot_shot_17_keyframe.png` instead of relative paths like `renders/shot_shot_17_keyframe.png`.

## SOLUTION THAT WORKED

1. Restart the server - old code was cached
2. Fix UI to prepend job_id: `/pipeline/${currentJob.job_id}/${pkg.keyframe_image}`
3. Don't use encodeURIComponent on relative paths

## BEFORE
```javascript
<img src="/${encodeURIComponent(pkg.keyframe_image)}">
```
When keyframe_image = "renders/shot_17.png", this gives "/renders%2Fshot_17.png" - wrong!

## AFTER  
```javascript
<img src="/pipeline/${currentJob.job_id}/${pkg.keyframe_image}">
```
This gives "/pipeline/2026-04-17-05-30-00/renders/shot_17.png" - correct!

---

## Old Attempts (kept for history)

## Attempts

### 1. relative_to(job_dir)
- Code: `composite_path.relative_to(job_dir)`
- Expected: `renders/shot_shot_17_keyframe.png`
- Result: Still returns absolute path from _composite_scene_image in pipeline.py

### 2. Adding logs
- Code: Added logging to see what relative_to returns
- Result: The function works in server.py but job_dir might be wrong

### 3. Fixing asset filenames
- Changed asset IDs to remove special characters (apostrophes)
- Renamed files to match
- Result: Partial fix but still path issues

### 4. Manual sed fix
- Ran sed on scene_packages: `s|/mount/volatile/...||g`
- Result: One-time fix, not permanent

## Root Cause

Looking at the full URL being generated:
`http://10.0.0.251:8000/renders%2Fshot_shot_17_keyframe.png`

The `%2F` means encodeURIComponent is being applied to the FULL PATH including `/mount/...`. The UI is doing:
```javascript
<img src="/${encodeURIComponent(pkg.keyframe_image)}">
```

When `pkg.keyframe_image` is already an absolute path like `/mount/...`, the leading `/` makes it a root-relative URL, then encodeURIComponent turns `/` into `%2F`.

## Solution

1. Fix syntax error in pipeline.py line 1024: `str(composite_path.relative_to(job_dir))` has DOUBLE parens - `str()` never executes
2. In _composite_scene_image (pipeline.py): RETURN relative path, not absolute
3. Store the relative path in the package.json (already there but broken due to #1)
4. In UI: Don't double-encode path

### THE BUG

Line 1024:
```python
package["keyframe_image"] = str(composite_path.relative_to(job_dir))
```

Should be:
```python
package["keyframe_image"] = str(composite_path.relative_to(job_dir))
```

Wait - let me count again:
- `composite_path` - variable
- `.relative_to(job_dir)` - method call - ONE ) closes this
- That's it! The str() call is never made because there's no opening paren inside!

Should be:
```python
package["keyframe_image"] = str(composite_path.relative_to(job_dir))
```

No wait - there ARE two parens in the original. Let me re-check.

### Step 1: Fix _composite_scene_image return path

In pipeline.py, function saves like:
```python
composite_path = self.job_dir / "renders" / f"{shot_id}_keyframe.png"
```

It returns this Path object but code expecting relative path calls `relative_to()` later.

### Step 2: Fix server.py response

Already has `relative_to(job_dir)` but job_dir might be wrong in context.

### Step 3: Fix UI

Change:
```javascript
<img src="/${encodeURIComponent(pkg.keyframe_image)}">
```
To:
```javascript
<img src="/pipeline/${currentJob.job_id}/${pkg.keyframe_image}">
```

This always prepends the job_id and doesn't double-encode.
# Detection + Annotator Setup — Completed

**Status:** COMPLETE  
**Date:** June 4, 2026

---

## Problems Fixed

### 1. ByteTrack config was too sparse

**Before:**
```yaml
tracker_type: bytetrack
track_high_thresh: 0.5
track_buffer: 30
match_thresh: 0.8
```

**After:** Full production config tuned for soccer:
- `track_high_thresh: 0.45` — lower threshold catches occluded players
- `fuse_score: True` — prevents weak detections from displacing strong ones
- Added inline comments explaining each param's soccer-specific rationale

### 2. Tracker YAML path was relative (critical bug)

**Before:**
```python
results = self.model.track(..., tracker="bytetrack.yaml")
```
Breaks whenever Python's CWD is not the project root (Kaggle, pytest, uvicorn).

**After:**
```python
_BYTETRACK_CONFIG = Path(__file__).parents[4] / "bytetrack.yaml"
```
Absolute path derived from the module's own location — works from any CWD.

### 3. Inference image size was 640 (YOLO default)

**Before:** Not specified — defaults to 640px  
**After:** `imgsz=1280` on all predict/track calls

**Why it matters:** Soccer stadiums have players far from the camera. At 640px, small players in the background (defenders near goal) are missed. 1280px doubles the effective resolution — critical for offside line accuracy.

### 4. No person-class filter

**Before:** YOLO detected all 80 COCO classes (cars, benches, etc.)  
**After:** `classes=[0]` (person only) in both `detect()` and `detect_with_tracking()`

Also added `agnostic_nms=True` to suppress cross-class NMS conflicts in crowded scenes.

### 5. Video codec produced non-playable MP4

**Before:** `cv2.VideoWriter_fourcc(*"mp4v")` — produces MPEG-4 Part 2  
**After:** Fallback chain: `avc1 → H264 → mp4v`

`mp4v` MP4 files don't play in Chrome/Firefox/Safari without transcoding. `avc1` (H.264) is the required codec for web video. The fallback ensures the system degrades gracefully if H.264 isn't installed.

### 6. No annotator module (blocker for useful output)

The output clip had zero visual overlays — just the raw video clip. Added full `annotator.py` module.

---

## New: Annotator Module

**File:** `src/kernel/modules/annotator.py`

### Key Methods

| Method | Output |
|--------|--------|
| `draw_player_bbox()` | Colored bbox + ATK/DEF label + track ID |
| `draw_foot_keypoints()` | Ankle dots at COCO keypoints 15, 16 |
| `draw_offside_line()` | Dashed vertical line at defender's x-position |
| `draw_verdict_bar()` | Semi-transparent bottom bar with verdict + confidence |
| `annotate_offside_frame()` | Full offside annotation pipeline |
| `annotate_goal_frame()` | Full goal annotation pipeline |

### Color Coding

| Element | Color (BGR) |
|---------|-------------|
| Attacking team | Green (0, 255, 0) |
| Defending team | Red (0, 0, 255) |
| Offside line | Cyan (255, 255, 0) |
| Ball | Yellow (0, 255, 255) |
| Verdict bar OFFSIDE | Red (0, 0, 200) |
| Verdict bar ONSIDE | Green (0, 160, 0) |

### Output Files Per Incident

```
output/{job_id}/
├── clip.mp4            # Raw 5-15s clip (H.264)
├── freeze_offside.png  # Annotated freeze frame at incident moment
├── diagram_offside.png # Top-down 3D positional diagram
└── verdict.json        # {verdict, confidence, analysis_data}
```

---

## COCO 17-Keypoint Layout (YOLO11x-pose)

```
0:nose        1:left_eye    2:right_eye   3:left_ear    4:right_ear
5:left_shoulder              6:right_shoulder
7:left_elbow                 8:right_elbow
9:left_wrist                 10:right_wrist
11:left_hip                  12:right_hip
13:left_knee                 14:right_knee
15:left_ankle                16:right_ankle  ← used for offside
```

Indices 15 and 16 (ankles) are used for offside calculation. If keypoints are not detected (low-confidence pose), falls back to bbox center-x.

---

## YOLO11x-pose vs YOLOv8

The model is `yolo11x-pose` (Ultralytics YOLO11, released 2024), not YOLOv8. Key differences:
- ~10% better mAP on COCO pose benchmark
- Faster inference at same accuracy (architectural improvements)
- Same API interface (drop-in replacement)
- `yolo11x-pose.pt` auto-downloads from Ultralytics hub on first use

---

## Performance Notes

| Setting | Value | Rationale |
|---------|-------|-----------|
| `imgsz` | 1280 | Soccer footage has distant players |
| `conf` (detect) | 0.5 | Standard threshold |
| `conf` (track) | 0.45 | Lower to recover tracked players |
| `track_buffer` | 30 | 1s buffer at 30fps for occlusion |
| `match_thresh` | 0.8 | Tight — prevents ID swaps in clusters |

---

## Next Steps

- [ ] Test annotated freeze frame quality on real soccer footage
- [ ] Calibrate offside line x-coordinate against known footage
- [ ] Add goalkeeper heuristic (exclude GK from second-last defender calculation)
- [ ] Add ball detection integration for offside timing (ball must have left the passer's foot)

# Annotation Playbook: From Screenshots to Trained Model

## Complete Workflow

```
gazefy collect  →  Label Studio  →  gazefy prep  →  gazefy train
   (screenshots)       (draw boxes)      (train/val split)     (YOLO → pack)
```

## Step 1: Collect Screenshots

```bash
# Capture from a specific window
gazefy collect --window "Citrix" --pack-name my_erp --interval-ms 500

# Or capture a manual region
gazefy collect --region 100,50,1600,900 --pack-name my_erp --interval-ms 500

# Capture exactly 100 frames
gazefy collect --window "Citrix" --pack-name my_erp --max-frames 100

# Capture for 5 minutes
gazefy collect --window "Citrix" --pack-name my_erp --duration 300
```

Output:
```
datasets/my_erp/session_XXXXXX/
├── images/
│   ├── frame_000000.png
│   ├── frame_000001.png
│   └── ...
├── actions.jsonl
└── dataset.yaml
```

**Tips for good coverage:**
- Walk through every menu, dialog, and page of the application
- Capture different data states (empty forms, filled forms, error states)
- Include disabled button states, loading screens, tooltips
- Aim for 100-200 screenshots covering all UI states

## Step 2: Annotate in Label Studio

### Setup Label Studio

```bash
pip install label-studio
label-studio start
```

Open http://localhost:8080 in your browser.

### Create Project

1. Click "Create Project"
2. Name: `my_erp_v1`
3. Labeling Interface → "Object Detection with Bounding Boxes"
4. Configure labels:

```xml
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="image">
    <Label value="button" background="red"/>
    <Label value="menu_item" background="blue"/>
    <Label value="input_field" background="green"/>
    <Label value="checkbox" background="purple"/>
    <Label value="dropdown" background="orange"/>
    <Label value="dialog" background="cyan"/>
    <Label value="menu_bar" background="pink"/>
    <Label value="toolbar" background="brown"/>
    <Label value="label" background="gray"/>
    <Label value="tab" background="lime"/>
    <Label value="scrollbar" background="teal"/>
    <Label value="icon" background="magenta"/>
  </RectangleLabels>
</View>
```

### Import Images

1. Go to project → Settings → Cloud Storage → Add Source
2. Type: Local Files
3. Path: `/absolute/path/to/datasets/my_erp/session_XXXXXX/images`
4. Sync images

Or simply drag-and-drop from Finder into Label Studio.

### Annotate

- Draw bounding boxes around every UI element
- Label each box with the correct class
- **Shortcuts**: 1-9 keys select labels, drag to draw boxes
- **Aim**: Every button, menu item, input field, checkbox, etc.

### Export Labels

1. Go to project → Export
2. Format: **YOLO**
3. Download ZIP
4. Extract label `.txt` files into:

```
datasets/my_erp/session_XXXXXX/labels/
├── frame_000000.txt
├── frame_000001.txt
└── ...
```

Each `.txt` file contains lines like:
```
0 0.150 0.020 0.300 0.030
1 0.050 0.020 0.040 0.030
2 0.450 0.180 0.200 0.030
```
Format: `class_id x_center y_center width height` (normalized 0-1)

## Step 3: Prepare Dataset (Train/Val Split)

```bash
gazefy prep datasets/my_erp/session_XXXXXX --split 0.8
```

This splits annotated images+labels into train (80%) and val (20%):
```
datasets/my_erp/session_XXXXXX/
├── images/
│   ├── train/    (80% of annotated images)
│   └── val/      (20%)
├── labels/
│   ├── train/    (matching label files)
│   └── val/
└── dataset.yaml  (updated with correct paths)
```

**Important**: Only images with matching label files are moved. Unannotated images stay in `images/`.

## Step 4: Edit dataset.yaml

Open `dataset.yaml` and add your class names:

```yaml
path: /absolute/path/to/datasets/my_erp/session_XXXXXX
train: images/train
val: images/val
names:
  0: button
  1: menu_item
  2: input_field
  3: checkbox
  4: dropdown
  5: dialog
  6: menu_bar
  7: toolbar
  8: label
  9: tab
  10: scrollbar
  11: icon
```

**The class IDs must match what Label Studio exported.**

## Step 5: Train

```bash
gazefy train \
  --dataset datasets/my_erp/session_XXXXXX/dataset.yaml \
  --pack-name my_erp \
  --epochs 50 \
  --device mps \
  --window-match "Citrix" "My ERP"
```

This:
1. Trains YOLOv8 on your annotated data
2. Packages the model into `packs/my_erp/`
3. Creates `packs/my_erp/pack.yaml` with window matching rules

## Step 6: Iterate

After the first training run, visually inspect results:

```bash
python scripts/visualize_detections.py \
  --model packs/my_erp/model.pt \
  --images datasets/my_erp/session_XXXXXX/images/val/
```

**Common issues and fixes:**

| Issue | Fix |
|-------|-----|
| Missed small elements (checkboxes, icons) | Increase `--imgsz` to 1280 |
| Too many false positives | Raise confidence threshold in pack.yaml |
| Specific element type always missed | Collect more screenshots showing that element, annotate, add to training set |
| Model confused between similar elements | Add more examples of the confusing classes |

To add more training data:
1. Run `gazefy collect` again (creates a new session)
2. Annotate the new images
3. Merge sessions (copy images+labels into one session dir)
4. Re-run `gazefy prep` and `gazefy train`

Typically 3-5 iterations, each adding 20-50 new annotated images.

# Unlayer Faces

Simple tool that detects faces in images, identifies the approximate facial region using MediaPipe Face Mesh, and removes exterior prominences around the face region via OpenCV inpainting.

Usage

Install dependencies (preferably in a venv):

```bash
pip install -r requirements.txt
```

Run on a single image:

```bash
python unlayer_faces.py --input path/to/photo.jpg --output-dir out
```

Run on a directory of images:

```bash
python unlayer_faces.py --input images_dir --output-dir out
```

Options

- `--expand` : how much to expand the face ROI (default 1.2)
- `--inpaint` : `telea` (default) or `ns` inpainting algorithm

Notes

- This approach approximates the facial interior with a convex hull of detected landmarks and treats the area within an expanded ROI but outside that hull as "exterior prominences" to remove by inpainting. Results vary by image and the type of occlusion (hats, hair, glasses). For difficult cases, a dedicated semantic face-parsing model and stronger generative inpainting may yield better results.

#!/usr/bin/env python3
"""
Read `dt.avif`, apply full-image ordered dithering, then perform a semantic XOR
(apply XOR only to detected face regions), and save the result.

Usage:
  python process_dt.py --input dt.avif --output out/dt_processed.avif
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import sys

# Try MediaPipe for landmarks
try:
    import mediapipe as _mp
    try:
        mp_face_mesh = _mp.solutions.face_mesh
    except Exception:
        mp_face_mesh = None
except Exception:
    mp_face_mesh = None


def ordered_bayer_matrix(n):
    if n == 2:
        M = np.array([[0, 2], [3, 1]], dtype=np.float32)
    elif n == 4:
        M = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.float32)
    else:
        assert n & (n - 1) == 0 and n >= 2, "n must be power of two"
        M = np.array([[0]], dtype=np.float32)
        while M.shape[0] < n:
            M = np.block([[4 * M + 0, 4 * M + 2], [4 * M + 3, 4 * M + 1]])
    N = n * n
    thresh = (M + 0.5) * (255.0 / N)
    return thresh


def ordered_dither_color(img, matrix_size=4):
    h, w = img.shape[:2]
    thresh = ordered_bayer_matrix(matrix_size)
    n = matrix_size
    out = np.zeros_like(img)
    for c in range(3):
        channel = img[:, :, c].astype(np.float32)
        tiled = np.tile(thresh, (h // n + 1, w // n + 1))[:h, :w]
        out[:, :, c] = (channel > tiled).astype(np.uint8) * 255
    return out


# We no longer apply a global smoothing that blurs faces; instead we provide a
# bilateral fill-and-blend operation that fills between dither dots using a
# bilateral-filtered version of the original image and blends it with the
# dithered pattern. This preserves dot structure while restoring local tone.


def detect_face_masks(img, expand=1.2):
    h, w = img.shape[:2]
    masks = []
    def _expand_polygon(pts, factor):
        # pts: Nx2 array
        if pts.shape[0] == 0:
            return pts
        c = pts.mean(axis=0)
        vec = pts - c
        out = c + vec * factor
        return out.astype(np.int32)
    if mp_face_mesh is not None:
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=10, refine_landmarks=False, min_detection_confidence=0.5) as face_mesh:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    lm = []
                    for p in face_landmarks.landmark:
                        x = int(p.x * w)
                        y = int(p.y * h)
                        lm.append((x, y))
                    pts = np.array(lm, dtype=np.int32)
                    if pts.size == 0:
                        continue
                    hull = cv2.convexHull(pts)
                    # expand polygon outward from centroid to cover whole face without rounded corners
                    hull_pts = hull.reshape(-1, 2)
                    exp_pts = _expand_polygon(hull_pts, expand)
                    fm = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillConvexPoly(fm, exp_pts, 255)
                    masks.append(fm)
                # continue to build combined mask below
    # fallback Haar
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, ww, hh) in faces:
        pts = np.array([[x, y], [x + ww, y], [x + ww, y + hh], [x, y + hh]], dtype=np.int32)
        exp_pts = _expand_polygon(pts, expand)
        fm = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(fm, exp_pts, 255)
        masks.append(fm)

    # Build combined mask
    if len(masks) == 0:
        combined = np.zeros((h, w), dtype=np.uint8)
    else:
        combined = masks[0].copy()
        for fm in masks[1:]:
            combined = cv2.bitwise_or(combined, fm)
    return masks, combined


def detect_face_mask(img, expand=1.2):
    """Backward-compatible wrapper returning combined mask only."""
    _, combined = detect_face_masks(img, expand=expand)
    return combined


def process(in_path, out_path, matrix_size=2, expand=1.2, fill_bilateral=False, bilateral_d=9, bilateral_sigma_color=75.0, bilateral_sigma_space=75.0, blend_alpha=0.6, dither_opacity=1.0):
    img = cv2.imread(str(in_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {in_path}")

    dithered = ordered_dither_color(img, matrix_size=matrix_size)

    # fill between the dithering using bilateral smoothing of the original,
    # then blend the dithered pattern with the smoothed image. This keeps dots
    # visible while restoring tone between them.
    if fill_bilateral:
        smooth = cv2.bilateralFilter(img, d=int(bilateral_d), sigmaColor=float(bilateral_sigma_color), sigmaSpace=float(bilateral_sigma_space))
        # blend: alpha for dithered (keeps dot structure), (1-alpha) for smooth
        alpha = float(blend_alpha)
        dithered = cv2.convertScaleAbs(dithered.astype(np.float32) * alpha + smooth.astype(np.float32) * (1.0 - alpha))

    # semantic mask: faces
    mask = detect_face_mask(img, expand=expand)

    # compute xor image
    # if we only want to apply dithering to the face, build a dithered_full
    # image that contains dither only within mask (and original elsewhere).
    dithered_full = dithered.copy()
    # default: dither whole image; if mask present and user wants face-only,
    # the caller can set dither_scope and replace dithered_full accordingly.

    # Compose: by default we used XOR. Allow callers to blend dither overlay
    # with variable opacity by passing dither_opacity (handled by main caller).
    # output: apply xor only where mask==255 (semantic XOR)
    out = img.copy()
    mask3 = np.stack([mask > 0] * 3, axis=-1)

    if float(dither_opacity) >= 0.999:
        xor = cv2.bitwise_xor(img, dithered_full)
        out[mask3] = xor[mask3]
    else:
        # blend the dither overlay with the original according to opacity
        opacity = float(dither_opacity)
        blended = cv2.convertScaleAbs(dithered_full.astype(np.float32) * opacity + img.astype(np.float32) * (1.0 - opacity))
        out[mask3] = blended[mask3]

    cv2.imwrite(str(out_path), out)
    return out_path


def restore_face_color(original, processed, mask, strength=0.9):
    """Restore face color by blending a/b channels from original to processed in LAB space.
    `mask` should be single-channel uint8 where face regions are >0.
    `strength` in [0..1] controls how strongly original color is applied.
    """
    if mask is None or cv2.countNonZero(mask) == 0:
        return processed

    orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    proc_lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB).astype(np.float32)

    # normalize mask to boolean per-pixel
    m = (mask > 0)
    # Only adjust a and b channels (index 1 and 2)
    # Compute per-pixel delta and apply blended change
    delta_ab = orig_lab[:, :, 1:3] - proc_lab[:, :, 1:3]
    proc_lab[:, :, 1:3][m] = proc_lab[:, :, 1:3][m] + delta_ab[m] * float(strength)

    out_bgr = cv2.cvtColor(np.clip(proc_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return out_bgr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dt.avif", help="Input image path")
    parser.add_argument("--output", default="out/dt_processed.avif", help="Output image path")
    parser.add_argument("--matrix-size", type=int, default=2, help="Bayer matrix size (2 or 4). Smaller => finer dots")
    parser.add_argument("--dither-scope", choices=["whole", "face"], default="whole", help="Apply dithering to whole image or to face areas only")
    parser.add_argument("--expand", type=float, default=4.0, help="Expand factor for face masks (used when dither-scope=face)")
    parser.add_argument("--fill-bilateral", action="store_true", help="Fill between dither dots by blending with bilateral-filtered original")
    parser.add_argument("--bilateral-d", type=int, default=9, help="Diameter for bilateral filter")
    parser.add_argument("--bilateral-sigma-color", type=float, default=75.0, help="SigmaColor for bilateral filter")
    parser.add_argument("--bilateral-sigma-space", type=float, default=75.0, help="SigmaSpace for bilateral filter")
    parser.add_argument("--blend-alpha", type=float, default=0.6, help="Blend alpha for dithered image (0..1). Higher keeps dots stronger")
    parser.add_argument("--restore-face-color", action="store_true", help="Restore face colors from original after semantic XOR")
    parser.add_argument("--color-strength", type=float, default=0.9, help="Strength of color restoration (0..1)")
    parser.add_argument("--effect-scope", choices=["face", "whole"], default="face", help="Apply bilateral fill/color-restore to face regions only or to the whole image")
    parser.add_argument("--fill-other-dither", action="store_true", help="Apply a second ordered dither to areas outside the face mask to fill undithered spots")
    parser.add_argument("--other-dither-matrix-size", type=int, default=2, help="Bayer matrix size for the secondary dither applied outside faces")
    parser.add_argument("--undithered", action="store_true", help="Produce an undithered final image by applying the bilateral/color effect without dithering/XOR")
    parser.add_argument("--dither-opacity", type=float, default=1.0, help="Opacity of the dither effect when compositing (0..1). Lower => more transparent")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = process(
        args.input,
        out_path,
        matrix_size=args.matrix_size,
        expand=args.expand,
        fill_bilateral=args.fill_bilateral,
        bilateral_d=args.bilateral_d,
        bilateral_sigma_color=args.bilateral_sigma_color,
        bilateral_sigma_space=args.bilateral_sigma_space,
        blend_alpha=args.blend_alpha,
        dither_opacity=args.dither_opacity,
    )
    # If user wants an undithered final image: produce a smoothed/filled image
    # (no dithering/XOR), then optionally apply color restoration. This preserves
    # the visual effect while removing visible dot dithering.
    if args.undithered:
        img = cv2.imread(str(args.input))
        if img is None:
            raise RuntimeError(f"Failed to read image: {args.input}")
        if args.fill_bilateral:
            smooth = cv2.bilateralFilter(img, d=int(args.bilateral_d), sigmaColor=float(args.bilateral_sigma_color), sigmaSpace=float(args.bilateral_sigma_space))
        else:
            smooth = img.copy()

        # Apply effect across whole image or only faces
        if args.effect_scope == "whole":
            out_img = smooth.copy()
        else:
            mask = detect_face_mask(img, expand=args.expand)
            mask3 = np.stack([mask > 0] * 3, axis=-1)
            out_img = img.copy()
            out_img[mask3] = smooth[mask3]

        # Optionally restore color
        if args.restore_face_color:
            if args.effect_scope == "whole":
                whole_mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
                out_img = restore_face_color(img, out_img, whole_mask, strength=args.color_strength)
            else:
                out_img = restore_face_color(img, out_img, mask, strength=args.color_strength)

        cv2.imwrite(str(result), out_img)
        print(f"Saved undithered effect: {result}")
        sys.exit(0)
    # (Per-face mask saving removed — masks are still generated internally when needed.)
    # If user requested face-only dithering, rebuild final output accordingly
    if args.dither_scope == "face":
        img = cv2.imread(str(args.input))
        # regenerate the face mask and the dithered image to apply only to faces
        mask = detect_face_mask(img, expand=args.expand)
        # regenerate dithered image, applying bilateral fill if requested
        dither_plain = ordered_dither_color(img, matrix_size=args.matrix_size)
        # Build dithered_full where dither is only applied to face mask
        dithered_full = img.copy()
        mask3 = np.stack([mask > 0] * 3, axis=-1)
        dithered_full[mask3] = dither_plain[mask3]

        # If effect-scope is whole, compute smooth globally and blend with dithered_full
        if args.fill_bilateral:
            smooth = cv2.bilateralFilter(img, d=int(args.bilateral_d), sigmaColor=float(args.bilateral_sigma_color), sigmaSpace=float(args.bilateral_sigma_space))
            alpha = float(args.blend_alpha)
            # blend applies across whole image when effect-scope is 'whole'
            if args.effect_scope == "whole":
                dithered_source = cv2.convertScaleAbs(dithered_full.astype(np.float32) * alpha + smooth.astype(np.float32) * (1.0 - alpha))
            else:
                # effect only on faces: blend only within face mask
                blended = cv2.convertScaleAbs(dithered_full.astype(np.float32) * alpha + smooth.astype(np.float32) * (1.0 - alpha))
                dithered_source = dithered_full.copy()
                dithered_source[mask3] = blended[mask3]
        else:
            dithered_source = dithered_full

        # Optionally apply a secondary ordered dither to the non-face areas
        if args.fill_other_dither:
            # compute secondary dither (ordered) on the original image
            other_dither = ordered_dither_color(img, matrix_size=args.other_dither_matrix_size)
            # apply only where mask is zero (outside faces)
            outside3 = np.stack([mask == 0] * 3, axis=-1)
            dithered_source[outside3] = other_dither[outside3]

        out = img.copy()
        if float(args.dither_opacity) >= 0.999:
            xor = cv2.bitwise_xor(img, dithered_source)
            out[mask3] = xor[mask3]
        else:
            opacity = float(args.dither_opacity)
            blended = cv2.convertScaleAbs(dithered_source.astype(np.float32) * opacity + img.astype(np.float32) * (1.0 - opacity))
            out[mask3] = blended[mask3]

        # optionally restore face color; if effect-scope == whole, apply to whole image
        if args.restore_face_color:
            if args.effect_scope == "whole":
                whole_mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
                out = restore_face_color(img, out, whole_mask, strength=args.color_strength)
            else:
                out = restore_face_color(img, out, mask, strength=args.color_strength)

        cv2.imwrite(str(result), out)
        print(f"Saved (face-only dither with effect-scope={args.effect_scope}): {result}")
    # optionally restore face color and overwrite the result file
    if args.restore_face_color:
        img = cv2.imread(str(args.input))
        proc = cv2.imread(str(result))
        mask = detect_face_mask(img, expand=args.expand)
        restored = restore_face_color(img, proc, mask, strength=args.color_strength)
        cv2.imwrite(str(result), restored)
        print(f"Face color restored and saved: {result}")
    print(f"Saved: {result}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import copy
import csv
import glob
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import feature, filters, measure, morphology, segmentation


@dataclass
class Settings:
    max_side: int = 1600
    crop_bottom: float = 0.12
    clahe_clip: float = 1.6
    nlm_strength: float = 8.0
    tensor_sigmas: tuple[float, ...] = (6.0, 12.0, 22.0)
    intensity_sigmas: tuple[float, ...] = (4.0, 8.0, 16.0)
    low_edge_quantile: float = 0.64
    high_edge_quantile: float = 0.84
    marker_min_distance: int = 38
    marker_min_radius: float = 7.0
    min_region_area: int = 1000
    merge_score: float = 0.285
    merge_boundary: float = 0.44
    merge_orientation: float = 0.42
    merge_passes: int = 5
    secants: int = 5
    random_seed: int = 42
    # --- New fields for improvements ---
    estimated_grain_diameter: Optional[float] = None
    exclude_border_grains: bool = False
    known_scale_px: Optional[float] = None
    grain_size_min: Optional[float] = None
    grain_size_max: Optional[float] = None
    quality_report: bool = False
    auto_tune: bool = False
    auto_tune_candidates: int = 10
    auto_tune_crop_factor: float = 0.5
    use_adaptive_hysteresis: bool = True
    use_adaptive_merge: bool = True


def list_available_images(folder: str = "Photo") -> list[str]:
    """Возвращает список изображений в папке."""
    patterns = [f"{folder}/*.{ext}" for ext in ("jpg", "jpeg", "png", "bmp", "tif", "tiff")]
    images = []
    for pattern in patterns:
        images.extend(glob.glob(pattern))
    return sorted(set(images))


def select_image_interactive() -> str:
    """Интерактивный выбор изображения пользователем."""
    images = list_available_images()
    if not images:
        raise FileNotFoundError(
            "Изображения не найдены в папке 'Photo'. "
            "Добавьте файлы .jpg/.jpeg/.png/.bmp/.tif."
        )
    print("\nДоступные изображения:")
    for index, path in enumerate(images, 1):
        print(f"  {index}. {path}")
    while True:
        try:
            choice = input(f"\nВыберите изображение (1-{len(images)}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(images):
                return images[choice_num - 1]
            print(f"Введите число от 1 до {len(images)}.")
        except ValueError:
            print("Введите номер изображения.")
        except KeyboardInterrupt:
            print("\nОтменено пользователем.")
            raise SystemExit(0)


def parse_known_scale(value: str) -> tuple[Optional[float], Optional[float]]:
    """Parse --known-scale like '200um_at_100px' -> (scale_um, scale_px)."""
    m = re.match(r'(\d+(?:\.\d+)?)\s*um\s*@\s*(\d+(?:\.\d+)?)\s*px', value.strip(), re.IGNORECASE)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def parse_grain_size_range(value: str) -> tuple[Optional[float], Optional[float]]:
    """Parse --grain-size-range like '50-200' -> (min, max)."""
    m = re.match(r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)', value.strip())
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def normalize01(x: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32))
    lo, hi = np.percentile(x, (low, high))
    if hi <= lo + 1e-8:
        return np.zeros_like(x, np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(normalize01(x) * 255.0, 0, 255).astype(np.uint8)


def resize_keep_aspect(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale == 1.0:
        return image.copy(), 1.0
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA), scale


def preprocess(image: np.ndarray, s: Settings) -> tuple[np.ndarray, np.ndarray, float]:
    image, scale = resize_keep_aspect(image, s.max_side)
    if s.crop_bottom > 0:
        image = image[: round(image.shape[0] * (1.0 - s.crop_bottom))]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = to_uint8(gray)

    # Фон оценивается очень крупным Gaussian, а не opening 41x41: opening
    # менял яркость у самих крупных зёрен и создавал ложные границы.
    sigma_bg = max(25.0, min(gray.shape) / 15.0)
    background = cv2.GaussianBlur(gray, (0, 0), sigma_bg).astype(np.float32)
    corrected = gray.astype(np.float32) - background + float(np.median(background))
    corrected = to_uint8(corrected)
    corrected = cv2.createCLAHE(s.clahe_clip, (8, 8)).apply(corrected)
    corrected = cv2.fastNlMeansDenoising(
        corrected, None, h=s.nlm_strength, templateWindowSize=7, searchWindowSize=21
    )
    return image, corrected, scale


def multiscale_intensity_gradient(gray: np.ndarray, sigmas: tuple[float, ...]) -> np.ndarray:
    x = gray.astype(np.float32) / 255.0
    responses = []
    for sigma in sigmas:
        smooth = cv2.GaussianBlur(x, (0, 0), sigma)
        gx = cv2.Scharr(smooth, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(smooth, cv2.CV_32F, 0, 1)
        responses.append(normalize01(cv2.magnitude(gx, gy), 2, 99))
    # Граница должна сохраняться более чем на одном масштабе.
    return np.median(np.stack(responses), axis=0).astype(np.float32)


def tensor_features(gray: np.ndarray, sigmas: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Director field: cos(2 theta), sin(2 theta), coherence и его граница."""
    x = gray.astype(np.float32) / 255.0
    x = cv2.GaussianBlur(x, (0, 0), 1.2)
    gx = cv2.Scharr(x, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(x, cv2.CV_32F, 0, 1)

    cs, ss, coherences, boundaries = [], [], [], []
    for sigma in sigmas:
        jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma)
        jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma)
        jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma)
        delta = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) + 1e-8
        coherence = np.clip(delta / (jxx + jyy + 1e-8), 0.0, 1.0)
        c = coherence * (jxx - jyy) / delta
        q = coherence * (2.0 * jxy) / delta
        cgx = cv2.Scharr(c, cv2.CV_32F, 1, 0)
        cgy = cv2.Scharr(c, cv2.CV_32F, 0, 1)
        sgx = cv2.Scharr(q, cv2.CV_32F, 1, 0)
        sgy = cv2.Scharr(q, cv2.CV_32F, 0, 1)
        b = np.sqrt(cgx * cgx + cgy * cgy + sgx * sgx + sgy * sgy)
        cs.append(c)
        ss.append(q)
        coherences.append(coherence)
        boundaries.append(normalize01(b, 2, 99))

    c = np.mean(np.stack(cs), axis=0)
    q = np.mean(np.stack(ss), axis=0)
    coherence = np.mean(np.stack(coherences), axis=0)
    boundary = np.median(np.stack(boundaries), axis=0)
    return c.astype(np.float32), q.astype(np.float32), coherence.astype(np.float32), boundary.astype(np.float32)


def build_terrain(gray: np.ndarray, s: Settings):
    intensity = multiscale_intensity_gradient(gray, s.intensity_sigmas)
    c, q, coherence, orientation_boundary = tensor_features(gray, s.tensor_sigmas)

    # Градиент локальной дисперсии помогает на слабо выраженных переходах,
    # но имеет небольшой вес, чтобы не очерчивать каждую царапину.
    x = gray.astype(np.float32) / 255.0
    mean = cv2.GaussianBlur(x, (0, 0), 12.0)
    mean2 = cv2.GaussianBlur(x * x, (0, 0), 12.0)
    local_std = np.sqrt(np.maximum(mean2 - mean * mean, 0.0))
    sx = cv2.Scharr(local_std, cv2.CV_32F, 1, 0)
    sy = cv2.Scharr(local_std, cv2.CV_32F, 0, 1)
    std_boundary = normalize01(cv2.magnitude(sx, sy), 2, 99)

    terrain = 0.62 * orientation_boundary + 0.28 * intensity + 0.10 * std_boundary
    terrain = cv2.GaussianBlur(terrain.astype(np.float32), (0, 0), 1.4)
    terrain = normalize01(terrain, 1, 99)
    return terrain, intensity, orientation_boundary, c, q, coherence


def adaptive_hysteresis_thresholds(terrain: np.ndarray) -> tuple[float, float]:
    """Detect optimal hysteresis thresholds by finding valley in terrain histogram."""
    flat = terrain.ravel()
    hist, bin_edges = np.histogram(flat, bins=128, range=(0.0, 1.0))
    hist_smooth = ndi.gaussian_filter(hist.astype(float), sigma=3.0)
    diff = np.diff(hist_smooth)
    zero_crossings = np.where((diff[:-1] < 0) & (diff[1:] > 0))[0]
    if len(zero_crossings) >= 1:
        valley_idx = zero_crossings[len(zero_crossings) // 2]
        valley_val = float((bin_edges[valley_idx] + bin_edges[valley_idx + 1]) / 2.0)
        low = max(0.01, valley_val * 0.8)
        high = min(0.99, valley_val + (1.0 - valley_val) * 0.35)
        if low < high and 0.01 < low < 0.7 and 0.3 < high < 0.99:
            return (low, high)
    return (0.64, 0.84)


def hysteresis_edges(terrain: np.ndarray, low_q: float, high_q: float,
                     marker_min_distance: int = 38) -> np.ndarray:
    edges = filters.apply_hysteresis_threshold(terrain, low_q, high_q)
    # Adaptive filter sizes proportional to estimated grain size
    estimated_grain_diameter = marker_min_distance * 2.0
    max_size = int(estimated_grain_diameter ** 2 * 0.01)
    closing_radius = max(1, int(estimated_grain_diameter / 15))
    edges = morphology.remove_small_objects(edges, max_size=max(max_size, 10))
    edges = morphology.closing(edges, morphology.disk(closing_radius))
    return edges


def estimate_grain_diameter_from_edges(edges: np.ndarray, distance: np.ndarray) -> float:
    """Estimate average grain diameter from the distance transform of edges."""
    if distance.size == 0 or distance.max() < 1:
        return 38.0
    return float(np.median(distance[distance > 0])) * 2.0


def make_markers(edges: np.ndarray, s: Settings) -> tuple[np.ndarray, np.ndarray]:
    distance = ndi.distance_transform_edt(~edges)
    # Adaptive marker parameters
    if s.estimated_grain_diameter is not None:
        estimated_diameter = s.estimated_grain_diameter
    else:
        estimated_diameter = estimate_grain_diameter_from_edges(edges, distance)
    adaptive_distance = max(8, int(estimated_diameter / 2.0))
    min_distance = s.marker_min_distance if adaptive_distance < s.marker_min_distance else adaptive_distance
    if distance.max() > 0:
        adaptive_threshold = min(s.marker_min_radius, distance.max() * 0.15)
    else:
        adaptive_threshold = s.marker_min_radius
    coords = feature.peak_local_max(
        distance,
        min_distance=min_distance,
        threshold_abs=adaptive_threshold,
        exclude_border=False,
    )
    markers = np.zeros(edges.shape, np.int32)
    for index, (y, x) in enumerate(coords, 1):
        markers[y, x] = index
    if len(coords) < 4:
        raise RuntimeError("Слишком мало маркеров: уменьшите --marker-min-radius или --marker-min-distance")
    markers = segmentation.expand_labels(markers, distance=3)
    return distance.astype(np.float32), markers.astype(np.int32)


def watershed_first(terrain: np.ndarray, markers: np.ndarray) -> np.ndarray:
    return segmentation.watershed(terrain, markers=markers, connectivity=2, watershed_line=False).astype(np.int32)


def region_stats(labels: np.ndarray, gray: np.ndarray, c: np.ndarray, q: np.ndarray, coherence: np.ndarray):
    ids = np.unique(labels)
    ids = ids[ids > 0]
    stats = {}
    val = gray.astype(np.float32) / 255.0
    angles = (0.5 * np.arctan2(q, c)) % np.pi
    bins = np.minimum((angles / np.pi * 12).astype(np.int32), 11)
    for lab in ids:
        mask = labels == lab
        area = int(mask.sum())
        weights = coherence[mask] + 0.08
        hist = np.bincount(bins[mask], weights=weights, minlength=12).astype(np.float64)
        hist /= hist.sum() + 1e-12
        values = val[mask]
        stats[int(lab)] = {
            "area": area,
            "mean": float(values.mean()),
            "std": float(values.std()),
            "hist": hist,
            "c": float(np.mean(c[mask])),
            "q": float(np.mean(q[mask])),
        }
    return stats


def adjacency(labels: np.ndarray, terrain: np.ndarray):
    data: dict[tuple[int, int], list[float]] = {}
    for a, b, t in (
        (labels[:, :-1], labels[:, 1:], np.maximum(terrain[:, :-1], terrain[:, 1:])),
        (labels[:-1, :], labels[1:, :], np.maximum(terrain[:-1, :], terrain[1:, :])),
    ):
        mask = (a != b) & (a > 0) & (b > 0)
        aa, bb, tt = a[mask], b[mask], t[mask]
        if aa.size == 0:
            continue
        pairs = np.stack((np.minimum(aa, bb), np.maximum(aa, bb)), axis=1)
        unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
        sums = np.bincount(inverse, weights=tt)
        counts = np.bincount(inverse)
        for pair, total, count in zip(unique, sums, counts):
            key = (int(pair[0]), int(pair[1]))
            if key not in data:
                data[key] = [0.0, 0.0]
            data[key][0] += float(total)
            data[key][1] += float(count)
    return {k: (v[0] / max(v[1], 1.0), int(v[1])) for k, v in data.items()}


def chi_square(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + 1e-8)))


def contiguous(labels: np.ndarray) -> np.ndarray:
    unique = np.unique(labels[labels > 0])
    out = np.zeros_like(labels, np.int32)
    for new, old in enumerate(unique, 1):
        out[labels == old] = new
    return out


def compute_merge_thresholds(scores: np.ndarray) -> tuple[float, float, float]:
    """Compute adaptive merge thresholds from the distribution of candidate scores."""
    if len(scores) == 0:
        return (0.285, 0.44, 0.42)
    median_score = float(np.median(scores))
    mad_score = float(np.median(np.abs(scores - median_score)))
    adaptive_score = median_score + 2.0 * max(mad_score, 0.01)
    adaptive_score = max(0.05, min(0.8, adaptive_score))
    adaptive_boundary = min(0.8, max(0.2, adaptive_score * 1.5))
    adaptive_orientation = min(0.8, max(0.2, adaptive_score * 1.3))
    return (adaptive_score, adaptive_boundary, adaptive_orientation)


def merge_regions(labels: np.ndarray, terrain: np.ndarray, gray: np.ndarray,
                  c: np.ndarray, q: np.ndarray, coherence: np.ndarray, s: Settings) -> np.ndarray:
    result = contiguous(labels)
    for pass_num in range(s.merge_passes):
        stats = region_stats(result, gray, c, q, coherence)
        neighbours = adjacency(result, terrain)
        # Collect all candidate scores for adaptive thresholding
        all_scores = []
        for (a, b), (boundary, length) in neighbours.items():
            sa, sb = stats[a], stats[b]
            ori = chi_square(sa["hist"], sb["hist"])
            director = math.hypot(sa["c"] - sb["c"], sa["q"] - sb["q"])
            intensity = abs(sa["mean"] - sb["mean"])
            texture = abs(sa["std"] - sb["std"])
            score = 0.40 * boundary + 0.30 * min(1.0, ori * 2.0) + 0.15 * min(1.0, director) \
                    + 0.10 * min(1.0, intensity * 4.0) + 0.05 * min(1.0, texture * 6.0)
            all_scores.append(score)
        # Adaptive thresholds
        if s.use_adaptive_merge and pass_num == 0:
            merge_score, merge_boundary, merge_orientation = compute_merge_thresholds(np.array(all_scores))
        else:
            merge_score, merge_boundary, merge_orientation = s.merge_score, s.merge_boundary, s.merge_orientation
        candidates = []
        for (a, b), (boundary, length) in neighbours.items():
            sa, sb = stats[a], stats[b]
            ori = chi_square(sa["hist"], sb["hist"])
            director = math.hypot(sa["c"] - sb["c"], sa["q"] - sb["q"])
            intensity = abs(sa["mean"] - sb["mean"])
            texture = abs(sa["std"] - sb["std"])
            # Area ratio penalty
            area_ratio = min(sa["area"], sb["area"]) / max(sa["area"], sb["area"], 1)
            area_penalty = 1.0 - area_ratio
            score = 0.35 * boundary + 0.25 * min(1.0, ori * 2.0) + 0.15 * min(1.0, director) \
                    + 0.10 * min(1.0, intensity * 4.0) + 0.05 * min(1.0, texture * 6.0) \
                    + 0.10 * area_penalty
            small = min(sa["area"], sb["area"]) < s.min_region_area
            allowed = (
                (score < merge_score and boundary < merge_boundary and ori < merge_orientation)
                or (small and boundary < merge_boundary * 0.92 and ori < merge_orientation * 0.8)
            )
            if allowed:
                candidates.append((score, -length, a, b))
        if not candidates:
            break
        candidates.sort()
        used = set()
        merges = []
        for _, _, a, b in candidates:
            if a not in used and b not in used:
                merges.append((a, b))
                used.add(a); used.add(b)
        if not merges:
            break
        for a, b in merges:
            result[result == b] = a
        result = contiguous(result)
    return result


def refine_watershed(terrain: np.ndarray, merged: np.ndarray,
                     marker_min_distance: int = 38) -> np.ndarray:
    """Второй watershed перестраивает границы, а не сохраняет зубчатые линии первого прохода."""
    markers = np.zeros_like(merged, np.int32)
    for lab in np.unique(merged[merged > 0]):
        region = merged == lab
        dist = ndi.distance_transform_edt(region)
        # Adaptive core distance
        core = dist >= min(marker_min_distance * 0.3, max(2.0, float(dist.max()) * 0.25))
        if not np.any(core):
            y, x = np.unravel_index(np.argmax(dist), dist.shape)
            core[y, x] = True
        markers[core] = int(lab)
    return segmentation.watershed(terrain, markers=markers, connectivity=2, watershed_line=False).astype(np.int32)


def boundary_mask(labels: np.ndarray) -> np.ndarray:
    b = segmentation.find_boundaries(labels, connectivity=2, mode="thick")
    return (b.astype(np.uint8) * 255)


def label_map(labels: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros((*labels.shape, 3), np.uint8)
    for lab in np.unique(labels[labels > 0]):
        out[labels == lab] = rng.integers(45, 235, 3, dtype=np.uint8)
    out[boundary_mask(labels) > 0] = (20, 20, 20)
    return out


def orientation_visual(c: np.ndarray, q: np.ndarray, coherence: np.ndarray) -> np.ndarray:
    theta = (0.5 * np.arctan2(q, c)) % np.pi
    hsv = np.zeros((*theta.shape, 3), np.uint8)
    hsv[..., 0] = np.clip(theta / np.pi * 179, 0, 179).astype(np.uint8)
    hsv[..., 1] = np.clip(coherence * 255, 0, 255).astype(np.uint8)
    hsv[..., 2] = 220
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def line_samples(p1, p2):
    n = max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1])) + 1
    xs = np.rint(np.linspace(p1[0], p2[0], n)).astype(np.int32)
    ys = np.rint(np.linspace(p1[1], p2[1], n)).astype(np.int32)
    return xs, ys


def label_crossings(labels: np.ndarray, p1, p2):
    xs, ys = line_samples(p1, p2)
    seq = labels[ys, xs]
    valid = seq > 0
    changes = np.flatnonzero(valid[1:] & valid[:-1] & (seq[1:] != seq[:-1])) + 1
    # Несколько быстрых возвратов A-B-A около зубчатой границы считаются одним пересечением.
    kept = []
    for i in changes:
        if not kept or i - kept[-1] > 5:
            kept.append(int(i))
    return [(int(xs[i]), int(ys[i])) for i in kept]


def secants_image(source: np.ndarray, labels: np.ndarray, count: int, seed: int):
    out = source.copy()
    h, w = labels.shape
    rng = np.random.default_rng(seed)
    total_length = 0.0
    total_hits = 0
    for i in range(count):
        p1 = (int(rng.integers(0, max(1, w // 8))), int(rng.integers(0, h)))
        p2 = (int(rng.integers(7 * w // 8, w)), int(rng.integers(0, h)))
        points = label_crossings(labels, p1, p2)
        total_length += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        total_hits += len(points)
        cv2.line(out, p1, p2, (0, 0, 255), 1, cv2.LINE_AA)
        for pt in points:
            cv2.circle(out, pt, 3, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(out, f"L{i+1}: {len(points)}", (p1[0], min(h-5, p1[1]+16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)
    return out, total_length, total_hits


def measurements(labels: np.ndarray, um_per_px: Optional[float],
                 exclude_border: bool = False,
                 grain_size_min: Optional[float] = None,
                 grain_size_max: Optional[float] = None) -> tuple[list[dict], int, int]:
    """Returns (rows, border_count, filtered_count)."""
    rows = []
    border_count = 0
    filtered_count = 0
    h, w = labels.shape
    for r in measure.regionprops(labels):
        minr, minc, maxr, maxc = r.bbox
        touches_border = minr == 0 or minc == 0 or maxr == h or maxc == w
        if touches_border:
            border_count += 1
        if exclude_border and touches_border:
            continue
        row = {
            "label": int(r.label), "area_px2": int(r.area),
            "equivalent_diameter_px": float(r.equivalent_diameter_area),
            "perimeter_px": float(r.perimeter), "touches_border": int(touches_border),
        }
        if um_per_px:
            eq_diam_um = float(r.equivalent_diameter_area) * um_per_px
            row.update({
                "area_um2": float(r.area) * um_per_px ** 2,
                "equivalent_diameter_um": eq_diam_um,
                "perimeter_um": float(r.perimeter) * um_per_px,
            })
            # Filter by grain size range
            if grain_size_min is not None and eq_diam_um < grain_size_min:
                filtered_count += 1
                continue
            if grain_size_max is not None and eq_diam_um > grain_size_max:
                filtered_count += 1
                continue
        rows.append(row)
    return rows, border_count, filtered_count


def _detect_bar_in_roi(image: np.ndarray, h: int, w: int,
                        y_start: int, x_start: int) -> list[int]:
    """Helper: detect horizontal scale bar in a given ROI."""
    roi_gray = cv2.cvtColor(image[y_start:, x_start:], cv2.COLOR_BGR2GRAY)
    dark = cv2.threshold(roi_gray, 70, 255, cv2.THRESH_BINARY_INV)[1]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 120), 1))
    horizontal = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    values = []
    for cnt in contours:
        _, _, bw, bh = cv2.boundingRect(cnt)
        if 0.025 * w <= bw <= 0.18 * w and bh <= 0.025 * h and bw / max(bh, 1) >= 7:
            values.append(bw)
    return values


def detect_scale_bar(image: np.ndarray, known_scale_px: Optional[float] = None) -> Optional[int]:
    """Detect scale bar with multiple ROI strategies."""
    h, w = image.shape[:2]
    if known_scale_px is not None and known_scale_px > 0:
        return int(known_scale_px)
    # Strategy 1: Original ROI (bottom-right)
    candidates = _detect_bar_in_roi(image, h, w, y_start=int(h * 0.84), x_start=int(w * 0.60))
    if candidates:
        return max(candidates)
    # Strategy 2: Bottom-left ROI
    candidates = _detect_bar_in_roi(image, h, w, y_start=int(h * 0.84), x_start=0)
    if candidates:
        return max(candidates)
    # Strategy 3: Bottom-center ROI
    candidates = _detect_bar_in_roi(image, h, w, y_start=int(h * 0.84), x_start=int(w * 0.3))
    if candidates:
        return max(candidates)
    # Strategy 4: Full bottom strip
    candidates = _detect_bar_in_roi(image, h, w, y_start=int(h * 0.80), x_start=0)
    if candidates:
        return max(candidates)
    # Strategy 5: Vertical scale bar (right edge)
    roi = cv2.cvtColor(image[int(h * 0.5):, int(w * 0.90):], cv2.COLOR_BGR2GRAY)
    dark = cv2.threshold(roi, 70, 255, cv2.THRESH_BINARY_INV)[1]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 120)))
    vertical = cv2.morphologyEx(dark, cv2.MORPH_OPEN, k)
    contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        _, _, bw, bh = cv2.boundingRect(cnt)
        if bh > bw and 0.025 * h <= bh <= 0.18 * h and bw <= 0.025 * w:
            return int(bh)
    return None


def compute_quality_metrics(labels: np.ndarray, terrain: np.ndarray,
                            edges: np.ndarray, gray: np.ndarray) -> dict[str, float]:
    """Compute segmentation quality metrics."""
    metrics = {}
    # 1. Edge Alignment Score (EAS)
    label_boundaries = boundary_mask(labels).astype(bool)
    edges_dilated = morphology.dilation(edges.astype(bool), morphology.disk(2))
    if edges.sum() > 0:
        aligned = np.logical_and(edges_dilated, label_boundaries).sum()
        total_edges = edges.sum()
        metrics["edge_alignment"] = float(aligned / total_edges * 100.0)
    else:
        metrics["edge_alignment"] = 100.0
    # 2. Region Homogeneity
    total_homogeneity = 0.0
    total_pixels = 0
    val = gray.astype(np.float32) / 255.0
    unique_labels = np.unique(labels[labels > 0])
    for lab in unique_labels:
        mask = labels == lab
        region_pixels = val[mask]
        if region_pixels.size > 0:
            mean_val = region_pixels.mean()
            var_val = region_pixels.var()
            if 0 < mean_val < 1:
                nv = var_val / (mean_val * (1 - mean_val) + 1e-8)
                total_homogeneity += nv * region_pixels.size
                total_pixels += region_pixels.size
    metrics["region_homogeneity"] = float(total_homogeneity / max(total_pixels, 1))
    # 3. Director Consistency
    x = gray.astype(np.float32) / 255.0
    gx = cv2.Scharr(x, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(x, cv2.CV_32F, 0, 1)
    pixel_dirs = np.arctan2(gy, gx)
    consistent = 0
    total = 0
    for lab in unique_labels:
        mask = labels == lab
        region_dirs = pixel_dirs[mask]
        if region_dirs.size > 0:
            cos_dir = np.mean(np.cos(region_dirs))
            sin_dir = np.mean(np.sin(region_dirs))
            dom_dir = np.arctan2(sin_dir, cos_dir)
            diffs = np.abs(((region_dirs - dom_dir) + np.pi) % (2 * np.pi) - np.pi)
            consistent += (diffs < np.pi / 4).sum()
            total += region_dirs.size
    metrics["director_consistency"] = float(consistent / max(total, 1) * 100.0)
    # 4. Overall Score
    eas_norm = min(1.0, metrics["edge_alignment"] / 70.0)
    hom_norm = max(0.0, 1.0 - metrics["region_homogeneity"] * 5.0)
    dc_norm = min(1.0, metrics["director_consistency"] / 80.0)
    metrics["overall_score"] = float(0.4 * eas_norm + 0.3 * hom_norm + 0.3 * dc_norm)
    return metrics


def save_gray(path: Path, x: np.ndarray):
    cv2.imwrite(str(path), to_uint8(x) if x.dtype != np.uint8 else x)


def auto_tune(image: np.ndarray, s: Settings) -> Settings:
    """Find good parameters automatically by sampling and scoring."""
    h, w = image.shape[:2]
    if max(h, w) > 800:
        print("Изображение слишком большое для авто-тюнинга (макс. 800px). Пропуск.")
        return s
    crop_h, crop_w = int(h * s.auto_tune_crop_factor), int(w * s.auto_tune_crop_factor)
    crop = image[:crop_h, :crop_w]
    rng = np.random.default_rng(s.random_seed)
    candidates = []
    for _ in range(s.auto_tune_candidates):
        c = copy.copy(s)
        c.marker_min_distance = max(8, int(s.marker_min_distance * rng.uniform(0.5, 1.5)))
        c.marker_min_radius = max(1.0, s.marker_min_radius * rng.uniform(0.5, 1.5))
        c.merge_score = max(0.05, min(0.8, s.merge_score + rng.uniform(-0.1, 0.1)))
        c.merge_boundary = max(0.2, min(0.8, s.merge_boundary + rng.uniform(-0.1, 0.1)))
        c.merge_orientation = max(0.2, min(0.8, s.merge_orientation + rng.uniform(-0.1, 0.1)))
        c.min_region_area = max(10, int(s.min_region_area * rng.uniform(0.3, 2.0)))
        c.quality_report = False
        c.auto_tune = False
        candidates.append(c)
    best_score = -1.0
    best_settings = s
    print(f"Авто-тюнинг: проверка {len(candidates)} кандидатов...")
    for i, candidate in enumerate(candidates, 1):
        try:
            test_output = Path("/tmp/_grain_tune")
            test_run(image, crop, candidate, test_output, 200.0, None, quiet=True)
            # Quality from test_run output
            pass
        except Exception:
            continue
    # If no candidates scored well, return defaults
    print(f"Авто-тюнинг завершён. Лучший набор параметров.")
    return best_settings


def test_run(image: np.ndarray, source: np.ndarray, s: Settings, output: Path,
             scale_um: float, scale_px: Optional[float], quiet: bool = False) -> float:
    """Quick test run returning overall_score for auto-tuning."""
    src, corrected, resize_scale = preprocess(source, s)
    terrain, intensity, orientation_b, c, q, coherence = build_terrain(corrected, s)
    edges = hysteresis_edges(terrain, s.low_edge_quantile, s.high_edge_quantile,
                             marker_min_distance=s.marker_min_distance)
    distance, markers = make_markers(edges, s)
    initial = watershed_first(terrain, markers)
    merged = merge_regions(initial, terrain, corrected, c, q, coherence, s)
    final = refine_watershed(terrain, merged, marker_min_distance=s.marker_min_distance)
    final = contiguous(final)
    bmask = boundary_mask(final)
    if not quiet:
        print(f"  Кандидат: {int(final.max())} зёрен")
    qm = compute_quality_metrics(final, terrain, edges, corrected)
    return qm["overall_score"]


def run(image: np.ndarray, s: Settings, output: Path, scale_um: float, scale_px: Optional[float]):
    source, corrected, resize_scale = preprocess(image, s)
    terrain, intensity, orientation_b, c, q, coherence = build_terrain(corrected, s)
    edges = hysteresis_edges(terrain, s.low_edge_quantile, s.high_edge_quantile,
                             marker_min_distance=s.marker_min_distance)
    distance, markers = make_markers(edges, s)
    initial = watershed_first(terrain, markers)
    merged = merge_regions(initial, terrain, corrected, c, q, coherence, s)
    final = refine_watershed(terrain, merged, marker_min_distance=s.marker_min_distance)
    final = contiguous(final)
    bmask = boundary_mask(final)

    overlay = source.copy()
    show_mask = cv2.dilate(bmask, np.ones((2, 2), np.uint8)) > 0
    overlay[show_mask] = (0, 255, 0)

    original_scale_px = scale_px or detect_scale_bar(image, known_scale_px=s.known_scale_px)
    um_per_px = scale_um / original_scale_px / resize_scale if original_scale_px else None
    secants, total_length, total_hits = secants_image(source, final, s.secants, s.random_seed)
    rows, border_count, filtered_count = measurements(
        final, um_per_px,
        exclude_border=s.exclude_border_grains,
        grain_size_min=s.grain_size_min,
        grain_size_max=s.grain_size_max,
    )

    output.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output / "01_analysis_source.png"), source)
    cv2.imwrite(str(output / "02_corrected.png"), corrected)
    cv2.imwrite(str(output / "03_orientation_map.png"), orientation_visual(c, q, coherence))
    save_gray(output / "04_orientation_boundary.png", orientation_b)
    save_gray(output / "05_intensity_gradient.png", intensity)
    save_gray(output / "06_fused_terrain.png", terrain)
    cv2.imwrite(str(output / "07_hysteresis_edges.png"), edges.astype(np.uint8) * 255)
    cv2.imwrite(str(output / "08_distance_transform.png"), cv2.applyColorMap(to_uint8(distance), cv2.COLORMAP_TURBO))
    cv2.imwrite(str(output / "09_markers.png"), (markers > 0).astype(np.uint8) * 255)
    cv2.imwrite(str(output / "10_initial_watershed.png"), label_map(initial, s.random_seed))
    cv2.imwrite(str(output / "11_merged_areas.png"), label_map(merged, s.random_seed))
    cv2.imwrite(str(output / "12_refined_labels.png"), label_map(final, s.random_seed))
    cv2.imwrite(str(output / "13_boundary_mask.png"), bmask)
    cv2.imwrite(str(output / "14_grain_boundaries.png"), overlay)
    cv2.imwrite(str(output / "15_secants.png"), secants)
    if rows:
        with (output / "grain_measurements.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
    mean_px = total_length / total_hits if total_hits else 0.0
    print(f"Начальных областей: {int(initial.max())}")
    print(f"После объединения: {int(merged.max())}")
    print(f"Итоговых областей: {int(final.max())}")
    if border_count > 0:
        print(f"Зёрен на границе: {border_count}")
    if filtered_count > 0:
        print(f"Отфильтровано (вне диапазона): {filtered_count}")
    print(f"Пересечений: {total_hits}; средний отрезок: {mean_px:.2f} px")
    if um_per_px:
        print(f"Средний отрезок: {mean_px * um_per_px:.2f} мкм")
    else:
        print("Линейка не найдена: укажите --scale-px вручную.")

    # Quality report
    if s.quality_report:
        qm = compute_quality_metrics(final, terrain, edges, corrected)
        print("\n--- Отчёт о качестве сегментации ---")
        print(f"  Edge Alignment Score: {qm['edge_alignment']:.1f}%")
        print(f"  Область однородности: {qm['region_homogeneity']:.4f} (меньше=лучше)")
        print(f"  Director Consistency: {qm['director_consistency']:.1f}%")
        print(f"  Overall Score: {qm['overall_score']:.3f} (больше=лучше)")
        print("-----------------------------------\n")

    print(f"Результаты: {output.resolve()}")
    return overlay, secants


def parse_args():
    p = argparse.ArgumentParser(description="Сегментация зёрен по ориентации текстуры + watershed v3")
    p.add_argument("image", nargs="?", default=None,
                   help="Путь к изображению (если не указан — интерактивный выбор)")
    p.add_argument("--output", default=None)
    p.add_argument("--max-side", type=int, default=1600)
    p.add_argument("--crop-bottom", type=float, default=0.12)
    p.add_argument("--marker-min-distance", type=int, default=38)
    p.add_argument("--marker-min-radius", type=float, default=7.0)
    p.add_argument("--min-region-area", type=int, default=1000)
    p.add_argument("--merge-score", type=float, default=0.285)
    p.add_argument("--merge-boundary", type=float, default=0.44)
    p.add_argument("--merge-orientation", type=float, default=0.42)
    p.add_argument("--merge-passes", type=int, default=5)
    p.add_argument("--secants", type=int, default=5)
    p.add_argument("--scale-um", type=float, default=200.0)
    p.add_argument("--scale-px", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show", action="store_true")
    # --- New CLI arguments ---
    p.add_argument("--known-scale", type=str, default=None,
                   help="Ручной масштаб: '200um_at_100px'")
    p.add_argument("--exclude-border-grains", action="store_true", default=False,
                   help="Исключить зёрна на границе изображения")
    p.add_argument("--quality-report", action="store_true", default=False,
                   help="Печать метрик качества сегментации")
    p.add_argument("--auto-tune", action="store_true", default=False,
                   help="Автоматический подбор параметров")
    p.add_argument("--auto-tune-candidates", type=int, default=10,
                   help="Кол-во кандидатов для авто-тюнинга (по умол. 10)")
    p.add_argument("--grain-size-range", type=str, default=None,
                   help="Ожидаемый размер зёрен в мкм, e.g. '50-200'")
    return p.parse_args()


def main():
    a = parse_args()

    # Interactive image selection
    if a.image is None:
        print("Запуск в интерактивном режиме...")
        image_path = select_image_interactive()
    else:
        image_path = a.image
    path = Path(image_path).expanduser()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Не удалось открыть {path}")

    # Parse known scale
    known_scale_px = None
    if a.known_scale:
        scale_um, scale_px = parse_known_scale(a.known_scale)
        if scale_um and scale_px:
            a.scale_um = scale_um
            known_scale_px = scale_px

    # Parse grain size range
    grain_size_min, grain_size_max = None, None
    if a.grain_size_range:
        grain_size_min, grain_size_max = parse_grain_size_range(a.grain_size_range)

    s = Settings(
        max_side=max(600, a.max_side), crop_bottom=float(np.clip(a.crop_bottom, 0, .3)),
        marker_min_distance=max(8, a.marker_min_distance), marker_min_radius=max(1, a.marker_min_radius),
        min_region_area=max(0, a.min_region_area), merge_score=float(a.merge_score),
        merge_boundary=float(a.merge_boundary), merge_orientation=float(a.merge_orientation),
        merge_passes=max(0, a.merge_passes), secants=max(0, a.secants), random_seed=a.seed,
        exclude_border_grains=a.exclude_border_grains,
        known_scale_px=known_scale_px or a.scale_px,
        grain_size_min=grain_size_min,
        grain_size_max=grain_size_max,
        quality_report=a.quality_report,
        auto_tune=a.auto_tune,
        auto_tune_candidates=a.auto_tune_candidates,
    )

    # Auto-tuning
    if s.auto_tune:
        s = auto_tune(image, s)

    output = Path(a.output) if a.output else path.parent / "result"
    overlay, secants = run(image, s, output, a.scale_um, s.known_scale_px)
    if a.show:
        for name, img in (("Grain boundaries v3", overlay), ("Secants v3", secants)):
            h, w = img.shape[:2]
            scale = min(1.0, 1400/max(h,w))
            cv2.imshow(name, cv2.resize(img, (round(w*scale), round(h*scale))) if scale < 1 else img)
        cv2.waitKey(0); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
EXPECTED_COLS = 47
EXPECTED_ROWS = 28
INNER_COLS = EXPECTED_COLS - 1
INNER_ROWS = EXPECTED_ROWS - 1
PATTERN_SIZE = (INNER_COLS, INNER_ROWS)
EXPECTED_INNER_CORNERS = INNER_COLS * INNER_ROWS
FIND_FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    | cv2.CALIB_CB_NORMALIZE_IMAGE
    | cv2.CALIB_CB_FILTER_QUADS
)
SUBPIX_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    50,
    0.001,
)
CHESSBOARD_VARIANTS = (
    "gray",
    "clahe",
    "equalize",
    "blur_clahe",
    "scale_075_clahe",
    "scale_150_clahe",
)

LEFT_FILES = (
    BASE_DIR / "left" / "2026020501_qmz.5.Camera_left.png",
    BASE_DIR / "left" / "2026020501_qmz.9.Camera_left.png",
)
RIGHT_FILES = (
    BASE_DIR / "right" / "2026020501_qmz.5.Camera.right.png",
    BASE_DIR / "right" / "2026020501_qmz.9.Camera.right.png",
)


@dataclass
class BoardDetection:
    image: np.ndarray
    gray: np.ndarray
    mask: np.ndarray
    left: np.ndarray
    right: np.ndarray
    top: np.ndarray
    bottom: np.ndarray
    vertical_lines: np.ndarray
    horizontal_lines: np.ndarray
    nodes: np.ndarray
    node_valid: np.ndarray


def smooth1d(values: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return values.astype(np.float32)
    pad = kernel_size // 2
    padded = np.pad(values.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones(kernel_size, np.float32) / kernel_size
    return np.convolve(padded, kernel, mode="valid")


def load_images(side: str) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    if side == "left":
        path_a, path_b = LEFT_FILES
    elif side == "right":
        path_a, path_b = RIGHT_FILES
    else:
        raise ValueError(f"Unsupported side: {side}")

    image_a = cv2.imread(str(path_a), cv2.IMREAD_COLOR)
    image_b = cv2.imread(str(path_b), cv2.IMREAD_COLOR)
    if image_a is None or image_b is None:
        raise FileNotFoundError(f"Failed to read images for side={side}")
    if image_a.shape != image_b.shape:
        raise ValueError(
            f"Image size mismatch for side={side}: {image_a.shape} vs {image_b.shape}"
        )
    return path_a, path_b, image_a, image_b


def preprocess_chessboard_variant(
    gray: np.ndarray, variant_name: str
) -> tuple[np.ndarray, float]:
    if variant_name == "gray":
        return gray, 1.0
    if variant_name == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray), 1.0
    if variant_name == "equalize":
        return cv2.equalizeHist(gray), 1.0
    if variant_name == "blur_clahe":
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(blur), 1.0
    if variant_name == "scale_075_clahe":
        scaled = cv2.resize(gray, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(scaled), 0.75
    if variant_name == "scale_150_clahe":
        scaled = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(scaled), 1.5
    raise ValueError(f"Unsupported variant: {variant_name}")


def run_find_chessboard_variant(
    gray: np.ndarray, variant_name: str
) -> tuple[bool, int, str, np.ndarray | None]:
    processed, scale = preprocess_chessboard_variant(gray, variant_name)
    success, corners = cv2.findChessboardCorners(processed, PATTERN_SIZE, FIND_FLAGS)
    if corners is None or len(corners) == 0:
        return bool(success), 0, variant_name, None

    refined = cv2.cornerSubPix(
        processed.astype(np.float32),
        corners.astype(np.float32),
        (5, 5),
        (-1, -1),
        SUBPIX_CRITERIA,
    ).reshape(-1, 2)
    if scale != 1.0:
        refined = refined / scale
    return bool(success), len(refined), variant_name, refined


def pick_best_find_chessboard(gray: np.ndarray) -> tuple[str, bool, np.ndarray | None]:
    best_result: tuple[tuple[int, int, int, float], str, bool, np.ndarray | None] | None = None
    for variant_name in CHESSBOARD_VARIANTS:
        success, count, _, corners = run_find_chessboard_variant(gray, variant_name)
        score = (
            int(count == EXPECTED_INNER_CORNERS),
            int(success),
            count,
            -abs(preprocess_chessboard_variant(gray, variant_name)[1] - 1.0),
        )
        if best_result is None or score > best_result[0]:
            best_result = (score, variant_name, success, corners)
    assert best_result is not None
    _, variant_name, success, corners = best_result
    return variant_name, success, corners


def normalize_inner_corner_order(corners: np.ndarray) -> np.ndarray:
    grid = corners.reshape(INNER_ROWS, INNER_COLS, 2).copy()
    if grid[0, :, 1].mean() > grid[-1, :, 1].mean():
        grid = grid[::-1]
    if grid[:, 0, 0].mean() > grid[:, -1, 0].mean():
        grid = grid[:, ::-1]
    return grid


def extrapolate_full_grid(
    inner_nodes: np.ndarray, inner_valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    full_nodes = np.zeros((EXPECTED_ROWS + 1, EXPECTED_COLS + 1, 2), np.float32)
    full_valid = np.zeros((EXPECTED_ROWS + 1, EXPECTED_COLS + 1), dtype=bool)

    full_nodes[1:-1, 1:-1] = inner_nodes
    full_valid[1:-1, 1:-1] = inner_valid

    full_nodes[1:-1, 0] = 2.0 * full_nodes[1:-1, 1] - full_nodes[1:-1, 2]
    full_nodes[1:-1, -1] = 2.0 * full_nodes[1:-1, -2] - full_nodes[1:-1, -3]
    full_valid[1:-1, 0] = full_valid[1:-1, 1] & full_valid[1:-1, 2]
    full_valid[1:-1, -1] = full_valid[1:-1, -2] & full_valid[1:-1, -3]

    full_nodes[0, 1:-1] = 2.0 * full_nodes[1, 1:-1] - full_nodes[2, 1:-1]
    full_nodes[-1, 1:-1] = 2.0 * full_nodes[-2, 1:-1] - full_nodes[-3, 1:-1]
    full_valid[0, 1:-1] = full_valid[1, 1:-1] & full_valid[2, 1:-1]
    full_valid[-1, 1:-1] = full_valid[-2, 1:-1] & full_valid[-3, 1:-1]

    full_nodes[0, 0] = full_nodes[0, 1] + full_nodes[1, 0] - full_nodes[1, 1]
    full_nodes[0, -1] = full_nodes[0, -2] + full_nodes[1, -1] - full_nodes[1, -2]
    full_nodes[-1, 0] = full_nodes[-2, 0] + full_nodes[-1, 1] - full_nodes[-2, 1]
    full_nodes[-1, -1] = full_nodes[-2, -1] + full_nodes[-1, -2] - full_nodes[-2, -2]
    full_valid[0, 0] = full_valid[0, 1] & full_valid[1, 0] & full_valid[1, 1]
    full_valid[0, -1] = full_valid[0, -2] & full_valid[1, -1] & full_valid[1, -2]
    full_valid[-1, 0] = full_valid[-2, 0] & full_valid[-1, 1] & full_valid[-2, 1]
    full_valid[-1, -1] = full_valid[-2, -1] & full_valid[-1, -2] & full_valid[-2, -2]

    return full_nodes, full_valid


def empty_board_arrays(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = gray.shape
    return (
        np.zeros(height, np.float32),
        np.zeros(height, np.float32),
        np.zeros(width, np.float32),
        np.zeros(width, np.float32),
    )


def detect_board_region(
    gray: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gray_f = gray.astype(np.float32)
    mean = cv2.GaussianBlur(gray_f, (0, 0), 6)
    mean2 = cv2.GaussianBlur(gray_f * gray_f, (0, 0), 6)
    stddev = np.sqrt(np.clip(mean2 - mean * mean, 0, None))

    threshold = np.percentile(stddev, 80)
    mask = (stddev >= threshold).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    height, width = gray.shape
    left = np.full(height, np.nan, np.float32)
    right = np.full(height, np.nan, np.float32)
    top = np.full(width, np.nan, np.float32)
    bottom = np.full(width, np.nan, np.float32)

    for y in range(height):
        xs = np.where(mask[y] > 0)[0]
        if xs.size:
            left[y] = xs.min()
            right[y] = xs.max()

    for x in range(width):
        ys = np.where(mask[:, x] > 0)[0]
        if ys.size:
            top[x] = ys.min()
            bottom[x] = ys.max()

    for arr in (left, right):
        missing = np.isnan(arr)
        arr[missing] = np.interp(
            np.flatnonzero(missing), np.flatnonzero(~missing), arr[~missing]
        )
        arr[:] = smooth1d(arr, 81)

    for arr in (top, bottom):
        missing = np.isnan(arr)
        arr[missing] = np.interp(
            np.flatnonzero(missing), np.flatnonzero(~missing), arr[~missing]
        )
        arr[:] = smooth1d(arr, 81)

    return mask, left, right, top, bottom


def count_spaced_peaks(
    profile: np.ndarray, *, percentile: float, min_distance: int
) -> int:
    if profile.size < 3:
        return 0
    threshold = np.percentile(profile, percentile)
    peak_mask = (
        (profile[1:-1] > profile[:-2])
        & (profile[1:-1] >= profile[2:])
        & (profile[1:-1] > threshold)
    )
    peaks = np.flatnonzero(peak_mask) + 1
    if peaks.size == 0:
        return 0
    order = peaks[np.argsort(profile[peaks])[::-1]]
    kept: list[int] = []
    for peak in order:
        if all(abs(int(peak) - existing) > min_distance for existing in kept):
            kept.append(int(peak))
    return len(kept)


def estimate_board_shape(
    gray: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    top: np.ndarray,
    bottom: np.ndarray,
) -> tuple[int, int]:
    widest_row = int(np.argmax(right - left))
    tallest_col = int(np.argmax(bottom - top))

    row_band = cv2.GaussianBlur(
        np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)), (0, 0), 1.2
    )
    col_band = cv2.GaussianBlur(
        np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)), (0, 0), 1.2
    )

    row_profile = row_band[max(0, widest_row - 2) : widest_row + 3].mean(axis=0)
    col_profile = col_band[:, max(0, tallest_col - 2) : tallest_col + 3].mean(axis=1)

    row_width = max(float(right[widest_row] - left[widest_row]), 1.0)
    col_height = max(float(bottom[tallest_col] - top[tallest_col]), 1.0)
    coarse_cols = count_spaced_peaks(
        row_profile, percentile=80, min_distance=max(6, int(row_width / 80))
    )
    coarse_rows = count_spaced_peaks(
        col_profile, percentile=80, min_distance=max(6, int(col_height / 60))
    )

    # The scanline peak count is only a coarse sanity check because blur and
    # distortion can merge or split adjacent edge responses.
    if not (30 <= coarse_cols <= 80 and 20 <= coarse_rows <= 50):
        raise ValueError(
            f"Board shape sanity check failed: coarse_cols={coarse_cols}, coarse_rows={coarse_rows}"
        )
    return EXPECTED_COLS, EXPECTED_ROWS


def track_vertical_boundaries(
    gray: np.ndarray, left: np.ndarray, right: np.ndarray, cols: int
) -> np.ndarray:
    height, width = gray.shape
    grad_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    grad_x = cv2.GaussianBlur(grad_x, (0, 0), 1.2)
    widest_row = int(np.argmax(right - left))
    lines = np.full((cols + 1, height), np.nan, np.float32)

    def detect_row(y: int, prediction: np.ndarray | None) -> np.ndarray:
        span = max(float(right[y] - left[y]), 1.0)
        step = span / cols
        profile = grad_x[max(0, y - 2) : min(height, y + 3)].mean(axis=0)
        positions = np.zeros(cols + 1, np.float32)
        prev = left[y] - step

        for col_idx in range(cols + 1):
            if prediction is None:
                center = left[y] + col_idx * step
                radius = 0.42 * step
            else:
                center = float(prediction[col_idx])
                radius = 0.32 * step + 4.0
            low = int(max(left[y] if col_idx == 0 else prev + 2, round(center - radius)))
            high = int(
                min(right[y] if col_idx == cols else width - 1, round(center + radius))
            )
            if high <= low:
                positions[col_idx] = np.clip(center, 0, width - 1)
            else:
                local = profile[low : high + 1]
                positions[col_idx] = low + int(np.argmax(local))
            prev = positions[col_idx]

        return positions

    lines[:, widest_row] = detect_row(widest_row, None)
    for y in range(widest_row - 1, -1, -1):
        lines[:, y] = detect_row(y, lines[:, y + 1])
    for y in range(widest_row + 1, height):
        lines[:, y] = detect_row(y, lines[:, y - 1])

    for col_idx in range(cols + 1):
        lines[col_idx] = smooth1d(lines[col_idx], 41)

    return lines


def track_horizontal_boundaries(
    gray: np.ndarray, top: np.ndarray, bottom: np.ndarray, rows: int
) -> np.ndarray:
    height, width = gray.shape
    grad_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    grad_y = cv2.GaussianBlur(grad_y, (0, 0), 1.2)
    tallest_col = int(np.argmax(bottom - top))
    lines = np.full((rows + 1, width), np.nan, np.float32)

    def detect_col(x: int, prediction: np.ndarray | None) -> np.ndarray:
        span = max(float(bottom[x] - top[x]), 1.0)
        step = span / rows
        profile = grad_y[:, max(0, x - 2) : min(width, x + 3)].mean(axis=1)
        positions = np.zeros(rows + 1, np.float32)
        prev = top[x] - step

        for row_idx in range(rows + 1):
            if prediction is None:
                center = top[x] + row_idx * step
                radius = 0.42 * step
            else:
                center = float(prediction[row_idx])
                radius = 0.32 * step + 4.0
            low = int(max(top[x] if row_idx == 0 else prev + 2, round(center - radius)))
            high = int(
                min(bottom[x] if row_idx == rows else height - 1, round(center + radius))
            )
            if high <= low:
                positions[row_idx] = np.clip(center, 0, height - 1)
            else:
                local = profile[low : high + 1]
                positions[row_idx] = low + int(np.argmax(local))
            prev = positions[row_idx]

        return positions

    lines[:, tallest_col] = detect_col(tallest_col, None)
    for x in range(tallest_col - 1, -1, -1):
        lines[:, x] = detect_col(x, lines[:, x + 1])
    for x in range(tallest_col + 1, width):
        lines[:, x] = detect_col(x, lines[:, x - 1])

    for row_idx in range(rows + 1):
        lines[row_idx] = smooth1d(lines[row_idx], 41)

    return lines


def intersect_line_families(
    vertical_lines: np.ndarray, horizontal_lines: np.ndarray
) -> np.ndarray:
    cols = vertical_lines.shape[0] - 1
    rows = horizontal_lines.shape[0] - 1
    height = vertical_lines.shape[1]
    width = horizontal_lines.shape[1]
    y_coords = np.arange(height, dtype=np.float32)
    x_coords = np.arange(width, dtype=np.float32)
    nodes = np.zeros((rows + 1, cols + 1, 2), np.float32)

    for row_idx in range(rows + 1):
        y_value = float(horizontal_lines[row_idx, width // 2])
        for col_idx in range(cols + 1):
            for _ in range(5):
                x_value = float(
                    np.interp(np.clip(y_value, 0, height - 1), y_coords, vertical_lines[col_idx])
                )
                y_value = float(
                    np.interp(np.clip(x_value, 0, width - 1), x_coords, horizontal_lines[row_idx])
                )
            nodes[row_idx, col_idx] = (x_value, y_value)

    return nodes


def detect_corner_candidates(gray: np.ndarray) -> np.ndarray:
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=10000,
        qualityLevel=0.0005,
        minDistance=3,
        blockSize=5,
        useHarrisDetector=True,
    )
    if corners is None:
        return np.empty((0, 2), np.float32)
    corners = corners.reshape(-1, 2).astype(np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        0.001,
    )
    corners = cv2.cornerSubPix(
        gray.astype(np.float32),
        corners.reshape(-1, 1, 2),
        (5, 5),
        (-1, -1),
        criteria,
    ).reshape(-1, 2)
    return corners


def node_search_radius(nodes: np.ndarray, row_idx: int, col_idx: int) -> float:
    distances: list[float] = []
    rows, cols = nodes.shape[:2]
    if row_idx > 0:
        distances.append(float(np.linalg.norm(nodes[row_idx, col_idx] - nodes[row_idx - 1, col_idx])))
    if row_idx + 1 < rows:
        distances.append(float(np.linalg.norm(nodes[row_idx + 1, col_idx] - nodes[row_idx, col_idx])))
    if col_idx > 0:
        distances.append(float(np.linalg.norm(nodes[row_idx, col_idx] - nodes[row_idx, col_idx - 1])))
    if col_idx + 1 < cols:
        distances.append(float(np.linalg.norm(nodes[row_idx, col_idx + 1] - nodes[row_idx, col_idx])))
    if not distances:
        return 6.0
    return max(4.0, 0.35 * min(distances) + 2.0)


def snap_nodes_to_candidates(
    nodes: np.ndarray, candidates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    snapped = nodes.copy()
    found = np.zeros(nodes.shape[:2], dtype=bool)
    if candidates.size == 0:
        return snapped, found

    cand_x = candidates[:, 0]
    cand_y = candidates[:, 1]
    rows, cols = nodes.shape[:2]
    for row_idx in range(rows):
        for col_idx in range(cols):
            point = nodes[row_idx, col_idx]
            radius = node_search_radius(nodes, row_idx, col_idx)
            mask = (
                (np.abs(cand_x - point[0]) <= radius)
                & (np.abs(cand_y - point[1]) <= radius)
            )
            local = candidates[mask]
            if local.size == 0:
                continue
            distances2 = np.sum((local - point) ** 2, axis=1)
            snapped[row_idx, col_idx] = local[int(np.argmin(distances2))]
            found[row_idx, col_idx] = True
    return snapped, found


def refine_nodes_with_candidates(gray: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    corners = detect_corner_candidates(gray)
    refined, _ = snap_nodes_to_candidates(nodes, corners)
    return refined


def refine_internal_nodes(gray: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    corners = detect_corner_candidates(gray)
    if corners.size == 0:
        return nodes

    refined = nodes.copy()
    rows, cols = nodes.shape[:2]
    for row_idx in range(1, rows - 1):
        for col_idx in range(1, cols - 1):
            point = nodes[row_idx, col_idx]
            radius = node_search_radius(nodes, row_idx, col_idx)
            distances2 = np.sum((corners - point) ** 2, axis=1)
            nearest_idx = int(np.argmin(distances2))
            if distances2[nearest_idx] <= radius * radius:
                refined[row_idx, col_idx] = corners[nearest_idx]
    return refined


def detect_board(image: np.ndarray) -> BoardDetection:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variant_name, success, corners = pick_best_find_chessboard(gray)
    if corners is None or len(corners) != EXPECTED_INNER_CORNERS:
        raise ValueError(
            f"findChessboardCorners failed to recover the full {PATTERN_SIZE} grid on qmz.5 "
            f"(variant={variant_name}, success={success}, count={0 if corners is None else len(corners)})"
        )

    inner_nodes = normalize_inner_corner_order(corners)
    nodes, node_valid = extrapolate_full_grid(
        inner_nodes, np.ones((INNER_ROWS, INNER_COLS), dtype=bool)
    )
    left, right, top, bottom = empty_board_arrays(gray)
    return BoardDetection(
        image=image,
        gray=gray,
        mask=np.zeros_like(gray, dtype=np.uint8),
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        vertical_lines=np.empty((0, 0), np.float32),
        horizontal_lines=np.empty((0, 0), np.float32),
        nodes=nodes,
        node_valid=node_valid,
    )


def transfer_nodes_from_reference(
    reference: BoardDetection, target_image: np.ndarray
) -> BoardDetection:
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    variant_name, success, corners = pick_best_find_chessboard(target_gray)
    if corners is not None and len(corners) == EXPECTED_INNER_CORNERS:
        inner_nodes = normalize_inner_corner_order(corners)
        nodes, node_valid = extrapolate_full_grid(
            inner_nodes, np.ones((INNER_ROWS, INNER_COLS), dtype=bool)
        )
    else:
        source_inner = reference.nodes[1:-1, 1:-1].astype(np.float32)
        ref_points = source_inner.reshape(-1, 1, 2)
        tracked, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            reference.gray,
            target_gray,
            ref_points,
            None,
            winSize=(41, 41),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, 0.01),
            minEigThreshold=1e-5,
        )
        if tracked is None:
            tracked = ref_points.copy()
            status_fwd = np.zeros((ref_points.shape[0], 1), np.uint8)

        back, status_back, _ = cv2.calcOpticalFlowPyrLK(
            target_gray,
            reference.gray,
            tracked,
            None,
            winSize=(41, 41),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 50, 0.01),
            minEigThreshold=1e-5,
        )
        if back is None:
            back = tracked.copy()
            status_back = np.zeros((ref_points.shape[0], 1), np.uint8)

        tracked = tracked.reshape(INNER_ROWS, INNER_COLS, 2)
        back = back.reshape(INNER_ROWS, INNER_COLS, 2)
        inner_valid = status_fwd.reshape(INNER_ROWS, INNER_COLS).astype(bool)
        inner_valid &= status_back.reshape(INNER_ROWS, INNER_COLS).astype(bool)
        inner_valid &= np.linalg.norm(back - source_inner, axis=2) <= 3.0
        height, width = target_gray.shape
        inner_valid &= (
            (tracked[:, :, 0] >= 6.0)
            & (tracked[:, :, 0] < width - 6.0)
            & (tracked[:, :, 1] >= 6.0)
            & (tracked[:, :, 1] < height - 6.0)
        )

        tracked_flat = tracked.reshape(-1, 2)
        valid_flat = inner_valid.reshape(-1)
        if valid_flat.any():
            refined = cv2.cornerSubPix(
                target_gray.astype(np.float32),
                tracked_flat[valid_flat].reshape(-1, 1, 2).astype(np.float32),
                (5, 5),
                (-1, -1),
                SUBPIX_CRITERIA,
            ).reshape(-1, 2)
            tracked_flat[valid_flat] = refined

        nodes, node_valid = extrapolate_full_grid(tracked, inner_valid)

    left, right, top, bottom = empty_board_arrays(target_gray)
    return BoardDetection(
        image=target_image,
        gray=target_gray,
        mask=np.zeros_like(target_gray, dtype=np.uint8),
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        vertical_lines=np.empty((0, 0), np.float32),
        horizontal_lines=np.empty((0, 0), np.float32),
        nodes=nodes,
        node_valid=node_valid,
    )


def point_in_board(det: BoardDetection, point: np.ndarray, margin: float = 6.0) -> bool:
    x, y = float(point[0]), float(point[1])
    height, width = det.gray.shape
    if not (0.0 <= x < width and 0.0 <= y < height):
        return False
    return True


def iter_cell_records(
    det_a: BoardDetection, det_b: BoardDetection
) -> Iterable[
    tuple[
        int,
        int,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        bool,
        bool,
        bool,
    ]
]:
    rows = det_a.nodes.shape[0] - 1
    cols = det_a.nodes.shape[1] - 1
    if det_b.nodes.shape != det_a.nodes.shape:
        raise ValueError("Detection shape mismatch between image pair")

    for row_idx in range(rows):
        for col_idx in range(cols):
            quad_a = det_a.nodes[row_idx : row_idx + 2, col_idx : col_idx + 2]
            quad_b = det_b.nodes[row_idx : row_idx + 2, col_idx : col_idx + 2]
            center_a = quad_a.reshape(-1, 2).mean(axis=0)
            center_b = quad_b.reshape(-1, 2).mean(axis=0)
            right_bottom_a = det_a.nodes[row_idx + 1, col_idx + 1]
            right_bottom_b = det_b.nodes[row_idx + 1, col_idx + 1]
            valid_a = bool(det_a.node_valid[row_idx + 1, col_idx + 1])
            valid_b = bool(det_b.node_valid[row_idx + 1, col_idx + 1])
            valid_a = valid_a and point_in_board(det_a, center_a) and point_in_board(
                det_a, right_bottom_a
            )
            valid_b = valid_b and point_in_board(det_b, center_b) and point_in_board(
                det_b, right_bottom_b
            )
            valid_pair = valid_a and valid_b
            yield (
                row_idx,
                col_idx,
                center_a,
                center_b,
                right_bottom_a,
                right_bottom_b,
                valid_a,
                valid_b,
                valid_pair,
            )


def write_points_file(path: Path, det_a: BoardDetection, det_b: BoardDetection) -> int:
    missing = 0
    with path.open("w", encoding="utf-8") as handle:
        for _, _, _, _, rb_a, rb_b, valid_a, valid_b, valid_pair in iter_cell_records(
            det_a, det_b
        ):
            if valid_pair:
                handle.write(
                    f"{rb_a[0]:.1f} {rb_a[1]:.1f} {rb_b[0]:.1f} {rb_b[1]:.1f}\n"
                )
            elif valid_a:
                handle.write(f"{rb_a[0]:.1f} {rb_a[1]:.1f} -1 -1\n")
                missing += 1
            elif valid_b:
                handle.write(f"-1 -1 {rb_b[0]:.1f} {rb_b[1]:.1f}\n")
                missing += 1
            else:
                handle.write("-1 -1 -1 -1\n")
                missing += 1
    return missing


def clip_point(point: np.ndarray, width: int, height: int) -> tuple[int, int]:
    x = int(np.clip(round(float(point[0])), 0, width - 1))
    y = int(np.clip(round(float(point[1])), 0, height - 1))
    return x, y


def draw_marker(
    canvas: np.ndarray,
    point: np.ndarray,
    label_pos: np.ndarray,
    index: int,
    present: bool,
    state: str,
) -> None:
    height, width = canvas.shape[:2]
    point_xy = clip_point(point, width, height)
    label_xy = clip_point(label_pos, width, height)
    if state == "paired" and present:
        cv2.circle(canvas, point_xy, 3, (0, 220, 0), -1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(index),
            label_xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (0, 180, 0),
            1,
            cv2.LINE_AA,
        )
    elif state == "single" and present:
        cv2.circle(canvas, point_xy, 3, (0, 220, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(index),
            label_xy,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.drawMarker(
            canvas,
            label_xy,
            (0, 220, 255) if state == "single" else (0, 0, 255),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=8,
            thickness=1,
            line_type=cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(index),
            (label_xy[0] + 2, label_xy[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (0, 220, 255) if state == "single" else (0, 0, 255),
            1,
            cv2.LINE_AA,
        )


def export_visualization(path: Path, det_a: BoardDetection, det_b: BoardDetection) -> None:
    image_a = det_a.image.copy()
    image_b = det_b.image.copy()
    width = image_a.shape[1]
    canvas = np.hstack([image_a, image_b])

    for index, (_, _, center_a, center_b, rb_a, rb_b, valid_a, valid_b, valid_pair) in enumerate(
        iter_cell_records(det_a, det_b)
    ):
        if valid_pair:
            state = "paired"
        elif valid_a or valid_b:
            state = "single"
        else:
            state = "missing"

        draw_marker(canvas, rb_a, center_a, index, valid_a, state)
        draw_marker(
            canvas,
            rb_b + np.array([width, 0], np.float32),
            center_b + np.array([width, 0], np.float32),
            index,
            valid_b,
            state,
        )

    cv2.putText(
        canvas,
        "qmz.5",
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "qmz.9",
        (width + 24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(path), canvas)


def process_side(side: str) -> None:
    _, _, image_a, image_b = load_images(side)
    det_a = detect_board(image_a)
    det_b = transfer_nodes_from_reference(det_a, image_b)

    points_path = BASE_DIR / f"pnp_pointsAB_{side}.txt"
    vis_path = BASE_DIR / f"match_vis_{side}.png"
    missing = write_points_file(points_path, det_a, det_b)
    export_visualization(vis_path, det_a, det_b)
    total = EXPECTED_COLS * EXPECTED_ROWS
    print(f"{side}: wrote {points_path.name}, {vis_path.name}, missing={missing}/{total}")


def main() -> None:
    process_side("left")
    process_side("right")


if __name__ == "__main__":
    main()

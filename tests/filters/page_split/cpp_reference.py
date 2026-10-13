"""Step-by-step transcription of upstream ScanTailor Advanced 1.2.1 page split.

This is the oracle for the tests: it follows ``PageLayoutEstimator.cpp``,
``VertLineFinder.cpp``, ``ContentSpanFinder.cpp`` and the imageproc routines they
call line by line, with explicit loops and C++ integer arithmetic, and shares no
code with the port. Black pixels are True. It is slow; use small pages.
"""

import math
from collections import deque

import numpy as np
from numba import njit

DEG2RAD = math.pi / 180.0
MM2INCH = 1.0 / 25.4


def q_round(value: float) -> int:
    return int(value + 0.5) if value >= 0 else int(value - 0.5)


# ----------------------------------------------------------------------------
# Transform.cpp: transformToGray for a pure scale, outside pixels weak black.
# ----------------------------------------------------------------------------


@njit(cache=True)
def _q_round_nb(value):
    if value >= 0:
        return int(value + 0.5)
    return int(value - 0.5)


@njit(cache=True)
def transform_to_gray(src, inv_m11, inv_m22, dw, dh, min_w, min_h):
    """transformGeneric, GrayColorMixer<unsigned>, outside pixels weak and black."""
    sh, sw = src.shape
    unit_w = max(min_w * 32.0, inv_m11)
    unit_h = max(min_h * 32.0, inv_m22)
    src32_unit_w = max(1, _q_round_nb(unit_w))
    src32_unit_h = max(1, _q_round_nb(unit_h))
    dst = np.zeros((dh, dw), dtype=np.uint8)

    for dy in range(dh):
        f_dy_center = dy + 0.5
        f_sx32_base = f_dy_center * 0.0 + 0.0
        f_sy32_base = f_dy_center * inv_m22 + 0.0
        for dx in range(dw):
            f_dx_center = dx + 0.5
            f_sx32_center = f_sx32_base + f_dx_center * inv_m11
            f_sy32_center = f_sy32_base + f_dx_center * 0.0
            src32_left = int(f_sx32_center) - (src32_unit_w >> 1)
            src32_top = int(f_sy32_center) - (src32_unit_h >> 1)
            src32_right = src32_left + src32_unit_w
            src32_bottom = src32_top + src32_unit_h
            src_left = src32_left >> 5
            src_right = (src32_right - 1) >> 5
            src_top = src32_top >> 5
            src_bottom = (src32_bottom - 1) >> 5

            if src_bottom < 0 or src_right < 0 or src_left >= sw or src_top >= sh:
                dst[dy, dx] = 0
                continue

            if src_top < 0:
                src_top = 0
                src32_top = 0
            if src_bottom >= sh:
                src_bottom = sh - 1
                src32_bottom = sh << 5
            if src_left < 0:
                src_left = 0
                src32_left = 0
            if src_right >= sw:
                src_right = sw - 1
                src32_right = sw << 5

            left_fraction = 32 - (src32_left & 31)
            top_fraction = 32 - (src32_top & 31)
            right_fraction = src32_right - (src_right << 5)
            bottom_fraction = src32_bottom - (src_bottom << 5)

            src_area = (src32_bottom - src32_top) * (src32_right - src32_left)
            if src_area == 0:
                dst[dy, dx] = 0
                continue

            accum = 0
            if src_top == src_bottom:
                if src_left == src_right:
                    dst[dy, dx] = src[src_top, src_left]
                    continue
                vert_fraction = src32_bottom - src32_top
                left_area = vert_fraction * left_fraction
                middle_area = vert_fraction << 5
                right_area = vert_fraction * right_fraction
                accum += np.int64(src[src_top, src_left]) * left_area
                for sx in range(src_left + 1, src_right):
                    accum += np.int64(src[src_top, sx]) * middle_area
                accum += np.int64(src[src_top, src_right]) * right_area
            elif src_left == src_right:
                hor_fraction = src32_right - src32_left
                top_area = hor_fraction * top_fraction
                middle_area = hor_fraction << 5
                bottom_area = hor_fraction * bottom_fraction
                accum += np.int64(src[src_top, src_left]) * top_area
                for sy in range(src_top + 1, src_bottom):
                    accum += np.int64(src[sy, src_left]) * middle_area
                accum += np.int64(src[src_bottom, src_left]) * bottom_area
            else:
                top_area = top_fraction << 5
                bottom_area = bottom_fraction << 5
                left_area = left_fraction << 5
                right_area = right_fraction << 5
                accum += np.int64(src[src_top, src_left]) * (
                    top_fraction * left_fraction
                )
                for sx in range(src_left + 1, src_right):
                    accum += np.int64(src[src_top, sx]) * top_area
                accum += np.int64(src[src_top, src_right]) * (
                    top_fraction * right_fraction
                )
                for sy in range(src_top + 1, src_bottom):
                    accum += np.int64(src[sy, src_left]) * left_area
                    for sx in range(src_left + 1, src_right):
                        accum += np.int64(src[sy, sx]) * (32 * 32)
                    accum += np.int64(src[sy, src_right]) * right_area
                accum += np.int64(src[src_bottom, src_left]) * (
                    bottom_fraction * left_fraction
                )
                for sx in range(src_left + 1, src_right):
                    accum += np.int64(src[src_bottom, sx]) * bottom_area
                accum += np.int64(src[src_bottom, src_right]) * (
                    bottom_fraction * right_fraction
                )
            dst[dy, dx] = (accum + (src_area >> 1)) // src_area
    return dst


# ----------------------------------------------------------------------------
# Scale.cpp: scaleToGray.
# ----------------------------------------------------------------------------


@njit(cache=True)
def _scale_down_int(src, dw, dh):
    sh, sw = src.shape
    xscale = sw // dw
    yscale = sh // dh
    total_area = xscale * yscale
    dst = np.zeros((dh, dw), dtype=np.uint8)
    for dy in range(dh):
        for dx in range(dw):
            gray_level = 0
            for i in range(yscale):
                for j in range(xscale):
                    gray_level += src[dy * yscale + i, dx * xscale + j]
            dst[dy, dx] = (gray_level + (total_area >> 1)) // total_area
    return dst


@njit(cache=True)
def _scale_up_int(src, dw, dh):
    sh, sw = src.shape
    xscale = dw // sw
    yscale = dh // sh
    dst = np.zeros((dh, dw), dtype=np.uint8)
    for sy in range(sh):
        for sx in range(sw):
            for i in range(yscale):
                for j in range(xscale):
                    dst[sy * yscale + i, sx * xscale + j] = src[sy, sx]
    return dst


@njit(cache=True)
def _calc_32x_ratio1(dst, src):
    src32 = src << 5
    ratio = src32 / dst
    while (int(ratio * (dst - 1)) >> 5) + 1 >= src:
        src32 -= 1
        ratio = src32 / dst
    return ratio


@njit(cache=True)
def _calc_32x_ratio2(dst, src):
    src32 = src << 5
    ratio = src32 / dst
    while ((int(ratio * dst) - 1) >> 5) >= src:
        src32 -= 1
        ratio = src32 / dst
    return ratio


@njit(cache=True)
def _scale_up(src, dw, dh):
    sh, sw = src.shape
    dx2sx32 = _calc_32x_ratio1(dw, sw)
    dy2sy32 = _calc_32x_ratio1(dh, sh)
    dst = np.zeros((dh, dw), dtype=np.uint8)
    for dy in range(dh):
        sy32 = int(dy * dy2sy32)
        sy = sy32 >> 5
        top_fraction = 32 - (sy32 & 31)
        bottom_fraction = sy32 & 31
        for dx in range(dw):
            sx32 = int(dx * dx2sx32)
            sx = sx32 >> 5
            left_fraction = 32 - (sx32 & 31)
            right_fraction = sx32 & 31
            gray_level = 0
            gray_level += np.int64(src[sy, sx]) * left_fraction * top_fraction
            gray_level += np.int64(src[sy, sx + 1]) * right_fraction * top_fraction
            gray_level += (
                np.int64(src[sy + 1, sx + 1]) * right_fraction * bottom_fraction
            )
            gray_level += np.int64(src[sy + 1, sx]) * left_fraction * bottom_fraction
            total_area = 32 * 32
            dst[dy, dx] = (gray_level + (total_area >> 1)) // total_area
    return dst


@njit(cache=True)
def _scale_generic(src, dw, dh):
    sh, sw = src.shape
    dx2sx32 = _calc_32x_ratio2(dw, sw)
    dy2sy32 = _calc_32x_ratio2(dh, sh)
    dst = np.zeros((dh, dw), dtype=np.uint8)
    sy32bottom = 0
    for dy1 in range(1, dh + 1):
        sy32top = sy32bottom
        sy32bottom = int(dy1 * dy2sy32)
        sytop = sy32top >> 5
        sybottom = (sy32bottom - 1) >> 5
        top_fraction = 32 - (sy32top & 31)
        bottom_fraction = sy32bottom - (sybottom << 5)
        top_area = top_fraction << 5
        bottom_area = bottom_fraction << 5
        sx32right = 0
        for dx in range(dw):
            sx32left = sx32right
            sx32right = int((dx + 1) * dx2sx32)
            sxleft = sx32left >> 5
            sxright = (sx32right - 1) >> 5
            left_fraction = 32 - (sx32left & 31)
            right_fraction = sx32right - (sxright << 5)
            gray_level = 0
            if sytop == sybottom:
                if sxleft == sxright:
                    dst[dy1 - 1, dx] = src[sytop, sxleft]
                    continue
                vert_fraction = sy32bottom - sy32top
                left_area = vert_fraction * left_fraction
                middle_area = vert_fraction << 5
                right_area = vert_fraction * right_fraction
                gray_level += np.int64(src[sytop, sxleft]) * left_area
                for sx in range(sxleft + 1, sxright):
                    gray_level += np.int64(src[sytop, sx]) * middle_area
                gray_level += np.int64(src[sytop, sxright]) * right_area
            elif sxleft == sxright:
                hor_fraction = sx32right - sx32left
                t_area = hor_fraction * top_fraction
                middle_area = hor_fraction << 5
                b_area = hor_fraction * bottom_fraction
                gray_level += np.int64(src[sytop, sxleft]) * t_area
                for sy in range(sytop + 1, sybottom):
                    gray_level += np.int64(src[sy, sxleft]) * middle_area
                gray_level += np.int64(src[sybottom, sxleft]) * b_area
            else:
                left_area = left_fraction << 5
                right_area = right_fraction << 5
                gray_level += np.int64(src[sytop, sxleft]) * (
                    top_fraction * left_fraction
                )
                for sx in range(sxleft + 1, sxright):
                    gray_level += np.int64(src[sytop, sx]) * top_area
                gray_level += np.int64(src[sytop, sxright]) * (
                    top_fraction * right_fraction
                )
                for sy in range(sytop + 1, sybottom):
                    gray_level += np.int64(src[sy, sxleft]) * left_area
                    for sx in range(sxleft + 1, sxright):
                        gray_level += np.int64(src[sy, sx]) << 10
                    gray_level += np.int64(src[sy, sxright]) * right_area
                gray_level += np.int64(src[sybottom, sxleft]) * (
                    bottom_fraction * left_fraction
                )
                for sx in range(sxleft + 1, sxright):
                    gray_level += np.int64(src[sybottom, sx]) * bottom_area
                gray_level += np.int64(src[sybottom, sxright]) * (
                    bottom_fraction * right_fraction
                )
            total_area = (sy32bottom - sy32top) * (sx32right - sx32left)
            dst[dy1 - 1, dx] = (gray_level + (total_area >> 1)) // total_area
    return dst


def scale_to_gray(src, dw, dh):
    sh, sw = src.shape
    if sw == dw and sh == dh:
        return src
    if sw % dw == 0 and sh % dh == 0:
        return _scale_down_int(src, dw, dh)
    if dw % sw == 0 and dh % sh == 0:
        return _scale_up_int(src, dw, dh)
    if dw > sw and dh > sh:
        return _scale_up(src, dw, dh)
    return _scale_generic(src, dw, dh)


# ----------------------------------------------------------------------------
# HoughLineDetector.cpp
# ----------------------------------------------------------------------------


@njit(cache=True)
def _hough_process_all(hist, cos_a, sin_a, bias, recip, xs, ys, weights):
    for i in range(xs.shape[0]):
        for a in range(cos_a.shape[0]):
            distance = cos_a[a] * xs[i] + sin_a[a] * ys[i]
            biased = distance + bias
            b = int(biased * recip + 0.5)
            hist[a, b] += weights[i]


def _max3x1(src):
    h, w = src.shape
    if w == 1:
        return src.copy()
    dst = np.zeros_like(src)
    for y in range(h):
        dst[y, 0] = max(src[y, 0], src[y, 1])
        for x in range(1, w - 1):
            dst[y, x] = max(src[y, x - 1], src[y, x], src[y, x + 1])
        dst[y, w - 1] = max(src[y, w - 1], src[y, w - 2])
    return dst


def _max1x3(src):
    return _max3x1(src.T.copy()).T.copy()


def _max5x5(src):
    tmp = _max3x1(src)
    dst = _max3x1(tmp)
    tmp = _max1x3(dst)
    return _max1x3(tmp)


def _peak_candidates(hist, lower_bound):
    maxed = _max5x5(hist)
    return (hist >= lower_bound) & (hist == maxed)


def _dilate_brick_5x5(img):
    h, w = img.shape
    out = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            if img[y, x]:
                out[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3] = True
    return out


def _seed_fill_8(seed, mask):
    h, w = mask.shape
    out = np.zeros_like(mask)
    queue = deque()
    for y, x in zip(*np.nonzero(seed & mask), strict=True):
        if not out[y, x]:
            out[y, x] = True
            queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not out[ny, nx]:
                    out[ny, nx] = True
                    queue.append((ny, nx))
    return out


def _find_histogram_peaks(hist, lower_bound):
    candidates = _peak_candidates(hist, lower_bound)
    neighborhood = _dilate_brick_5x5(candidates) ^ candidates
    new_hist = hist.copy()
    new_hist[neighborhood] += 1
    diff = _peak_candidates(new_hist, lower_bound) ^ candidates
    not_peaks = _seed_fill_8(diff, candidates)
    return candidates ^ not_peaks


def _conn_comps_8(img):
    """ConnCompEraser order: raster scan for the next black pixel, 8-connected."""
    work = img.copy()
    h, w = work.shape
    comps = []
    for y in range(h):
        for x in range(w):
            if not work[y, x]:
                continue
            x0 = x1 = x
            y0 = y1 = y
            count = 0
            stack = [(y, x)]
            work[y, x] = False
            while stack:
                cy, cx = stack.pop()
                count += 1
                x0, x1 = min(x0, cx), max(x1, cx)
                y0, y1 = min(y0, cy), max(y1, cy)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and work[ny, nx]:
                            work[ny, nx] = False
                            stack.append((ny, nx))
            comps.append(((x, y), (x0, y0, x1, y1), count))
    return comps


class HoughLine:
    def __init__(self, norm_uv, distance, quality):
        self.norm_uv = norm_uv
        self.distance = distance
        self.quality = quality

    def point_at_y(self, y):
        x = (self.distance - y * self.norm_uv[1]) / self.norm_uv[0]
        return (x, y)


def hough_find_lines(
    width,
    height,
    thickness,
    start_angle,
    angle_delta,
    num_angles,
    xs,
    ys,
    weights,
    quality_lower_bound,
):
    max_x, max_y = width - 1, height - 1
    checkpoints = [(max_x, max_y), (max_x, 0), (0, max_y)]
    max_distance = 0.0
    min_distance = 0.0
    uvs = []
    for i in range(num_angles):
        angle = start_angle + angle_delta * i
        angle *= DEG2RAD
        uv = (math.cos(angle), math.sin(angle))
        for px, py in checkpoints:
            distance = uv[0] * px + uv[1] * py
            max_distance = max(max_distance, distance)
            min_distance = min(min_distance, distance)
        uvs.append(uv)
    bias = -min_distance
    recip = 1.0 / thickness
    max_bin = int((max_distance + bias) * recip + 0.5)
    hist = np.zeros((num_angles, max_bin + 1), dtype=np.int64)
    _hough_process_all(
        hist,
        np.array([u[0] for u in uvs]),
        np.array([u[1] for u in uvs]),
        bias,
        recip,
        np.asarray(xs, dtype=np.int64),
        np.asarray(ys, dtype=np.int64),
        np.asarray(weights, dtype=np.int64),
    )
    peaks = _find_histogram_peaks(hist, quality_lower_bound)
    lines = []
    for seed, (x0, y0, x1, y1), _count in _conn_comps_8(peaks):
        level = int(hist[seed[1], seed[0]])
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        distance = (cx + 0.5) * thickness - bias
        lines.append(HoughLine(uvs[cy], distance, level))
    lines.sort(key=lambda line: -line.quality)
    return lines


# ----------------------------------------------------------------------------
# VertLineFinder.cpp
# ----------------------------------------------------------------------------


@njit(cache=True)
def _gray_window(src, half_x, half_y, take_max, outside):
    """dilateOrErodeGray, centred (2*half+1) brick, outside pixels = ``outside``."""
    h, w = src.shape
    dst = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            best = 0 if take_max else 255
            for dy in range(-half_y, half_y + 1):
                for dx in range(-half_x, half_x + 1):
                    yy, xx = y + dy, x + dx
                    v = src[yy, xx] if (0 <= yy < h and 0 <= xx < w) else outside
                    if take_max:
                        best = max(best, v)
                    else:
                        best = min(best, v)
            dst[y, x] = best
    return dst


@njit(cache=True)
def _close_gray_column(src, half_y, outside):
    """closeGray with a 1 x (2*half_y+1) brick: Darker (min) then Lighter (max)."""
    h, w = src.shape
    tmp = np.zeros((h + 2 * half_y, w), dtype=np.uint8)
    for ty in range(h + 2 * half_y):
        for x in range(w):
            best = 255
            for dy in range(-half_y, half_y + 1):
                yy = ty - half_y + dy
                v = src[yy, x] if 0 <= yy < h else outside
                best = min(best, v)
            tmp[ty, x] = best
    dst = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            best = 0
            for dy in range(-half_y, half_y + 1):
                best = max(best, tmp[y + half_y + dy, x])
            dst[y, x] = best
    return dst


def build_weight_table():
    table = [0] * 256
    gray_level = 0
    weight = 2
    segment = 2
    prev_segment = 1
    while gray_level < 256:
        limit = min(256, gray_level + segment)
        while gray_level < limit:
            table[gray_level] = weight
            gray_level += 1
        weight += 1
        segment += prev_segment
        prev_segment = segment
    return table


def vert_line_finder(gray, dpi, max_lines):
    """VertLineFinder::findLines; lines are ((x1, y1), (x2, y2)) in input pixels."""
    dpi100 = 100
    xscale = dpi100 / dpi
    gh, gw = gray.shape
    target_w = q_round(gw * xscale)
    target_h = q_round(gh * xscale)
    if target_w <= 0 or target_h <= 0:
        target_w = target_h = 1
    inv = (1.0 / xscale) * 32.0
    gray100 = transform_to_gray(gray, inv, inv, target_w, target_h, 5.0, 5.0)

    h_gradient = _gray_window(gray100, 5, 0, True, 0)
    v_gradient = _gray_window(gray100, 0, 5, True, 0)
    h_gradient = np.clip(
        h_gradient.astype(np.int32) - v_gradient.astype(np.int32), 0, 255
    ).astype(np.uint8)
    raster_lines = _close_gray_column(h_gradient, 9, 0)

    line_thickness = 5.0
    max_angle = 7.0
    angle_step = 0.25
    angle_steps_to_max = int(max_angle / angle_step)
    total_angle_steps = angle_steps_to_max * 2 + 1
    min_angle = -angle_steps_to_max * angle_step

    weight_table = build_weight_table()
    margin = math.floor(0.5 + 3.5 * MM2INCH * dpi100)
    height, width = raster_lines.shape
    x_limit = width - margin
    xs, ys, ws = [], [], []
    for y in range(height):
        for x in range(margin, x_limit):
            val = int(raster_lines[y, x])
            if val > 1:
                xs.append(x)
                ys.append(y)
                ws.append(weight_table[val])

    min_quality = int(height * line_thickness * 1.8) + 1
    hough_lines = hough_find_lines(
        width,
        height,
        line_thickness,
        min_angle,
        angle_step,
        total_angle_steps,
        xs,
        ys,
        ws,
        min_quality,
    )

    groups = []  # each: [leader, left, right]
    for hl in hough_lines:
        top = hl.point_at_y(0.0)
        bottom = hl.point_at_y(float(height))
        if top[0] < bottom[0]:
            new_line = (top, bottom, hl.quality)
        else:
            new_line = (bottom, top, hl.quality)
        home = None
        i = 0
        while i < len(groups):
            g = groups[i]
            if not (g[1] > new_line[1][0] or g[2] < new_line[0][0]):
                if home is not None:
                    home[1] = min(home[1], g[1])
                    home[2] = max(home[2], g[2])
                    if g[0][2] > home[0][2]:
                        home[0] = g[0]
                    del groups[i]
                    continue
                g[1] = min(g[1], new_line[0][0])
                g[2] = max(g[2], new_line[1][0])
                if new_line[2] > g[0][2]:
                    g[0] = new_line
                home = g
            i += 1
        if home is None:
            groups.append([new_line, new_line[0][0], new_line[1][0]])

    lines = []
    for g in groups:
        leader = g[0]
        lines.append((leader[0], leader[1]))
        if len(lines) == max_lines:
            break
    scale = dpi / dpi100
    return [
        ((a[0] * scale, a[1] * scale), (b[0] * scale, b[1] * scale)) for a, b in lines
    ]


# ----------------------------------------------------------------------------
# Binary image operations.
# ----------------------------------------------------------------------------


@njit(cache=True)
def _reduce_nb(img, threshold):
    h, w = img.shape
    dh, dw = h // 2, w // 2
    out = np.zeros((dh, dw), dtype=np.bool_)
    for y in range(dh):
        for x in range(dw):
            count = (
                int(img[2 * y, 2 * x])
                + int(img[2 * y, 2 * x + 1])
                + int(img[2 * y + 1, 2 * x])
                + int(img[2 * y + 1, 2 * x + 1])
            )
            out[y, x] = count >= threshold
    return out


def reduce_threshold(img, threshold):
    return _reduce_nb(img, threshold)


@njit(cache=True)
def _open_brick_nb(padded, bw, bh, pad_x, pad_y, h, w, surround):
    ph, pw = padded.shape
    sat = np.zeros((ph + 1, pw + 1), dtype=np.int64)
    for y in range(ph):
        row = 0
        for x in range(pw):
            row += padded[y, x]
            sat[y + 1, x + 1] = sat[y, x + 1] + row
    eroded = np.full((ph, pw), surround, dtype=np.int64)
    for y in range(ph - bh + 1):
        for x in range(pw - bw + 1):
            total = sat[y + bh, x + bw] - sat[y, x + bw] - sat[y + bh, x] + sat[y, x]
            eroded[y, x] = 1 if total == bw * bh else 0
    sat2 = np.zeros((ph + 1, pw + 1), dtype=np.int64)
    for y in range(ph):
        row = 0
        for x in range(pw):
            row += eroded[y, x]
            sat2[y + 1, x + 1] = sat2[y, x + 1] + row
    out = np.zeros((h, w), dtype=np.bool_)
    for y in range(h):
        for x in range(w):
            py = y + pad_y
            px = x + pad_x
            y0 = py - bh + 1
            x0 = px - bw + 1
            total = (
                sat2[py + 1, px + 1]
                - sat2[y0, px + 1]
                - sat2[py + 1, x0]
                + sat2[y0, x0]
            )
            out[y, x] = total > 0
    return out


def open_brick(img, bw, bh, surround_black):
    """Black pixels covered by a bw x bh rectangle lying wholly in black."""
    h, w = img.shape
    pad_x, pad_y = 3 * bw, 3 * bh
    padded = np.pad(
        img.astype(np.int64),
        ((pad_y, pad_y), (pad_x, pad_x)),
        constant_values=int(surround_black),
    )
    return _open_brick_nb(padded, bw, bh, pad_x, pad_y, h, w, int(surround_black))


def seed_fill_8(seed, mask):
    return _seed_fill_8(seed, mask)


def dilate_3x3(img):
    h, w = img.shape
    out = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            if img[y, x]:
                out[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = True
    return out


# ----------------------------------------------------------------------------
# Shear.cpp, SkewFinder.cpp
# ----------------------------------------------------------------------------


def v_shear(src, shear, x_origin):
    height, width = src.shape
    dst = np.zeros_like(src)
    shift = 0.5 + shear * (0.5 - x_origin)
    shift_end = 0.5 + shear * (width - 0.5 - x_origin)
    shift1 = math.floor(shift)
    if shift1 == math.floor(shift_end):
        return src.copy()
    x1 = 0
    x2 = 0
    while True:
        x2 += 1
        shift += shear
        shift2 = math.floor(shift)
        if shift1 != shift2 or x2 == width:
            bw = x2 - x1
            if abs(shift1) >= height:
                dst[:, x1 : x1 + bw] = False
            elif shift1 < 0:
                dst[: height + shift1, x1 : x1 + bw] = src[-shift1:, x1 : x1 + bw]
                dst[height + shift1 :, x1 : x1 + bw] = False
            elif shift1 > 0:
                dst[shift1:, x1 : x1 + bw] = src[: height - shift1, x1 : x1 + bw]
                dst[:shift1, x1 : x1 + bw] = False
            else:
                dst[:, x1 : x1 + bw] = src[:, x1 : x1 + bw]
            if x2 == width:
                break
            x1 = x2
            shift1 = shift2
    return dst


def h_shear(src, shear, y_origin):
    return v_shear(src.T.copy(), shear, y_origin).T.copy()


def _calc_score(img):
    counts = img.sum(axis=1)
    score = 0.0
    for y in range(1, len(counts)):
        diff = float(counts[y] - counts[y - 1])
        score += diff * diff
    return score


def find_skew(
    img,
    coarse_reduction=2,
    fine_reduction=1,
    accuracy=0.1,
    max_angle=7.0,
    min_angle=0.1,
    resolution_ratio=1.0,
):
    low_score = 1000.0
    good_confidence = 2.0
    min_reduction = min(coarse_reduction, fine_reduction)
    coarse = img
    for i in range(min_reduction):
        coarse = reduce_threshold(coarse, 1 if i == 0 else 2)
    fine = coarse
    for i in range(min_reduction, coarse_reduction):
        coarse = reduce_threshold(coarse, 1 if i == 0 else 2)

    def process(src, angle):
        tg = math.tan(angle * DEG2RAD)
        x_center = 0.5 * src.shape[1]
        return _calc_score(v_shear(src, tg / resolution_ratio, x_center))

    coarse_step = 1.0
    num_coarse = 0
    sum_coarse = 0.0
    best_coarse_score = 0.0
    best_coarse_angle = -max_angle
    angle = -max_angle
    while angle <= max_angle:
        score = process(coarse, angle)
        sum_coarse += score
        num_coarse += 1
        if score > best_coarse_score:
            best_coarse_angle = angle
            best_coarse_score = score
        angle += coarse_step

    if accuracy >= coarse_step:
        confidence = 0.0
        if num_coarse > 1:
            confidence = best_coarse_score / sum_coarse * num_coarse
        return (-best_coarse_angle, confidence - 1.0)

    for i in range(min_reduction, fine_reduction):
        fine = reduce_threshold(fine, 1 if i == 0 else 2)

    angle_plus = best_coarse_angle + 0.5 * coarse_step
    angle_minus = best_coarse_angle - 0.5 * coarse_step
    score_plus = process(fine, angle_plus)
    score_minus = process(fine, angle_minus)
    fine_score1 = score_plus
    fine_score2 = score_minus
    while angle_plus - angle_minus > accuracy:
        if score_plus > score_minus:
            angle_minus = 0.5 * (angle_plus + angle_minus)
            score_minus = process(fine, angle_minus)
        elif score_plus < score_minus:
            angle_plus = 0.5 * (angle_plus + angle_minus)
            score_plus = process(fine, angle_plus)
        else:
            break

    if score_plus > score_minus:
        best_angle, best_score = angle_plus, score_plus
    else:
        best_angle, best_score = angle_minus, score_minus

    if abs(best_angle) < min_angle:
        return (0.0, good_confidence)
    if best_score <= low_score:
        return (-best_angle, 0.0)
    if num_coarse > 1:
        confidence = best_score / sum_coarse * num_coarse
    else:
        num_scores = num_coarse + 2
        sum_scores = sum_coarse + fine_score1 + fine_score2
        confidence = best_score / sum_scores * num_scores
    return (-best_angle, confidence - 1.0)


# ----------------------------------------------------------------------------
# ContentSpanFinder.cpp and Span.h
# ----------------------------------------------------------------------------


class Span:
    def __init__(self, begin, end):
        self.begin = begin
        self.end = end

    @staticmethod
    def gap(a, b):
        """Span(begin, end) overloads: int means a point, Span means its inner edge."""
        begin = a.end if isinstance(a, Span) else a
        end = b.begin if isinstance(b, Span) else b
        return Span(begin, end)

    def width(self):
        return self.end - self.begin

    def center(self):
        return 0.5 * (self.begin + self.end)


def content_span_finder(hist, min_content_width, min_whitespace_width):
    spans = []
    hist_size = len(hist)
    i = 0
    content_end = -min_whitespace_width
    content_begin = content_end
    while True:
        while i < hist_size:
            if hist[i] != 0:
                break
            i += 1
        if i - content_end >= min_whitespace_width:
            if content_end - content_begin >= min_content_width:
                spans.append(Span(content_begin, content_end))
            content_begin = i
        if i == hist_size:
            break
        while i < hist_size:
            if hist[i] == 0:
                break
            i += 1
        content_end = i
    if content_end - content_begin >= min_content_width:
        spans.append(Span(content_begin, content_end))
    return spans


# ----------------------------------------------------------------------------
# PageLayoutEstimator.cpp
# ----------------------------------------------------------------------------

AUTO, SINGLE_UNCUT, PLUS_OFFCUT, TWO_PAGES = "auto", "uncut", "offcut", "two"
UNCUT, CUT, TWO = "single-uncut", "single-cut", "two-pages"


def line_center_x(line):
    return 0.5 * (line[0][0] + line[1][0])


def num_pages(layout_type, w, h):
    if layout_type == AUTO:
        return 2 if w > h else 1
    if layout_type in (SINGLE_UNCUT, PLUS_OFFCUT):
        return 1
    return 2


def auto_detect_single_page_layout(layout_type, ltr_lines, w, h):
    image_center = 0.5 * w
    uncut = False
    while True:  # a loop just to be able to break from it
        if layout_type == PLUS_OFFCUT:
            break
        if ltr_lines:
            if abs(image_center - line_center_x(ltr_lines[0])) > 0.65 * image_center:
                break
        if len(ltr_lines) > 1:
            if abs(image_center - line_center_x(ltr_lines[-1])) > 0.65 * image_center:
                break
        uncut = True
        break
    if uncut:
        return (UNCUT, None, None)
    if not ltr_lines:
        return None
    if len(ltr_lines) > 1:
        return (CUT, ltr_lines[0], ltr_lines[-1])
    line = ltr_lines[0]
    if line_center_x(line) < image_center:
        return (CUT, line, ((float(w), 0.0), (float(w), float(h))))
    return (CUT, ((0.0, 0.0), (0.0, float(h))), line)


def auto_detect_two_page_layout(ltr_lines, w):
    if not ltr_lines:
        return None
    if len(ltr_lines) == 1:
        return (TWO, ltr_lines[0], ltr_lines[0])
    image_center = 0.5 * w
    min_distance = float("inf")
    best = None
    for line in ltr_lines:
        distance = abs(line_center_x(line) - image_center)
        if distance < min_distance:
            min_distance = distance
            best = line
    return (TWO, best, best)


def try_cut_at_folding_line(layout_type, gray, dpi):
    h, w = gray.shape
    pages = num_pages(layout_type, w, h)
    lines = vert_line_finder(gray, dpi, 8)
    lines.sort(key=line_center_x)
    center_x = 0.5 * w
    if pages == 1:
        while len(lines) > 1:
            left_line = lines[0]
            right_line = lines[-1]
            threshold = 0.3 * center_x
            left_dist = center_x - line_center_x(left_line)
            right_dist = center_x - line_center_x(right_line)
            if (left_dist < 0) != (right_dist < 0):
                break
            left_dist = abs(left_dist)
            right_dist = abs(right_dist)
            if left_dist < threshold or right_dist < threshold:
                break
            lines = [left_line if left_dist < right_dist else right_line]
            break
        return auto_detect_single_page_layout(layout_type, lines, w, h)
    lines = [
        ln for ln in lines if not abs(line_center_x(ln) - center_x) > 0.6 * center_x
    ]
    return auto_detect_two_page_layout(lines, w)


def to_300_dpi_binary(gray, dpi, threshold):
    xfactor = 300.0 / dpi
    xform = np.eye(3)
    if abs(xfactor - 1.0) < 0.1:
        return gray < threshold, xform
    xform = xform @ np.diag([xfactor, xfactor, 1.0])
    new_w = max(1, math.ceil(xfactor * gray.shape[1]))
    new_h = max(1, math.ceil(xfactor * gray.shape[0]))
    return scale_to_gray(gray, new_w, new_h) < threshold, xform


def remove_garbage_and_2x_downscale(img):
    reduced = reduce_threshold(img, 2)
    seed = open_brick(reduced, 4, 1, False)
    seed2 = open_brick(reduced, 1, 4, False)
    seed = seed | seed2
    reduced = seed_fill_8(seed, reduced)
    hor_seed = open_brick(reduced, 200, 14, True)
    ver_seed = open_brick(reduced, 14, 300, True)
    seed = hor_seed | ver_seed
    dilated = dilate_3x3(reduced)
    shadows_dilated = seed_fill_8(seed, dilated)
    return reduced & ~shadows_dilated


def check_for_left_offcut(img):
    margin, width = 2, 3
    h = img.shape[0]
    return bool(img[margin : h - margin, margin : margin + width].any())


def check_for_right_offcut(img):
    margin, width = 2, 3
    h, w = img.shape
    return bool(img[margin : h - margin, w - margin - width : w - margin].any())


def remove_insignificant_edge_spans(spans):
    if not spans:
        return
    total = sum(span.width() for span in spans)
    may_be_removed = total // 15
    while True:
        first = spans[0]
        last = spans[-1]
        if first is last:
            break
        if first.width() < last.width():
            if first.width() > may_be_removed:
                break
            may_be_removed -= first.width()
            spans.pop(0)
        else:
            if last.width() > may_be_removed:
                break
            may_be_removed -= last.width()
            spans.pop()
        if not spans:
            break


def vert_line(x):
    return ((x, 0.0), (x, 1.0))


def process_content_spans_single_page(
    layout_type, spans, w, h, left_offcut, right_offcut
):
    right_edge = ((float(w), 0.0), (float(w), float(h)))
    left_edge = ((0.0, 0.0), (0.0, float(h)))
    while left_offcut and not right_offcut and layout_type == AUTO:
        if not spans:
            x = 0.0
        elif spans[0].begin > 0:
            x = 0.5 * spans[0].begin
        else:
            if spans[0].width() > w // 2:
                break
            elif len(spans) > 1:
                x = Span.gap(spans[0], spans[1]).center()
            else:
                x = min(spans[0].end + 20, w)
        return (CUT, vert_line(float(x)), right_edge)
    while right_offcut and not left_offcut and layout_type == AUTO:
        if not spans:
            x = w
        elif spans[-1].end < w:
            x = Span.gap(spans[-1], w).center()
        else:
            if spans[-1].width() > w // 2:
                break
            elif len(spans) > 1:
                x = Span.gap(spans[-2], spans[-1]).center()
            else:
                x = max(spans[-1].begin - 20, 0)
        return (CUT, left_edge, vert_line(float(x)))
    if layout_type == PLUS_OFFCUT:
        return (CUT, left_edge, right_edge)
    return (UNCUT, None, None)


def process_two_pages_with_single_span(span, w):
    page_center = 0.5 * w
    box_center = span.center()
    box_half_width = 0.5 * span.width()
    distance_to_page_center = abs(page_center - box_center) - box_half_width
    if distance_to_page_center > 15:
        x = page_center
    else:
        left_ws = Span.gap(0, span)
        right_ws = Span.gap(span, w)
        if left_ws.width() > right_ws.width():
            x = max(0, span.begin - 15)
        else:
            x = min(w, span.end + 15)
    line = vert_line(float(x))
    return (TWO, line, line)


def process_content_spans_two_pages(spans, w):
    if not spans:
        x = 0.5 * w
    elif len(spans) == 1:
        return process_two_pages_with_single_span(spans[0], w)
    else:
        content_begin = spans[0].begin
        content_end = spans[-1].end
        gaps = [
            (spans[i].end - content_begin, content_end - spans[i + 1].begin)
            for i in range(len(spans) - 1)
        ]
        best_gap = 0
        best_ratio = 0.0
        for i, (a, b) in enumerate(gaps):
            mn, mx = float(min(a, b)), float(max(a, b))
            ratio = mn / mx
            if ratio > best_ratio:
                best_ratio = ratio
                best_gap = i
        if best_ratio < 0.25:
            return process_two_pages_with_single_span(
                Span(content_begin, content_end), w
            )
        acceptable_ratio = best_ratio * 0.90
        widest_gap = best_gap
        max_width = Span.gap(spans[best_gap], spans[best_gap + 1]).width()
        for i in range(best_gap - 1, -1, -1):
            mn, mx = float(min(gaps[i])), float(max(gaps[i]))
            if mn / mx < acceptable_ratio:
                break
            gw = Span.gap(spans[i], spans[i + 1]).width()
            if gw > max_width:
                max_width = gw
                widest_gap = i
        for i in range(best_gap + 1, len(gaps)):
            mn, mx = float(min(gaps[i])), float(max(gaps[i]))
            if mn / mx < acceptable_ratio:
                break
            gw = Span.gap(spans[i], spans[i + 1]).width()
            if gw > max_width:
                max_width = gw
                widest_gap = i
        x = Span.gap(spans[widest_gap], spans[widest_gap + 1]).center()
    line = vert_line(x)
    return (TWO, line, line)


def cut_at_whitespace_deskewed_150(layout_type, pages, img, left_offcut, right_offcut):
    h, w = img.shape
    cc_img = np.zeros_like(img)
    for _seed, (x0, y0, x1, y1), _count in _conn_comps_8(img):
        cw, ch = x1 - x0 + 1, y1 - y0 + 1
        if cw < 5 or ch < 5:
            continue
        if ch / cw > 6:
            continue
        cc_img[y0 : y1 + 1, x0 : x1 + 1] = True
    hist = cc_img.sum(axis=0)
    spans = content_span_finder(hist, 2, 8)
    if pages == 1:
        return process_content_spans_single_page(
            layout_type, spans, w, h, left_offcut, right_offcut
        )
    remove_insignificant_edge_spans(spans)
    return process_content_spans_two_pages(spans, w)


def _map(matrix, line):
    out = []
    for x, y in line:
        p = np.array([x, y, 1.0]) @ matrix
        out.append((float(p[0]), float(p[1])))
    return tuple(out)


def cut_at_whitespace(layout_type, gray, dpi, threshold):
    img, xform = to_300_dpi_binary(gray, dpi, threshold)
    img = remove_garbage_and_2x_downscale(img)
    # QTransform::scale pre-multiplies, which for two diagonal matrices commutes.
    xform = np.diag([0.5, 0.5, 1.0]) @ xform

    left_offcut = check_for_left_offcut(img)
    right_offcut = check_for_right_offcut(img)

    angle, confidence = find_skew(img, 0, 0, 0.5)
    if angle != 0.0 and confidence >= 2.0:
        h, w = img.shape
        tg = math.tan(angle * DEG2RAD)
        margin = math.ceil(abs(0.5 * h * tg))
        new_width = w - margin * 2
        if new_width > 0:
            sheared = h_shear(img, tg, 0.5 * h)
            img = sheared[:, margin : margin + new_width].copy()
            t1 = np.array([[1, 0, 0], [0, 1, 0], [-0.5 * w, -0.5 * h, 1.0]])
            t2 = np.array([[1, 0, 0], [tg, 1, 0], [0, 0, 1.0]])
            t3 = np.array([[1, 0, 0], [0, 1, 0], [0.5 * w - margin, 0.5 * h, 1.0]])
            xform = xform @ t1 @ t2 @ t3

    pages = num_pages(layout_type, gray.shape[1], gray.shape[0])
    kind, c1, c2 = cut_at_whitespace_deskewed_150(
        layout_type, pages, img, left_offcut, right_offcut
    )
    back = np.linalg.inv(xform)
    if c1 is not None:
        c1 = _map(back, c1)
    if c2 is not None:
        c2 = _map(back, c2)
    return (kind, c1, c2)


def otsu_threshold(gray):
    hist = np.bincount(gray.ravel(), minlength=256)
    pixels_by_threshold = [0] * 256
    moment_by_threshold = [0] * 256
    pixels_by_threshold[0] = int(hist[0])
    for i in range(1, 256):
        pixels_by_threshold[i] = pixels_by_threshold[i - 1] + int(hist[i])
        moment_by_threshold[i] = moment_by_threshold[i - 1] + int(hist[i]) * i
    total_pixels = pixels_by_threshold[255]
    total_moment = moment_by_threshold[255]
    max_variance = 0.0
    first = last = -1
    for i in range(256):
        below = pixels_by_threshold[i]
        above = total_pixels - below
        if below > 0 and above > 0:
            moment_below = moment_by_threshold[i]
            moment_above = total_moment - moment_below
            mean_below = moment_below / below
            mean_above = moment_above / above
            mean_diff = mean_below - mean_above
            variance = mean_diff * mean_diff * below * above
            if variance > max_variance:
                max_variance = variance
                first = last = i
            elif variance == max_variance:
                last = i
    first += 1
    last += 1
    return (first + last) >> 1


def estimate_page_layout(layout_type, gray, dpi, threshold=None):
    """Return (kind, cutter1, cutter2); cutters are raw ((x1, y1), (x2, y2))."""
    h, w = gray.shape
    if layout_type == SINGLE_UNCUT:
        return (UNCUT, None, None)
    layout = try_cut_at_folding_line(layout_type, gray, dpi)
    if layout is not None:
        return layout
    if threshold is None:
        threshold = otsu_threshold(gray)
    return cut_at_whitespace(layout_type, gray, dpi, threshold)


def x_at(line, y):
    """X coordinate of the (infinite) line at height y."""
    (x1, y1), (x2, y2) = line
    return x1 + (y - y1) * (x2 - x1) / (y2 - y1)

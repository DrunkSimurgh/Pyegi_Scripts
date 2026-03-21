from PyegiInitializer import *
# -*- coding: utf-8 -*-
"""
Originally created on November 12 2022
Updated on March 20 2026

@author: Drunk Simurgh
"""

from pyonfx import *
import re
import numpy as np
from copy import deepcopy
import math
from scipy.optimize import least_squares
import time
from fractions import Fraction
from pathlib import Path
import os
from video_timestamps import FPSTimestamps, RoundingMethod, VideoTimestamps

import Pyegi

project_properties_table = Pyegi.get_project_properties()
video_file = project_properties_table["video_file"]

parameters_table = Pyegi.get_parameters()
for items in parameters_table["Windows"][0]["Controls"]:
    if items["name"] == "floatedit1":
        object_frx1 = items["value"]
    if items["name"] == "floatedit2":
        object_frx2 = items["value"]
    if items["name"] == "floatedit3":
        acc = items["value"]
    if items["name"] == "dropdown1":
        calculation_type = items["value"]
    # if items["name"] == "floatedit4":
    #     axis_offset = items["value"]

axis_offset = 0

# ============================================================
# POTENTIALLY WILL BE REMOVED IN LATER VERSIONS
# ============================================================

def ensure_meta_timestamps(meta):
    if meta.timestamps is not None:
        return meta.timestamps

    if not video_file:
        return None

    is_dummy = video_file.startswith("?dummy")

    if not is_dummy and os.path.isfile(video_file):
        meta.timestamps = VideoTimestamps.from_video_file(Path(video_file))
    elif is_dummy:
        parts = video_file.split(":")
        if len(parts) >= 2:
            fps = Fraction(parts[1])
            meta.timestamps = FPSTimestamps(
                RoundingMethod.ROUND,
                Fraction(1000),
                fps,
                Fraction(0),
            )

    return meta.timestamps

# ============================================================
# ASS / BASIC CONSTANTS
# ============================================================

io = Ass(Pyegi.get_input_file_path(), extended=True)
meta, styles, lines = io.get_data()
ensure_meta_timestamps(meta)

np_a = np.array
m_r = math.radians
m_c = math.cos
m_s = math.sin

frs = 24000 / 1001
fT = 1000 / frs

k = 0.00321 * 720 / meta.play_res_y
M = np_a([
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [0., 0., k, 1.]
])

RE_POS = re.compile(r"pos\((\-?\d*\.?\d*),(\-?\d*\.?\d*)")
RE_FRZ = re.compile(r"frz(\-?\d*\.?\d*)")
RE_P   = re.compile(r"p\d+")
RE_P = re.compile(r"\\p-?\d+\b", re.IGNORECASE)
RE_DRAWING_TEXT = re.compile(
    r"^\s*(?:m|n|l|b|s|p|c)\s+[-\d]",
    re.IGNORECASE
)

solver_max_nfev = 120
residual_tol = 2e-2


# ============================================================
# ROTATION MATRICES
# ============================================================

def Rx(a):
    return np_a([
        [1., 0.,           0.,          0.],
        [0., m_c(m_r(a)),  m_s(m_r(a)), 0.],
        [0., -m_s(m_r(a)), m_c(m_r(a)), 0.],
        [0., 0.,           0.,          1.]
    ])


def Ry(b):
    return np_a([
        [m_c(m_r(b)), 0., -m_s(m_r(b)), 0.],
        [0.,          1., 0.,           0.],
        [m_s(m_r(b)), 0.,  m_c(m_r(b)), 0.],
        [0.,          0., 0.,           1.]
    ])


def Rz(c):
    return np_a([
        [ m_c(m_r(c)), m_s(m_r(c)), 0., 0.],
        [-m_s(m_r(c)), m_c(m_r(c)), 0., 0.],
        [ 0.,          0.,          1., 0.],
        [ 0.,          0.,          0., 1.]
    ])


def u(theta):
    return np_a([
        [m_c(m_r(theta))],
        [-m_s(m_r(theta))],
        [0.]
    ])


def Ru(d, u_vec):
    M_t = np_a([
        [0.,          -u_vec[2, 0],  u_vec[1, 0]],
        [u_vec[2, 0],  0.,          -u_vec[0, 0]],
        [-u_vec[1, 0], u_vec[0, 0],  0.]
    ])
    R_t = M_t * m_s(m_r(-d)) + (np.eye(3) - u_vec @ u_vec.T) * m_c(m_r(d)) + u_vec @ u_vec.T
    R = np.eye(4)
    R[:3, :3] = R_t
    return R


# ============================================================
# ORIGINAL HELPERS
# ============================================================

def rotate_z(x, y, theta):
    xy_vector = np_a([[x], [y], [0.], [1.]])
    xy2 = Rz(theta) @ xy_vector
    return xy2[0, 0], xy2[1, 0]


def rotated_xy(x, y, theta, u1):
    xy_vector = np_a([[x], [y], [0.], [1.]])
    T = M @ Ru(theta, u1)
    Re = T @ xy_vector
    if Re[3, 0] != 0:
        Re = Re / Re[3, 0]
    return Re[0, 0], Re[1, 0]


def c1(arg1):
    return m_c(m_r(arg1))


def s1(arg1):
    return m_s(m_r(arg1))


def equ_x(x0, y0, x_O, a, b, c, val_x):
    return (
        x0 * (c1(b) * c1(c) - s1(a) * s1(b) * s1(c))
        + y0 * (c1(b) * s1(c) + s1(a) * s1(b) * c1(c))
    ) / (
        k * (
            x0 * (s1(b) * c1(c) + s1(a) * c1(b) * s1(c))
            + y0 * (s1(b) * s1(c) - s1(a) * c1(b) * c1(c))
        ) + 1
    ) + x_O - val_x


def equ_y(x0, y0, y_O, a, b, c, val_y):
    return (
        x0 * (-c1(a) * s1(c))
        + y0 * (c1(a) * c1(c))
    ) / (
        k * (
            x0 * (s1(b) * c1(c) + s1(a) * c1(b) * s1(c))
            + y0 * (s1(b) * s1(c) - s1(a) * c1(b) * c1(c))
        ) + 1
    ) + y_O - val_y


# ============================================================
# NUMERICAL HELPERS
# ============================================================

def rotate_z_points(points_xy, theta_deg):
    c = math.cos(math.radians(theta_deg))
    s = math.sin(math.radians(theta_deg))
    R = np.array([
        [c,  s],
        [-s, c]
    ], dtype=float)
    return points_xy @ R.T


def project_points(points_xy, theta_deg, u1):
    pts = np.column_stack((
        points_xy[:, 0],
        points_xy[:, 1],
        np.zeros(len(points_xy)),
        np.ones(len(points_xy))
    ))
    T = M @ Ru(theta_deg, u1)
    out = (T @ pts.T).T
    nz = np.abs(out[:, 3]) > 1e-12
    out[nz] /= out[nz, 3:4]
    return out[:, :2]


def in_swap_region_x(theta_deg):
    t = theta_deg % 360.0
    return 90.0 < t < 270.0


def in_swap_region_z(frz_deg):
    t = frz_deg % 360.0
    return 90.0 < t < 270.0


def angle_matches_branch(angle_deg, want_swap):
    t = angle_deg % 360.0
    if want_swap:
        return 90.0 < t < 270.0
    return (0.0 <= t < 90.0) or (270.0 < t < 360.0)


def geometric_residual(z, targets):
    x0_1, x0_2, y0_1, y0_2, a, b, c, x_org, y_org = z

    x0 = np.array([x0_1, x0_2, x0_2, x0_1], dtype=float)
    y0 = np.array([y0_1, y0_1, y0_2, y0_2], dtype=float)

    ca = math.cos(math.radians(a))
    sa = math.sin(math.radians(a))
    cb = math.cos(math.radians(b))
    sb = math.sin(math.radians(b))
    cc = math.cos(math.radians(c))
    sc = math.sin(math.radians(c))

    den = k * (
        x0 * (sb * cc + sa * cb * sc) +
        y0 * (sb * sc - sa * cb * cc)
    ) + 1.0

    x = (
        x0 * (cb * cc - sa * sb * sc) +
        y0 * (cb * sc + sa * sb * cc)
    ) / den + x_org

    y = (
        x0 * (-ca * sc) +
        y0 * (ca * cc)
    ) / den + y_org

    pred = np.column_stack((x, y)).ravel()
    return pred - targets


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def continuity_cost(z, prev_z, w0, h0):
    """
    Small tie-break cost used only after a candidate already passed
    branch and geometric-fit checks.
    """
    if prev_z is None:
        return 0.0

    return (
        ((z[0] - prev_z[0]) / max(w0, 1.0)) ** 2 +
        ((z[1] - prev_z[1]) / max(w0, 1.0)) ** 2 +
        ((z[2] - prev_z[2]) / max(h0, 1.0)) ** 2 +
        ((z[3] - prev_z[3]) / max(h0, 1.0)) ** 2 +
        (wrap180(z[4] - prev_z[4]) / 180.0) ** 2 +
        (wrap180(z[5] - prev_z[5]) / 180.0) ** 2 +
        (wrap180(z[6] - prev_z[6]) / 180.0) ** 2 +
        ((z[7] - prev_z[7]) / max(w0, 1.0)) ** 2 +
        ((z[8] - prev_z[8]) / max(h0, 1.0)) ** 2
    )

def is_shape_line(line):
    raw = line.raw_text or ""
    txt = (line.text or "").strip()

    # Standard ASS drawing mode tag, e.g. {\p1}... or {\p4}...
    if RE_P.search(raw):
        return True

    # Fallback: stripped text itself looks like vector drawing commands
    # e.g. "m 0 0 l 100 0 100 50 0 50"
    if RE_DRAWING_TEXT.match(txt):
        return True

    return False

def residual_only(z, targets):
    return geometric_residual(z, targets)

def make_fallback_seeds(x_est0, y_est0, w0, h0, frx_swap, frz_swap, prev_z=None):
    seeds = []

    a0 = 180.0 if frx_swap else 0.0
    c0 = 180.0 if frz_swap else 0.0

    # main fallback family around the expected branch
    for da in (-15.0, 0.0, 15.0):
        for dc in (-15.0, 0.0, 15.0):
            seeds.append(np.array([
                x_est0 - 0.50 * w0,
                x_est0 + 0.50 * w0,
                y_est0 - 0.50 * h0,
                y_est0 + 0.50 * h0,
                a0 + da,
                0.0,
                c0 + dc,
                x_est0,
                y_est0
            ], dtype=float))

    # slightly different size/origin assumptions
    for scale in (0.45, 0.55, 0.65):
        seeds.append(np.array([
            x_est0 - scale * w0,
            x_est0 + scale * w0,
            y_est0 - scale * h0,
            y_est0 + scale * h0,
            a0,
            0.0,
            c0,
            x_est0,
            y_est0
        ], dtype=float))

    # also try around previous solution, if available
    if prev_z is not None:
        seeds.append(prev_z.copy())

        for da in (-12.0, 12.0):
            s = prev_z.copy()
            s[4] += da
            seeds.append(s)

        for dc in (-12.0, 12.0):
            s = prev_z.copy()
            s[6] += dc
            seeds.append(s)

    return seeds

def make_branch_seeds(x_est0, y_est0, w0, h0, frx_swap, frz_swap, prev_z=None):
    """
    Very small deterministic seed bank.
    This is the key speed fix versus the previous 2187-seed version.
    """
    seeds = []

    # 1) Previous-frame seed, if available
    if prev_z is not None:
        seeds.append(prev_z.copy())

    # 2) Main branch seed
    a0 = 180.0 if frx_swap else 0.0
    c0 = 180.0 if frz_swap else 0.0
    seeds.append(np.array([
        x_est0 - 0.5 * w0,
        x_est0 + 0.5 * w0,
        y_est0 - 0.5 * h0,
        y_est0 + 0.5 * h0,
        a0,
        0.0,
        c0,
        x_est0,
        y_est0
    ], dtype=float))

    # 3) Slight alternate branch-local perturbation
    seeds.append(np.array([
        x_est0 - 0.55 * w0,
        x_est0 + 0.55 * w0,
        y_est0 - 0.55 * h0,
        y_est0 + 0.55 * h0,
        a0 + (-8.0 if frx_swap else 8.0),
        0.0,
        c0 + (-8.0 if frz_swap else 8.0),
        x_est0,
        y_est0
    ], dtype=float))

    # 4) Another tiny perturbation
    seeds.append(np.array([
        x_est0 - 0.45 * w0,
        x_est0 + 0.45 * w0,
        y_est0 - 0.45 * h0,
        y_est0 + 0.45 * h0,
        a0 + (8.0 if frx_swap else -8.0),
        0.0,
        c0 + (8.0 if frz_swap else -8.0),
        x_est0,
        y_est0
    ], dtype=float))

    return seeds


def solve_frame(targets, prev_z, x_est0, y_est0, w0, h0, theta, frz):
    frx_swap = in_swap_region_x(theta)
    frz_swap = in_swap_region_z(frz)

    def evaluate_seed_list(seeds, max_nfev):
        best = None
        best_geom_cost = np.inf
        best_score = np.inf
        best_nfev = None
        total_nfev = 0

        lb = np.array([
            x_est0 - 3 * w0, x_est0 - 3 * w0,
            y_est0 - 3 * h0, y_est0 - 3 * h0,
            -1440.0, -89.9, -1440.0,
            -meta.play_res_x, -meta.play_res_y
        ], dtype=float)

        ub = np.array([
            x_est0 + 3 * w0, x_est0 + 3 * w0,
            y_est0 + 3 * h0, y_est0 + 3 * h0,
            1440.0, 89.9, 1440.0,
            2 * meta.play_res_x, 2 * meta.play_res_y
        ], dtype=float)

        for seed in seeds:
            try:
                res = least_squares(
                    residual_only,
                    x0=seed,
                    bounds=(lb, ub),
                    args=(targets,),
                    method="trf",
                    x_scale="jac",
                    ftol=1e-9,
                    xtol=1e-9,
                    gtol=1e-9,
                    max_nfev=max_nfev
                )
            except Exception:
                continue

            total_nfev += res.nfev

            z = res.x
            geom = geometric_residual(z, targets)
            geom_cost = float(np.linalg.norm(geom, ord=2))
            w1 = abs(z[1] - z[0])

            branch_ok = (
                angle_matches_branch(z[4], frx_swap) and
                angle_matches_branch(z[6], frz_swap) and
                not (90.0 < z[5] % 360.0 < 270.0)
            )

            feasible = (
                np.all(np.isfinite(z)) and
                branch_ok and
                (0.2 < w1 / max(w0, 1e-9) < 5.0) and
                geom_cost < 5 * residual_tol
            )

            if feasible:
                cont_cost = continuity_cost(z, prev_z, w0, h0)
                score = geom_cost + 1e-4 * cont_cost

                if score < best_score:
                    best = z.copy()
                    best_geom_cost = geom_cost
                    best_score = score
                    best_nfev = res.nfev

        return best, best_geom_cost, best_nfev, total_nfev

    # pass 1: fast tiny seed bank
    seeds1 = make_branch_seeds(
        x_est0=x_est0,
        y_est0=y_est0,
        w0=w0,
        h0=h0,
        frx_swap=frx_swap,
        frz_swap=frz_swap,
        prev_z=prev_z
    )

    best, best_geom_cost, best_nfev, total_nfev = evaluate_seed_list(
        seeds1, solver_max_nfev
    )

    if best is not None:
        return best, best_geom_cost, best_nfev, total_nfev

    # pass 2: fallback only when needed
    seeds2 = make_fallback_seeds(
        x_est0=x_est0,
        y_est0=y_est0,
        w0=w0,
        h0=h0,
        frx_swap=frx_swap,
        frz_swap=frz_swap,
        prev_z=prev_z
    )

    best2, cost2, nfev2, total2 = evaluate_seed_list(seeds2, max(180, solver_max_nfev))

    return best2, cost2, nfev2, total_nfev + total2


# ============================================================
# MAIN
# ============================================================

for l_ind, line in enumerate(lines):
    l = line.copy()
    str1 = line.raw_text

    pos_match = RE_POS.findall(str1)
    if not pos_match:
        continue
    pos_x = float(pos_match[0][0])
    pos_y = float(pos_match[0][1])

    frz_match = RE_FRZ.findall(str1)
    frz = float(frz_match[0]) if frz_match else 0.0

    u1 = u(frz + axis_offset)
    shape_flag = is_shape_line(line)

    # ========================================================
    # SHAPE BRANCH
    # ========================================================
    if (calculation_type != "original") or shape_flag:

        if shape_flag:
            try:
                shape1 = Shape(line.text)
            except Exception:
                shape1 = Convert.text_to_shape(line)
        else:
            shape1 = Convert.text_to_shape(line)

        x_left, y_top, x_right, y_bottom = shape1.bounding()
        shape_width = x_right - x_left
        shape_height = y_bottom - y_top

        shape1.move(-x_left, -y_top)
        shape1.move(-shape_width / 2, -shape_height / 2)

        shape1.map(lambda x, y: rotate_z(x, y, frz))

        FU = FrameUtility(line.start_time, line.end_time, meta.timestamps)

        for s, e, i, n in FU:
            l.start_time = s
            l.end_time = e

            shape2 = deepcopy(shape1)
            theta = Utils.interpolate(i / n, object_frx1, object_frx2, acc)
            shape2.map(lambda x, y: rotated_xy(x, y, theta, u1))

            str_out = "{\\pos(%.1f,%.1f)\\p1\\an7}%s" % (pos_x, pos_y, shape2)
            l.text = str_out
            Pyegi.send_line(l)

    # ========================================================
    # TEXT / RECTANGLE BRANCH
    # ========================================================
    else:
        w0 = line.width
        h0 = line.height

        if w0 <= 0 or h0 <= 0:
            continue

        base_corners = np.array([
            [-w0 / 2.0, -h0 / 2.0],
            [ w0 / 2.0, -h0 / 2.0],
            [ w0 / 2.0,  h0 / 2.0],
            [-w0 / 2.0,  h0 / 2.0]
        ], dtype=float)

        base_corners = rotate_z_points(base_corners, frz)

        FU = FrameUtility(line.start_time, line.end_time, meta.timestamps)

        z_prev = None

        for s, e, i, n in FU:
            l.start_time = s
            l.end_time = e

            theta = Utils.interpolate(i / n, object_frx1, object_frx2, acc)
            print(f"processing frame {i}/{n} (theta = {theta})...")
            frame_start_time = time.perf_counter()

            projected = project_points(base_corners, theta, u1)
            projected[:, 0] += line.width / 2.0 + line.left
            projected[:, 1] += line.height / 2.0 + line.top

            targets = projected.ravel()
            x_est0 = projected[:, 0].mean()
            y_est0 = projected[:, 1].mean()

            z, geom_cost, nfev, total_nfev = solve_frame(
                targets=targets,
                prev_z=z_prev,
                x_est0=x_est0,
                y_est0=y_est0,
                w0=w0,
                h0=h0,
                theta=theta,
                frz=frz
            )

            if z is None:
                raise RuntimeError(
                    f"Could not solve frame {i}/{n} "
                    f"(theta={theta:.6f}, line={l_ind}, text={line.text!r})"
                )

            z_prev = z.copy()

            w1 = abs(z[1] - z[0])
            h1 = abs(z[3] - z[2])
            pos_x = 0.5 * (z[0] + z[1]) + z[7]
            pos_y = 0.5 * (z[2] + z[3]) + z[8]

            frame_end_time = time.perf_counter()
            print(
                f"best_nfev={nfev}, total_nfev={total_nfev}, residual={geom_cost:.3e}, "
                f"elapsed={frame_end_time - frame_start_time:.3f}s"
            )

            str_out = (
                "{\\an5\\shad0\\bord%.3f\\frx%.3f\\fry%.3f\\frz%.3f"
                "\\pos(%.3f,%.3f)\\org(%.3f,%.3f)\\fscx%.3f\\fscy%.3f}%s"
            ) % (
                w1 / w0 * 3.0,
                z[4],
                z[5],
                z[6],
                pos_x,
                pos_y,
                z[7],
                z[8],
                w1 / w0 * 100.0,
                h1 / h0 * 100.0,
                line.text
            )

            l.text = str_out
            Pyegi.send_line(l)


Pyegi.create_output_file()

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regenerate aoi_ring_light.usda with per-LED tilt using natural aim angle atan2(r, z).

Each LED is rotated to aim precisely at (0, 0, 0).

Rotation math:
  USD rotateZYX float3 is ALWAYS (X_val, Y_val, Z_val) -- component order is XYZ!
  The name "ZYX" only determines composition: M = Rz(Z_val) * Ry(Y_val) * Rx(X_val)

  So: rotateZYX = (0, tilt, yaw) means Rx=0, Ry=tilt, Rz=yaw
      M = Rz(yaw) * Ry(tilt)

  DiskLight default direction = (0, 0, -1)
  Step 1: Ry(tilt) rotates (0,0,-1) -> (-sin(tilt), 0, -cos(tilt))
  Step 2: Rz(yaw)  rotates that   -> (-sin(tilt)*cos(yaw), -sin(tilt)*sin(yaw), -cos(tilt))

  Matching to target direction (-x,-y,-z)/dist gives:
      yaw  = atan2(y, x)
      tilt = atan2(r, z)
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

OUT = "INPUT_YOUR_OUTPUT_PATH/aoi_ring_light.usda"

# (group, color_rgb, radius, z, count, intensity, cone_angle, tilt_extra)
# tilt = atan2(r, z) + tilt_extra
# tilt_extra > 0 -> beam overshoots origin, hits further inward
#
# Bowl geometry: sphere center=(0,0,-16), R=71
# z(r) = -16 + sqrt(71^2 - r^2)  ->  gentle concave bowl shape
sub_rings: list[tuple[str, tuple[int, int, int], int, float, int, int, int, float]] = [
    # --- Red: near-vertical (tilt 0 deg, 10 deg), cone=120 deg ---
    ("Inner_Red", (1, 0, 0), 10, 54.3, 22, 5000, 120, -10.43),
    ("Inner_Red", (1, 0, 0), 16, 53.2, 35, 5000, 120, -6.74),
    # --- Green: tilt 20 deg~55 deg, cone=120 deg ---
    ("Middle_Green", (0, 1, 0), 22, 51.5, 49, 2000, 120, -3.13),
    ("Middle_Green", (0, 1, 0), 28, 49.2, 62, 2200, 120, 0.36),
    ("Middle_Green", (0, 1, 0), 34, 46.3, 75, 2400, 120, 3.71),
    ("Middle_Green", (0, 1, 0), 40, 42.7, 88, 2600, 120, 6.87),
    ("Middle_Green", (0, 1, 0), 43, 40.5, 95, 2800, 120, 8.28),
    # --- Blue: tilt 60 deg~89 deg, cone=120 deg/180 deg ---
    ("Outer_Blue", (0, 0, 1), 46, 38.1, 101, 3000, 120, 9.63),
    ("Outer_Blue", (0, 0, 1), 49, 35.4, 108, 3200, 120, 10.83),
    ("Outer_Blue", (0, 0, 1), 52, 32.3, 115, 3500, 120, 11.85),
    ("Outer_Blue", (0, 0, 1), 55, 28.9, 121, 3800, 120, 12.72),
    ("Outer_Blue", (0, 0, 1), 58, 25.0, 128, 4000, 180, 22.32),
]


def make_light(
    idx: int,
    color: tuple[int, int, int],
    intensity: int,
    tx: float,
    ty: float,
    tz: float,
    yaw_deg: float,
    tilt_deg: float,
    cone_angle: int,
    radius: float,
) -> str:
    r, g, b = color
    return f"""\
            def DiskLight "L_{idx:03d}" (
                prepend apiSchemas = ["ShapingAPI"]
            )
            {{
                color3f inputs:color = ({r}, {g}, {b})
                float inputs:exposure = 0
                float inputs:intensity = {intensity}
                float inputs:radius = {radius:.2f}
                float inputs:shaping:cone:angle = {cone_angle}
                float inputs:shaping:cone:softness = 1
                bool visibleInPrimaryRay = 0
                double3 xformOp:translate = ({tx:.6f}, {ty:.6f}, {tz})
                float3 xformOp:rotateZYX = (0, {tilt_deg:.2f}, {yaw_deg:.2f})
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZYX"]
            }}"""


def main() -> None:
    lines: list[str] = []
    lines.append("#usda 1.0")
    lines.append("(")
    lines.append('    defaultPrim = "AOI_RingLight"')
    lines.append("    metersPerUnit = 1")
    lines.append('    upAxis = "Z"')
    lines.append(")")
    lines.append("")
    lines.append('def Xform "AOI_RingLight"')
    lines.append("{")
    lines.append(
        '    custom string comment = "AOI Multi-Color Funnel Ring Light -- natural aim at origin"'
    )

    # Group sub_rings by group name
    group_rings: dict[str, list] = {}
    for sr in sub_rings:
        group = sr[0]
        if group not in group_rings:
            group_rings[group] = []
        group_rings[group].append(sr)

    for group_name in ["Inner_Red", "Middle_Green", "Outer_Blue"]:
        rings = group_rings[group_name]
        lines.append("")
        lines.append(f'    def Xform "{group_name}"')
        lines.append("    {")

        for ring_idx, (_, color, r, z, count, intensity, cone_angle, tilt_extra) in enumerate(
            rings
        ):
            ring_label = f"Ring_{ring_idx:02d}_r{r}"
            light_radius = r * 2 * math.pi / count * 0.45  # spacing-based radius
            natural_tilt = math.degrees(math.atan2(r, z))
            final_tilt = natural_tilt + tilt_extra
            extra_str = f" + {tilt_extra} deg extra" if tilt_extra else ""
            lines.append(
                f"        # tilt = {final_tilt:.1f} deg{extra_str} (natural = {natural_tilt:.1f} deg)"
            )
            lines.append(f'        def Xform "{ring_label}"')
            lines.append("        {")

            for i in range(count):
                angle = 2 * math.pi * i / count
                tx = r * math.cos(angle)
                ty = r * math.sin(angle)

                # Yaw: atan2(y, x) -- NO +180 deg needed!
                # rotateZYX applies Ry(tilt) first which flips -Z toward -X,
                # then Rz(yaw) rotates into the correct azimuth.
                yaw_deg = math.degrees(math.atan2(ty, tx))

                # Tilt: natural geometric angle + extra offset
                # natural = aim at (0,0,0), extra > 0 = overshoot past center
                tilt_deg = final_tilt

                led = make_light(
                    i, color, intensity, tx, ty, z, yaw_deg, tilt_deg, cone_angle, light_radius
                )
                lines.append(led)

            lines.append("        }")

        lines.append("    }")

    lines.append("}")
    lines.append("")

    with open(OUT, "w") as f:
        f.write("\n".join(lines))

    # Summary & verification
    total = sum(sr[4] for sr in sub_rings)
    logger.info("Generated %s with %d DiskLights", OUT, total)
    logger.info("")
    logger.info("Ring tilt summary:")
    for group, color, r, z, count, intensity, cone, t_extra in sub_rings:
        natural = math.degrees(math.atan2(r, z))
        final = natural + t_extra
        extra_str = f" (+{t_extra} deg)" if t_extra else ""
        logger.info(
            "  %15s r=%2d z=%4.1f -> tilt=%5.1f deg%7s (natural=%5.1f deg) cone=%d deg",
            group,
            r,
            z,
            final,
            extra_str,
            natural,
            cone,
        )

    # Mathematical verification: check multiple LEDs at different azimuthal angles
    # rotateZYX = (rx, ry, rz) in USD: M = Rz(rz) * Ry(ry) * Rx(rx)
    logger.info("")
    logger.info("Verification (rotated direction vs expected direction to origin):")
    test_angles = [0, 90, 180, 270]  # test all quadrants
    for group, color, r, z, count, intensity, cone, t_extra in [
        sub_rings[0],
        sub_rings[4],
        sub_rings[8],
    ]:
        tilt_rad = math.atan2(r, z)
        for az in test_angles:
            az_rad = math.radians(az)
            tx = r * math.cos(az_rad)
            ty = r * math.sin(az_rad)
            yaw_rad = math.atan2(ty, tx)
            # rotateZYX = (0, tilt, yaw) -> M = Rz(yaw) * Ry(tilt) * Rx(0)
            dx = -math.sin(tilt_rad) * math.cos(yaw_rad)
            dy = -math.sin(tilt_rad) * math.sin(yaw_rad)
            dz = -math.cos(tilt_rad)
            # Expected: from (tx,ty,z) toward (0,0,0)
            dist = math.sqrt(tx**2 + ty**2 + z**2)
            ex, ey, ez = -tx / dist, -ty / dist, -z / dist
            err = math.sqrt((dx - ex) ** 2 + (dy - ey) ** 2 + (dz - ez) ** 2)
            status = "OK" if err < 1e-6 else f"ERROR={err:.6f}"
            logger.info(
                "  %15s r=%2d az=%3d deg pos=(%+7.2f,%+7.2f,%.1f) %s",
                group,
                r,
                az,
                tx,
                ty,
                z,
                status,
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()

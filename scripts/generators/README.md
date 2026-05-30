# AOI Ring Light Design Document

## Overview

AOI (Automated Optical Inspection) ring light simulation for solder joint inspection.
The light source consists of 12 rings of DiskLights, totaling 999 LEDs, arranged on a
bowl-shaped spherical surface. Three colors correspond to different incidence angles,
allowing a top-down orthographic camera to distinguish surface tilt angles via specular reflection.

**Generator script:** `scripts/generators/gen_ring_light.py`
**Output file:** `assets/aoi_ring_light.usda`

---

## AOI Color Principle

Real AOI equipment uses multi-color ring lights to illuminate solder joints from different angles.
An overhead orthographic camera observes specular reflections, where different surface tilt angles
reflect different colors of light:

- **Red:** Near-vertical illumination → reflects from flat surfaces (pads, component tops)
- **Green:** Medium incidence → reflects from moderate slopes (mid-section of solder fillets)
- **Blue:** High-angle grazing → reflects from steep surfaces (solder fillet walls)

### Specular Reflection Formula

For a top-down orthographic camera (viewing along -Z):

```
Surface tilt angle α → Required light source tilt = 2α
```

**Physical limitation:** Surface angles > 45° require tilt > 90° (light coming from below), which is impossible.

---

## Bowl Geometry

LEDs are arranged on a spherical arc, forming a bowl shape:

```
Sphere parameters:
  Center: (0, 0, -16)
  Radius: R = 71
  z(r) = -16 + sqrt(71² - r²)
```

Constraints for choosing this sphere:
- z(0) ≈ 55 (bowl top height)
- z(58) = 25 (outermost ring should not be too low)

### Why a Sphere Instead of a Cone?

A spherical arc provides more uniform inter-layer spacing and more closely resembles
the bowl shape of real AOI fixtures. Compared to a linear descent (funnel shape),
the sphere has a gentler z drop-off at the outer rings.

---

## 12-Ring Configuration

### Ring Layout

| # | Group | Radius | Z | Tilt | Natural | Extra | Count | Cone | Intensity |
|---|-------|--------|------|------|---------|-------|-------|------|-----------|
| 1 | Inner_Red | 10 | 54.3 | 0° | 10.4° | -10.43° | 22 | 120° | 5000 |
| 2 | Inner_Red | 16 | 53.2 | 10° | 16.7° | -6.74° | 35 | 120° | 5000 |
| 3 | Middle_Green | 22 | 51.5 | 20° | 23.1° | -3.13° | 49 | 120° | 2000 |
| 4 | Middle_Green | 28 | 49.2 | 30° | 29.6° | +0.36° | 62 | 120° | 2200 |
| 5 | Middle_Green | 34 | 46.3 | 40° | 36.3° | +3.71° | 75 | 120° | 2400 |
| 6 | Middle_Green | 40 | 42.7 | 50° | 43.1° | +6.87° | 88 | 120° | 2600 |
| 7 | Middle_Green | 43 | 40.5 | 55° | 46.7° | +8.28° | 95 | 120° | 2800 |
| 8 | Outer_Blue | 46 | 38.1 | 60° | 50.4° | +9.63° | 101 | 120° | 3000 |
| 9 | Outer_Blue | 49 | 35.4 | 65° | 54.2° | +10.83° | 108 | 120° | 3200 |
| 10 | Outer_Blue | 52 | 32.3 | 70° | 58.2° | +11.85° | 115 | 120° | 3500 |
| 11 | Outer_Blue | 55 | 28.9 | 75° | 62.3° | +12.72° | 121 | 120° | 3800 |
| 12 | Outer_Blue | 58 | 25.0 | 89° | 66.7° | +22.32° | 128 | 180° | 4000 |

**Total: 999 DiskLights**

### Tilt Coverage Range

```
0°  10°  20°  30°  40°  50°  55°  60°  65°  70°  75°  89°
R    R    G    G    G    G    G    B    B    B    B    B
```

Coverage in 5°–10° increments, from vertical (0°) to near-horizontal (89°).

---

## Rotation Calculation

### USD rotateZYX Semantics

```
float3 xformOp:rotateZYX = (X_val, Y_val, Z_val)
```

**Note:** Parameter order is always (X, Y, Z)! The name "ZYX" only determines the matrix composition order:

```
M = Rz(Z_val) * Ry(Y_val) * Rx(X_val)
```

### Per-LED Rotation

```
rotateZYX = (0, tilt, yaw)
```

Where:
- **yaw** = `atan2(y, x)` — LED azimuth angle, pointing toward center
- **tilt** = `atan2(r, z) + tilt_extra` — tilt angle

### Derivation

DiskLight default emission direction = (0, 0, -1)

1. `Ry(tilt)` rotation: (0, 0, -1) → (-sin(tilt), 0, -cos(tilt))
2. `Rz(yaw)` rotation: → (-sin(tilt)cos(yaw), -sin(tilt)sin(yaw), -cos(tilt))

Matching target direction (-x, -y, -z)/dist yields:
- `yaw = atan2(y, x)` (no +180° needed)
- `tilt = atan2(r, z)` (natural aiming angle, directly pointing at origin)

### Effect of tilt_extra

- `tilt_extra = 0`: LED aims precisely at origin (0, 0, 0)
- `tilt_extra > 0`: Beam overshoots origin, hitting farther away (more grazing)
- `tilt_extra < 0`: Beam falls short of origin (more vertical)

Natural tilt is the geometric angle that aims directly at the origin.
To achieve the target incidence angle, a tilt_extra offset is added:
`final_tilt = natural_tilt + tilt_extra`

---

## DiskLight Parameters

USD attributes used for each LED:

| Attribute | Description |
|-----------|-------------|
| `inputs:color` | RGB color (red/green/blue) |
| `inputs:intensity` | Light intensity, higher for outer rings to compensate for distance |
| `inputs:exposure` | Exposure compensation = 1 (2x brightness) |
| `inputs:radius` | Physical disk size, calculated from inter-ring spacing |
| `inputs:shaping:cone:angle` | Cone half-angle (120° or 180°) |
| `inputs:shaping:cone:softness` | Edge softness = 1 (full gradient) |
| `visibleInPrimaryRay` | = 0 (light not directly visible to camera) |

### Cone Angle Design

- **Red/Green (120°):** Wide spread, simulating LED Lambertian emission
- **Blue inner rings (120°):** Same as above
- **Blue outermost ring r=58 (180°):** Full spherical spread, because tilt=89° is near-horizontal,
  requiring maximum spread angle to effectively illuminate the target area

### LED Count Calculation

LED count per ring is based on dense packing:

```
count = round(2π × radius / spacing)
spacing ≈ 2.85 (unit spacing)
```

### LED Disk Radius

```
light_radius = radius × 2π / count × 0.45
```

Based on 45% of the inter-LED spacing to avoid overlap between adjacent LEDs.

---

## Surface Angle vs Color Reference Table

| Surface Angle α | Required Tilt (2α) | Reflected Color | AOI Meaning |
|-----------------|---------------------|-----------------|-------------|
| 0° | 0° | Red | Flat pad |
| 5° | 10° | Red | Slight tilt |
| 10°–27° | 20°–55° | Green | Mid-section fillet slope |
| 30°–44° | 60°–89° | Blue | Steep fillet wall |
| > 45° | > 90° | Dark (impossible) | Physical limitation |

---

## Test Geometry

### Solder Fillet Ramp (`assets/solder_fillet_ramp.usda`)

Unidirectional ramp with power curve: `z = 1.5 × (x/3.0)^1.5`

- Dimensions: 3.0 × 3.0 cm, height 1.5 cm
- Maximum surface angle: 36.9°
- metersPerUnit = 0.01

### Solder Fillet Ramp Reversed (`assets/solder_fillet_ramp_reversed.usda`)

Mirror image of the above ramp, with the steep end on the left and flat end on the right.

Placing both side by side forms a symmetric solder joint, simulating:
`Flat pad → fillet ramp → component body → fillet ramp → flat pad`

---

## Bowl Shape ASCII Diagram

```
Side view (r→, z↑):

z=54  * *                    ← Red (r=10,16)
z=51    * *                  ← Green inner (r=22,28)
z=46       * *               ← Green middle (r=34,40)
z=40         *               ← Green outer (r=43)
z=38          *              ← Blue 1 (r=46)
z=35           *             ← Blue 2 (r=49)
z=32            *            ← Blue 3 (r=52)
z=29             *           ← Blue 4 (r=55)
z=25              *          ← Blue 5 (r=58)
       |----|----|----|----|
       r=10  r=22  r=40  r=58


Top view (AOI camera):

         RRRRR
       GGGGGGGGG
      GGGGGGGGGGG
     BBBBBBBBBBBBB
    BBBBBBBBBBBBBBB
     BBBBBBBBBBBBB
      GGGGGGGGGGG
       GGGGGGGGG
         RRRRR
```

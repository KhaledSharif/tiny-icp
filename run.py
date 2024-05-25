import numpy as np
import math

# axis sequences for Euler angles
NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
AXIS_MAP = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}


def euler_matrix(ai, aj, ak, axes='sxyz'):
    first_axis, parity, repetition, frame = AXIS_MAP[axes]

    i = first_axis
    j = NEXT_AXIS[i + parity]
    k = NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    m = np.identity(4)
    if repetition:
        m[i, i] = cj
        m[i, j] = sj * si
        m[i, k] = sj * ci
        m[j, i] = sj * sk
        m[j, j] = -cj * ss + cc
        m[j, k] = -cj * cs - sc
        m[k, i] = -sj * ck
        m[k, j] = cj * sc + cs
        m[k, k] = cj * cc - ss
    else:
        m[i, i] = cj * ck
        m[i, j] = sj * sc - cs
        m[i, k] = sj * cc + ss
        m[j, i] = cj * sk
        m[j, j] = sj * ss + cc
        m[j, k] = sj * cs - sc
        m[k, i] = -sj
        m[k, j] = cj * si
        m[k, k] = cj * ci

    return m


def icp_point_to_plane(source_points, dest_points):
    """
    Point to plane matching using least squares

    source_points:  nx3 matrix of n 3D points
    dest_points: nx6 matrix of n 3D points + 3 normal vectors
    """

    a = []
    b = []

    for i in range(0, dest_points.shape[0] - 1):
        dx = dest_points[i][0]
        dy = dest_points[i][1]
        dz = dest_points[i][2]
        nx = dest_points[i][3]
        ny = dest_points[i][4]
        nz = dest_points[i][5]

        sx = source_points[i][0]
        sy = source_points[i][1]
        sz = source_points[i][2]

        _a1 = (nz * sy) - (ny * sz)
        _a2 = (nx * sz) - (nz * sx)
        _a3 = (ny * sx) - (nx * sy)

        _a = np.array([_a1, _a2, _a3, nx, ny, nz])

        _b = (nx * dx) + (ny * dy) + (nz * dz) - (nx * sx) - (ny * sy) - (nz * sz)

        a.append(_a)
        b.append(_b)

    a_inv = np.linalg.pinv(np.array(a))
    tr = np.dot(a_inv, b)

    r = euler_matrix(tr[0], tr[1], tr[2])
    r[0, 3] = tr[3]
    r[1, 3] = tr[4]
    r[2, 3] = tr[5]

    source_transformed = []

    for i in range(0, dest_points.shape[0] - 1):
        ss = np.array([(source_points[i][0]), (source_points[i][1]), (source_points[i][2]), (1)])
        p = np.dot(r, ss)
        source_transformed.append(p)

    return np.array(source_transformed)

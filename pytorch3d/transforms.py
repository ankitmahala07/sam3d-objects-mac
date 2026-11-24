# Minimal pure-PyTorch reimplementation of the pytorch3d.transforms subset
# used by sam3d-objects (alt3d shim, 2026-11-21). Follows pytorch3d
# conventions as exercised by this codebase: stored 4x4 matrices are in
# row-vector form (translation in the bottom row), chained ops apply in
# call order, transform_points(p) = p @ A + t with A = M[:3, :3],
# t = M[3, :3]. Quaternions are (w, x, y, z).
import torch


def _as_batched_matrix(matrix):
    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0)
    return matrix


class Transform3d:
    def __init__(self, device="cpu", dtype=torch.float32, matrix=None):
        if matrix is None:
            self._matrix = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)
        else:
            m = _as_batched_matrix(matrix.clone())
            self._matrix = m
        self._device = self._matrix.device
        self._dtype = self._matrix.dtype

    # -- composition helpers -------------------------------------------------
    def _op(self, op_matrix):
        op = _as_batched_matrix(op_matrix.to(dtype=self._dtype, device=self._device))
        new = Transform3d(matrix=torch.matmul(self._matrix, op))
        return new

    def compose(self, *others):
        result = Transform3d(matrix=self._matrix.clone())
        for other in others:
            result._matrix = torch.matmul(result._matrix, other.get_matrix())
        return result

    def rotate(self, rotations):
        R = rotations.to(dtype=self._dtype, device=self._device)
        if R.dim() == 2:
            R = R.unsqueeze(0)
        op = torch.eye(4, dtype=self._dtype, device=self._device).repeat(R.shape[0], 1, 1)
        op[:, :3, :3] = R
        return self._op(op)

    def translate(self, translations):
        t = translations.to(dtype=self._dtype, device=self._device)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        op = torch.eye(4, dtype=self._dtype, device=self._device).repeat(t.shape[0], 1, 1)
        op[:, 3, :3] = t
        return self._op(op)

    def scale(self, scales):
        s = scales.to(dtype=self._dtype, device=self._device)
        if s.dim() == 0:
            s = s.reshape(1, 1).expand(1, 3)
        elif s.dim() == 1:
            s = s.unsqueeze(0)
        op = torch.eye(4, dtype=self._dtype, device=self._device).repeat(s.shape[0], 1, 1)
        op[:, 0, 0] = s[:, 0]
        op[:, 1, 1] = s[:, 1]
        op[:, 2, 2] = s[:, 2]
        return self._op(op)

    # -- application ---------------------------------------------------------
    def transform_points(self, points, eps=None):
        M = self._matrix
        A = M[:, :3, :3]
        t = M[:, 3, :3]
        orig_dtype = points.dtype
        pts = points.to(M.dtype)
        if pts.dim() == 2:
            out = torch.matmul(pts.unsqueeze(0), A) + t.unsqueeze(1)
            out = out[0] if M.shape[0] == 1 else out
            return out.to(orig_dtype)
        return (torch.matmul(pts, A) + t.unsqueeze(1)).to(orig_dtype)

    def transform_normals(self, normals):
        M = self._matrix
        A = M[:, :3, :3]
        # inverse-transpose for row-vector convention: n @ A^{-1}... use
        # n transformed by the inverse of A transposed appropriately:
        Ainv = torch.inverse(A)
        orig_dtype = normals.dtype
        n = normals.to(M.dtype)
        if n.dim() == 2:
            out = torch.matmul(n.unsqueeze(0), Ainv)
            out = out[0] if M.shape[0] == 1 else out
            return out.to(orig_dtype)
        return torch.matmul(n, Ainv).to(orig_dtype)

    def inverse(self):
        return Transform3d(matrix=torch.inverse(self._matrix))

    def get_matrix(self):
        return self._matrix

    def clone(self):
        return Transform3d(matrix=self._matrix.clone())

    def to(self, device):
        self._matrix = self._matrix.to(device)
        self._device = self._matrix.device
        return self

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype


class Rotate(Transform3d):
    def __init__(self, R, device="cpu", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self._matrix = self.rotate(R)._matrix


class Translate(Transform3d):
    def __init__(self, T, device="cpu", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self._matrix = self.translate(T)._matrix


class Scale(Transform3d):
    def __init__(self, S, device="cpu", dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self._matrix = self.scale(S)._matrix


# -- quaternions (w, x, y, z), pytorch3d formulas ----------------------------

def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix):
    """Shepperd-style robust conversion, returns (w, x, y, z)."""
    m = matrix
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    trace = m00 + m11 + m22
    one = torch.ones_like(trace)

    def safe_sqrt(x):
        return torch.sqrt(torch.clamp(x, min=0.0))

    # Four candidate quaternions (w, x, y, z), pick largest norm component.
    s_w = safe_sqrt(1.0 + trace)
    q_w = torch.stack(
        [0.5 * s_w, (m21 - m12) / (2 * s_w + 1e-12), (m02 - m20) / (2 * s_w + 1e-12),
         (m10 - m01) / (2 * s_w + 1e-12)], -1)
    s_x = safe_sqrt(1.0 + m00 - m11 - m22)
    q_x = torch.stack(
        [(m21 - m12) / (2 * s_x + 1e-12), 0.5 * s_x, (m01 + m10) / (2 * s_x + 1e-12),
         (m02 + m20) / (2 * s_x + 1e-12)], -1)
    s_y = safe_sqrt(1.0 + m11 - m00 - m22)
    q_y = torch.stack(
        [(m02 - m20) / (2 * s_y + 1e-12), (m01 + m10) / (2 * s_y + 1e-12), 0.5 * s_y,
         (m12 + m21) / (2 * s_y + 1e-12)], -1)
    s_z = safe_sqrt(1.0 + m22 - m00 - m11)
    q_z = torch.stack(
        [(m10 - m01) / (2 * s_z + 1e-12), (m02 + m20) / (2 * s_z + 1e-12),
         (m12 + m21) / (2 * s_z + 1e-12), 0.5 * s_z], -1)
    candidates = torch.stack([q_w, q_x, q_y, q_z], -2)  # (..., 4, 4)
    norms = candidates.norm(dim=-1)
    best = norms.argmax(dim=-1, keepdim=True)
    q = torch.gather(candidates, -2, best.unsqueeze(-1).expand(*norms.shape[:-1], 1, 4)).squeeze(-2)
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    # canonicalize: w >= 0
    sign = torch.where(q[..., 0] < 0, -one, one)
    return q * sign.unsqueeze(-1)


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = torch.unbind(q1, -1)
    w2, x2, y2, z2 = torch.unbind(q2, -1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
            w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
        ],
        -1,
    )


def quaternion_invert(quaternions):
    conj = quaternions * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=quaternions.dtype, device=quaternions.device)
    return conj / (quaternions * quaternions).sum(-1, keepdim=True)


def axis_angle_to_quaternion(axis_angle):
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half = angles * 0.5
    eps = 1e-6
    small = angles < eps
    scale = torch.where(small, 0.5 - angles ** 2 / 48.0, torch.sin(half) / (angles + 1e-12))
    quat = torch.cat([torch.cos(half), axis_angle * scale], dim=-1)
    return quat


def axis_angle_to_matrix(axis_angle):
    """Axis-angle (..., 3) -> rotation matrix (..., 3, 3).

    GenIA port addition: composed from this module's own
    axis_angle_to_quaternion + quaternion_to_matrix (identical math to
    pytorch3d's direct Rodrigues implementation).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

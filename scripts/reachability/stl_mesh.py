"""Minimal numpy-only STL utilities (no trimesh dependency).

Provides binary/ASCII STL loading, mesh-connectivity components, and
vectorised ray casting against the triangles, used to locate the bracket's
holes and inside corner.
"""
from pathlib import Path
import struct

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


class TriMesh:
    def __init__(self, triangles: np.ndarray):
        """triangles: (n, 3, 3) array of vertex coordinates."""
        self.tri = np.asarray(triangles, dtype=float)
        e1 = self.tri[:, 1] - self.tri[:, 0]
        e2 = self.tri[:, 2] - self.tri[:, 0]
        n = np.cross(e1, e2)
        self.area = np.linalg.norm(n, axis=1) / 2.0
        self.normals = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-15)
        self.centroids = self.tri.mean(axis=1)
        self._e1, self._e2 = e1, e2

    @classmethod
    def from_stl(cls, path, scale: float = 1.0) -> "TriMesh":
        data = Path(path).read_bytes()
        if len(data) >= 84:
            n = struct.unpack("<I", data[80:84])[0]
            if 84 + 50 * n == len(data):
                rec = np.frombuffer(data[84:], dtype=np.dtype(
                    [("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")]))
                return cls(rec["v"].astype(float) * scale)
        # ASCII fallback
        verts = [list(map(float, line.split()[1:4]))
                 for line in data.decode(errors="ignore").splitlines()
                 if line.strip().startswith("vertex")]
        return cls(np.array(verts).reshape(-1, 3, 3) * scale)

    @property
    def bounds(self):
        v = self.tri.reshape(-1, 3)
        return v.min(axis=0), v.max(axis=0)

    def components(self, mask: np.ndarray, weld_decimals: int = 4) -> np.ndarray:
        """Label connected components (shared welded vertices) among the
        triangles selected by mask. Returns labels for those triangles."""
        idx = np.flatnonzero(mask)
        keys = np.round(self.tri[idx].reshape(-1, 3), weld_decimals)
        _, vid = np.unique(keys, axis=0, return_inverse=True)
        vid = vid.ravel()
        t = np.repeat(np.arange(len(idx)), 3)
        size = len(idx) + vid.max() + 1
        graph = coo_matrix((np.ones(len(t)), (t, vid + len(idx))), shape=(size, size))
        _, labels = connected_components(graph, directed=False)
        return labels[:len(idx)]

    def ray_hits(self, origins: np.ndarray, dirs: np.ndarray, eps: float = 1e-12):
        """Möller-Trumbore for many rays vs all triangles.

        origins, dirs: (m, 3). Returns (m,) distance along each dir (in units
        of |dir|) to the nearest hit with t > 0, or inf if none.
        """
        origins = np.atleast_2d(origins).astype(float)
        dirs = np.atleast_2d(dirs).astype(float)
        out = np.full(len(origins), np.inf)
        v0, e1, e2 = self.tri[:, 0], self._e1, self._e2
        # chunk rays to bound memory (m_chunk x n_tri)
        chunk = max(1, int(2e7 // max(1, len(v0))))
        for s in range(0, len(origins), chunk):
            o = origins[s:s + chunk, None, :]
            d = dirs[s:s + chunk, None, :]
            p = np.cross(d, e2[None])
            det = np.einsum("mnk,nk->mn", p, e1)
            ok = np.abs(det) > eps
            inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
            tv = o - v0[None]
            u = np.einsum("mnk,mnk->mn", tv, p) * inv
            q = np.cross(tv, e1[None])
            v = np.einsum("mnk,mnk->mn", d, q) * inv
            t = np.einsum("nk,mnk->mn", e2, q) * inv
            hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > eps)
            out[s:s + chunk] = np.where(hit, t, np.inf).min(axis=1)
        return out

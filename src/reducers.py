"""PCA and Wang et al. (2025), Algorithm 1, for row-wise image arrays."""

import numpy as np
from scipy.linalg import eigh, solve


def grid_laplacian(height, width):
    """Unweighted four-neighbor pixel graph in row-major order."""
    d = height * width
    lap = np.zeros((d, d), dtype=np.float64)
    for row in range(height):
        for col in range(width):
            a = row * width + col
            for r, c in ((row + 1, col), (row, col + 1)):
                if r < height and c < width:
                    b = r * width + c
                    lap[a, a] += 1
                    lap[b, b] += 1
                    lap[a, b] -= 1
                    lap[b, a] -= 1
    return lap


def principal_axes(centered, rank):
    """Eigenvectors of the scatter matrix in descending eigenvalue order."""
    d = centered.shape[1]
    if not 1 <= rank <= min(centered.shape):
        raise ValueError("Rank is outside the matrix dimensions")
    _, vectors = eigh(centered.T @ centered, subset_by_index=(d - rank, d - 1))
    return vectors[:, ::-1].copy()


def mm_direction(w, v, lap, eta1, eta2):
    """Unnormalized update. With eta1 > 0, solve the symmetric form of Eq. (32)."""
    if eta1 == 0:
        return v if eta2 == 0 else solve(
            np.eye(len(w)) + eta2 * lap, v, assume_a="pos"
        )

    magnitude = np.abs(w)
    root = np.sqrt(magnitude)
    matrix = eta2 * (root[:, None] * lap * root[None, :])
    matrix.flat[:: len(w) + 1] += magnitude + eta1
    return root * solve(matrix, root * v, assume_a="pos")


def orient(w):
    """Choose the display sign using the largest absolute coefficient."""
    return -w if w[np.argmax(np.abs(w))] < 0 else w


def robust_axes(centered, initial, lap, eta1, eta2, tolerance=1e-4, max_steps=100):
    """Extract successive directions with the paper's L1 stopping and deflation."""
    if min(eta1, eta2) < 0:
        raise ValueError("Penalty weights must be nonnegative")
    residual = centered.copy()
    w_all = np.empty_like(initial)
    iterations, final_changes = [], []
    for j, init in enumerate(initial.T):
        w = init.copy()
        scores = residual @ w
        score = np.abs(scores).sum()
        for step in range(1, max_steps + 1):
            v = residual.T @ np.sign(scores)
            u = mm_direction(w, v, lap, eta1, eta2)
            norm = np.linalg.norm(u)
            if norm <= 1e-14 or not np.isfinite(norm):
                raise FloatingPointError("Degenerate RSSPCA update")
            w = u / norm
            scores = residual @ w
            next_score = np.abs(scores).sum()
            relative_change = abs(next_score - score) / max(score, 1e-14)
            score = next_score
            if relative_change <= tolerance:
                break
        w_all[:, j] = orient(w)
        iterations.append(step)
        final_changes.append(float(relative_change))
        fitted = w_all[:, :j + 1]
        residual -= (residual @ fitted) @ fitted.T
    if not np.allclose(np.linalg.norm(w_all, axis=0), 1.0, atol=1e-10):
        raise FloatingPointError("Non-unit loadings")
    return w_all, iterations, final_changes

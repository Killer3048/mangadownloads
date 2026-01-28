import cv2
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image


class SmartSlicer:
    def __init__(
        self,
        edge_weight: float = 1.0,
        variance_weight: float = 0.5,
        gradient_weight: float = 0.5,
        white_space_weight: float = 2.0,
        distance_penalty: float = 0.0001,
        forbidden_cost: float = 1e9,
    ):
        self.edge_weight = edge_weight
        self.variance_weight = variance_weight
        self.gradient_weight = gradient_weight
        self.white_space_weight = white_space_weight
        self.distance_penalty = distance_penalty
        self.forbidden_cost = forbidden_cost

    def _compute_edge_density(self, img_gray: np.ndarray) -> np.ndarray:
        """Computes row-wise edge density using Canny."""
        edges = cv2.Canny(img_gray, 50, 150)
        row_density = np.sum(edges, axis=1) / 255.0
        width = img_gray.shape[1]
        if width <= 0:
            return row_density * 0
        return row_density / width

    def _compute_row_variance(self, img_gray: np.ndarray) -> np.ndarray:
        """Computes variance of pixels in each row."""
        return np.var(img_gray, axis=1)

    def _compute_vertical_gradient(self, img_gray: np.ndarray) -> np.ndarray:
        """Computes the difference between row i and row i-1."""
        grad = np.abs(np.diff(img_gray.astype(np.float32), axis=0))
        row_grad = np.sum(grad, axis=1)
        row_grad = np.insert(row_grad, 0, 0)
        width = img_gray.shape[1]
        if width <= 0:
            return row_grad * 0
        return row_grad / width

    def _compute_white_space_score(self, img_gray: np.ndarray) -> np.ndarray:
        """
        Computes a score for 'white space' or solid light color.
        High brightness + Low variance = High White Space Score.
        """
        mean_brightness = np.mean(img_gray, axis=1)
        variance = np.var(img_gray, axis=1)

        norm_brightness = mean_brightness / 255.0
        inv_variance = 1.0 / (1.0 + variance)

        return norm_brightness * inv_variance

    def compute_cost_map(
        self,
        image: Image.Image,
        boxes: List[Tuple[float, float, float, float, float, int]],
        margin: int = 10
    ) -> np.ndarray:
        """
        Computes a cost score for every row in the image.
        Lower cost = better place to cut.
        """
        img_np = np.array(image.convert("RGB"))
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        height, width = img_gray.shape

        edge_score = self._compute_edge_density(img_gray)
        variance_score = self._compute_row_variance(img_gray)
        gradient_score = self._compute_vertical_gradient(img_gray)
        white_space_score = self._compute_white_space_score(img_gray)

        def normalize(arr: np.ndarray) -> np.ndarray:
            if arr.size == 0:
                return arr
            mn, mx = float(arr.min()), float(arr.max())
            if mx - mn < 1e-6:
                return np.zeros_like(arr, dtype=np.float32)
            return (arr - mn) / (mx - mn)

        edge_score = normalize(edge_score)
        variance_score = normalize(variance_score)
        gradient_score = normalize(gradient_score)
        white_space_score = normalize(white_space_score)

        total_cost = (
            self.edge_weight * edge_score +
            self.variance_weight * variance_score +
            self.gradient_weight * gradient_score -
            self.white_space_weight * white_space_score
        )

        # Apply forbidden zones
        mask = np.zeros(height, dtype=bool)
        for (x1, y1, x2, y2, conf, cls_id) in boxes:
            y_start = max(0, int(y1) - margin)
            y_end = min(height, int(y2) + margin)
            if y_end > y_start:
                mask[y_start:y_end] = True

        total_cost = total_cost.astype(np.float32, copy=False)
        total_cost[mask] += self.forbidden_cost

        return total_cost

    def find_best_cut(
        self,
        cost_map: np.ndarray,
        current_y: int,
        min_h: int,
        max_h: int,
        preferred_h: Optional[int] = None
    ) -> int:
        """
        Finds the row with the minimum cost within the range
        [current_y + min_h, current_y + max_h].
        """
        height = len(cost_map)
        start = current_y + min_h
        end = min(current_y + max_h, height)

        if start >= height:
            return height

        if start >= end:
            return min(start, height)

        segment_costs = cost_map[start:end].copy()

        safe_mask = segment_costs < (self.forbidden_cost / 2)

        if np.any(safe_mask):
            segment_costs[~safe_mask] = np.inf

        if preferred_h:
            target_y = current_y + preferred_h
            indices = np.arange(start, end)
            dist = np.abs(indices - target_y).astype(np.float32)
            segment_costs = segment_costs + dist * float(self.distance_penalty)

        best_local_idx = int(np.argmin(segment_costs))
        best_global_y = start + best_local_idx

        if best_global_y < 0:
            best_global_y = 0
        if best_global_y > height:
            best_global_y = height

        if best_global_y < height and cost_map[best_global_y] >= (self.forbidden_cost / 2):
            print(
                f"[WARN] No safe cut found between {start} and {end}. "
                f"Cutting at minimum cost (likely inside an object)."
            )

        return int(best_global_y)

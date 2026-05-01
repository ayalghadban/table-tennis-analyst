"""
Player Analyzer
Tracks ball trajectories and player posture frame-by-frame,
classifies each shot, and returns per-stage statistics.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np


# ── Shot dataclass ─────────────────────────────────────────────────────────────

@dataclass
class Shot:
    """Represents a single detected shot."""
    frame_idx : int
    result    : str    # "correct" | "net" | "miss" | "out"
    speed     : float  # px / frame
    direction : str    # "right→left" | "left→right" | "straight" | "unknown"
    ball_pos  : tuple  # (x, y) in original resolution


# ── PlayerAnalyzer ─────────────────────────────────────────────────────────────

class PlayerAnalyzer:
    """
    Accumulates per-frame TTNet outputs and produces per-stage statistics.

    Parameters
    ----------
    stage_config : dict
        One entry from STAGES (see stage_manager.py).
    bounce_thresh : float
        Minimum bounce_prob to register a shot event.
    net_thresh : float
        Minimum net_prob to classify a shot as a net-hit.
    scene_width : int
        Video width in pixels (for left/right zone checks). Default 1920.
    """

    def __init__(self, stage_config: dict,
                 bounce_thresh: float = 0.25,
                 net_thresh: float    = 0.30,
                 scene_width: int     = 1920):

        self.stage_config   = stage_config
        self.bounce_thresh  = bounce_thresh
        self.net_thresh     = net_thresh
        self.scene_width    = scene_width

        # Ball tracking
        self._ball_history: List[tuple]  = []   # [(frame_idx, x, y)]
        self._shots       : List[Shot]   = []

        # Posture tracking – centroid of left-player segmentation over time
        self._posture_x: List[float] = []
        self._posture_y: List[float] = []

        # Cooldown to avoid double-counting the same bounce
        self._last_shot_frame: int = -20

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(self,
               ball_pos    : list,
               bounce_prob : float,
               net_prob    : float,
               seg_mask    : np.ndarray,
               frame_idx   : int) -> Optional[Shot]:
        """
        Feed one frame of model output.

        Returns a Shot object if a shot was detected in this frame, else None.
        """
        x, y = int(ball_pos[0]), int(ball_pos[1])
        self._ball_history.append((frame_idx, x, y))

        # Track left-player posture centroid
        self._update_posture(seg_mask)

        # Detect shot on bounce event (with cooldown)
        shot = None
        cooldown = max(6, self.stage_config.get("num_balls", 10))
        if (bounce_prob >= self.bounce_thresh and
                frame_idx - self._last_shot_frame > cooldown):

            shot = self._classify_shot(frame_idx, x, y, net_prob)
            self._shots.append(shot)
            self._last_shot_frame = frame_idx

        return shot

    def get_stage_stats(self) -> dict:
        """Return a statistics dictionary compatible with StageManager."""
        shots = self._shots
        total = len(shots)

        correct_shots       = sum(1 for s in shots if s.result in ("correct", "out"))
        fully_correct_shots = sum(1 for s in shots if s.result == "correct")
        net_hits            = sum(1 for s in shots if s.result == "net")
        misses              = sum(1 for s in shots if s.result == "miss")
        out_shots           = sum(1 for s in shots if s.result == "out")
        incorrect_shots     = net_hits + misses + out_shots

        shot_accuracy_pct = (correct_shots / total * 100) if total else 0.0
        full_accuracy_pct = (fully_correct_shots / total * 100) if total else 0.0

        speeds = [s.speed for s in shots if s.speed > 0]
        avg_speed = float(np.mean(speeds)) if speeds else 0.0
        max_speed = float(np.max(speeds))  if speeds else 0.0

        direction_counts: Dict[str, int] = {}
        for s in shots:
            direction_counts[s.direction] = direction_counts.get(s.direction, 0) + 1

        posture_analysis = self._compute_posture()
        stage_score      = self._compute_stage_score(
            shot_accuracy_pct, avg_speed, posture_analysis["posture_score"]
        )

        return {
            "total_shots"          : total,
            "correct_shots"        : correct_shots,
            "fully_correct_shots"  : fully_correct_shots,
            "incorrect_shots"      : incorrect_shots,
            "net_hits"             : net_hits,
            "misses"               : misses,
            "out_shots"            : out_shots,
            "shot_accuracy_pct"    : shot_accuracy_pct,
            "full_accuracy_pct"    : full_accuracy_pct,
            "avg_ball_speed"       : avg_speed,
            "max_ball_speed"       : max_speed,
            "direction_counts"     : direction_counts,
            "posture_analysis"     : posture_analysis,
            "stage_score"          : stage_score,
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _classify_shot(self, frame_idx: int,
                       x: int, y: int, net_prob: float) -> Shot:
        """Determine shot speed, direction, and result."""
        speed     = self._estimate_speed()
        direction = self._estimate_direction()
        result    = self._determine_result(x, y, speed, direction, net_prob)
        return Shot(frame_idx=frame_idx, result=result,
                    speed=speed, direction=direction, ball_pos=(x, y))

    def _estimate_speed(self) -> float:
        """Average speed over the last few ball positions (px/frame)."""
        hist = self._ball_history
        if len(hist) < 2:
            return 0.0
        window = hist[-6:] if len(hist) >= 6 else hist
        dists = []
        for i in range(1, len(window)):
            f0, x0, y0 = window[i - 1]
            f1, x1, y1 = window[i]
            df = max(f1 - f0, 1)
            dists.append(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / df)
        return float(np.mean(dists)) if dists else 0.0

    def _estimate_direction(self) -> str:
        """Classify ball direction based on recent x-movement."""
        hist = self._ball_history
        if len(hist) < 3:
            return "unknown"
        window = hist[-8:] if len(hist) >= 8 else hist
        dx = window[-1][1] - window[0][1]   # x[-1] - x[0]
        if abs(dx) < 5:
            return "straight"
        return "right→left" if dx < 0 else "left→right"

    def _determine_result(self, x: int, y: int,
                          speed: float, direction: str,
                          net_prob: float) -> str:
        """Map shot characteristics to a result label."""
        # Ball not detected (position at or near origin) → miss
        if x == 0 and y == 0:
            return "miss"
        if x < 5 and y < 5:
            return "miss"

        cfg = self.stage_config

        # Speed check
        sp_min, sp_max = cfg.get("target_speed_range", (0, 9999))
        speed_ok = sp_min <= speed <= sp_max

        # Net hit: only if speed > 0 (ball actually moving) AND net_prob is high
        if net_prob >= self.net_thresh and speed > 0:
            return "net"

        # Direction check
        target_dir  = cfg.get("target_direction", "any")
        dir_ok   = (target_dir == "any") or (direction == target_dir)

        # Zone check (left/right half of the table based on x in original res)
        target_zone = cfg.get("target_zone", "any")
        if target_zone == "any":
            zone_ok = True
        elif target_zone == "left":
            half = max(self.scene_width // 2, 1)
            zone_ok = x < half
        else:
            half = max(self.scene_width // 2, 1)
            zone_ok = x >= half

        if speed_ok and dir_ok and zone_ok:
            return "correct"
        return "out"

    def _update_posture(self, seg_mask: np.ndarray):
        """Track centroid of left-player region in segmentation mask."""
        if seg_mask is None:
            return
        arr = np.array(seg_mask)
        if arr.ndim == 3:
            mask = arr[:, : arr.shape[1] // 2, 0]  # left half, first channel
        else:
            mask = arr[:, : arr.shape[1] // 2]

        ys, xs = np.where(mask > 0)
        if len(xs) > 10:
            self._posture_x.append(float(np.mean(xs)))
            self._posture_y.append(float(np.mean(ys)))

    def _compute_posture(self) -> dict:
        """Return posture stability metrics."""
        if len(self._posture_x) < 3:
            return {"posture": "غير متاح", "std_x": 0.0,
                    "std_y": 0.0, "posture_score": 50.0}

        std_x = float(np.std(self._posture_x))
        std_y = float(np.std(self._posture_y))
        total_std = std_x + std_y

        if total_std < 10:
            posture      = "ممتاز  ✅"
            posture_score = 95.0
        elif total_std < 25:
            posture      = "جيد    🟡"
            posture_score = 75.0
        elif total_std < 50:
            posture      = "متذبذب ⚠️"
            posture_score = 50.0
        else:
            posture      = "ضعيف   ❌"
            posture_score = 25.0

        return {"posture"      : posture,
                "std_x"        : std_x,
                "std_y"        : std_y,
                "posture_score": posture_score}

    def _compute_stage_score(self, accuracy_pct: float,
                             avg_speed: float,
                             posture_score: float) -> float:
        """
        Weighted stage score (0-100):
          60% → shot accuracy
          20% → speed within target range
          20% → posture stability
        """
        sp_min, sp_max = self.stage_config.get("target_speed_range", (0, 9999))
        sp_mid   = (sp_min + sp_max) / 2
        sp_range = max(sp_max - sp_min, 1)
        speed_score = max(0.0, 100.0 - abs(avg_speed - sp_mid) / sp_range * 100)

        return (accuracy_pct * 0.60
                + speed_score  * 0.20
                + posture_score * 0.20)

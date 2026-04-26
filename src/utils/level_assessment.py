"""
Level Assessment
Runs a short trial session (10 balls by default) and determines the
player's starting level: beginner / intermediate / advanced.

Assessment criteria (weighted score 0-100)
──────────────────────────────────────────
  50%  → shot accuracy   (correct bounces / total shots)
  30%  → average ball speed  (compared to reference ranges)
  20%  → posture score   (from PlayerAnalyzer)

Level thresholds
────────────────
  score < 50   → beginner      (المبتدئ)   → starts at Stage 1
  50 ≤ score < 75 → intermediate (المتوسط)  → starts at Stage 4
  score ≥ 75   → advanced      (المتقدم)   → starts at Stage 7
"""

from utils.player_analysis import PlayerAnalyzer

# Speed reference ranges for each level (px/frame)
SPEED_REFS = {
    "beginner"    : (2,  12),
    "intermediate": (10, 22),
    "advanced"    : (18, 40),
}

TRIAL_CONFIG = {
    "num_balls" : 10,
    "max_frames": 120,
    # No direction/zone requirements – just see what the player can do
    "target_speed_range" : (0, 9999),
    "target_direction"   : "any",
    "target_zone"        : "any",
}


class LevelAssessment:
    """
    Wraps PlayerAnalyzer for the trial period and converts results
    into a level label and a starting stage.

    Usage
    -----
    assessor = LevelAssessment()
    analyzer = assessor.get_analyzer()          # use this in the demo loop
    # ... feed frames to analyzer.update(...)   ...
    level, start_stage, report = assessor.finalise(analyzer)
    """

    def __init__(self, num_trial_balls=10, max_frames=120):
        self.num_trial_balls = num_trial_balls
        self.max_frames      = max_frames

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_analyzer(self):
        """Return a fresh PlayerAnalyzer configured for the trial."""
        return PlayerAnalyzer(stage_config=TRIAL_CONFIG)

    def finalise(self, analyzer):
        """
        Compute level from analyzer state.

        Returns
        -------
        level       : str  ("beginner" | "intermediate" | "advanced")
        start_stage : int
        report      : str  (human-readable Arabic report)
        """
        stats = analyzer.get_stage_stats()
        level, assessment_score = self._determine_level(stats)

        from utils.stage_manager import LEVEL_START_STAGE
        start_stage = LEVEL_START_STAGE[level]

        report = self._build_report(stats, level, assessment_score, start_stage)
        return level, start_stage, report

    # ── Private helpers ────────────────────────────────────────────────────────

    def _determine_level(self, stats):
        """Return (level_str, assessment_score_0_to_100)."""
        accuracy_pct  = stats.get("shot_accuracy_pct", 0)
        avg_speed     = stats.get("avg_ball_speed", 0)
        posture_score = stats.get("posture_analysis", {}).get("posture_score", 50)

        speed_score = self._speed_score(avg_speed)

        assessment_score = (
            accuracy_pct  * 0.50
            + speed_score * 0.30
            + posture_score * 0.20
        )

        if assessment_score < 50:
            level = "beginner"
        elif assessment_score < 75:
            level = "intermediate"
        else:
            level = "advanced"

        return level, assessment_score

    def _speed_score(self, avg_speed):
        """
        Map average ball speed to 0-100:
          very slow (< 4 px/f)   → 20
          beginner range          → 50
          intermediate range      → 75
          advanced range          → 100
        """
        if avg_speed < 4:
            return 20
        elif avg_speed < 12:
            return 50
        elif avg_speed < 22:
            return 75
        else:
            return 100

    def _build_report(self, stats, level, score, start_stage):
        level_ar = {
            "beginner"    : "مبتدئ     🟢",
            "intermediate": "متوسط     🟡",
            "advanced"    : "متقدم     🔴",
        }[level]

        sep = "═" * 48
        lines = [
            sep,
            "  🎯 نتيجة تقييم المستوى",
            sep,
            f"  عدد الكرات التجريبية : {stats.get('total_shots', 0)}",
            f"  الضربات الصحيحة      : {stats.get('correct_shots', 0)}",
            f"  دقة الضربات          : {stats.get('shot_accuracy_pct', 0):.1f}%",
            f"  متوسط سرعة الكرة    : {stats.get('avg_ball_speed', 0):.1f} px/frame",
            f"  وضعية اللعب          : {stats.get('posture_analysis', {}).get('posture', '—')}",
            sep,
            f"  📊 درجة التقييم     : {score:.1f} / 100",
            f"  🏅 مستواك            : {level_ar}",
            f"  ▶  ستبدأ من          : المرحلة {start_stage}",
            sep,
        ]
        return "\n".join(lines)

"""
Stage Manager
Defines all training stages, evaluates performance, and produces reports.

Stage structure
───────────────
  Level      │ Stages │ Focus
  ───────────┼────────┼──────────────────────────────────────
  مبتدئ      │  1-3   │ أي ضربة صحيحة، سرعة منخفضة
  متوسط      │  4-6   │ اتجاه محدد، سرعة متوسطة
  متقدم      │  7-9   │ توجيه دقيق، سرعة عالية، ثبات أعلى
"""

# ── Stage definitions ──────────────────────────────────────────────────────────
# target_direction : "right→left" | "left→right" | "straight" | "any"
# target_zone      : "left" | "right" | "any"
# target_speed_range: (min_px_per_frame, max_px_per_frame)
# pass_threshold   : minimum stage_score (0-100) to advance

STAGES = {
    1: {
        "name"               : "المرحلة 1 - مبتدئ (البداية)",
        "level"              : "beginner",
        "num_balls"          : 10,
        "max_frames"         : 120,
        "target_speed_range" : (2, 10),
        "target_direction"   : "any",
        "target_zone"        : "any",
        "pass_threshold"     : 60,
        "description"        : "ضربات بسيطة بسرعة منخفضة – أي اتجاه مقبول",
    },
    2: {
        "name"               : "المرحلة 2 - مبتدئ (تطوير)",
        "level"              : "beginner",
        "num_balls"          : 12,
        "max_frames"         : 120,
        "target_speed_range" : (4, 12),
        "target_direction"   : "any",
        "target_zone"        : "any",
        "pass_threshold"     : 65,
        "description"        : "زيادة عدد الكرات مع رفع طفيف في السرعة",
    },
    3: {
        "name"               : "المرحلة 3 - مبتدئ (إتقان)",
        "level"              : "beginner",
        "num_balls"          : 15,
        "max_frames"         : 125,
        "target_speed_range" : (6, 14),
        "target_direction"   : "any",
        "target_zone"        : "right",
        "pass_threshold"     : 68,
        "description"        : "توجيه الكرة للجهة اليمنى من الطاولة",
    },
    4: {
        "name"               : "المرحلة 4 - متوسط (البداية)",
        "level"              : "intermediate",
        "num_balls"          : 15,
        "max_frames"         : 125,
        "target_speed_range" : (8, 18),
        "target_direction"   : "right→left",
        "target_zone"        : "any",
        "pass_threshold"     : 70,
        "description"        : "ضربات باتجاه اليسار، سرعة متوسطة",
    },
    5: {
        "name"               : "المرحلة 5 - متوسط (تطوير)",
        "level"              : "intermediate",
        "num_balls"          : 18,
        "max_frames"         : 128,
        "target_speed_range" : (10, 20),
        "target_direction"   : "right→left",
        "target_zone"        : "left",
        "pass_threshold"     : 73,
        "description"        : "توجيه دقيق لليسار مع رفع السرعة",
    },
    6: {
        "name"               : "المرحلة 6 - متوسط (إتقان)",
        "level"              : "intermediate",
        "num_balls"          : 20,
        "max_frames"         : 128,
        "target_speed_range" : (12, 22),
        "target_direction"   : "any",
        "target_zone"        : "any",
        "pass_threshold"     : 75,
        "description"        : "تنويع الاتجاهات مع سرعة متوسطة-عالية",
    },
    7: {
        "name"               : "المرحلة 7 - متقدم (البداية)",
        "level"              : "advanced",
        "num_balls"          : 20,
        "max_frames"         : 128,
        "target_speed_range" : (15, 30),
        "target_direction"   : "right→left",
        "target_zone"        : "left",
        "pass_threshold"     : 78,
        "description"        : "ضربات قوية دقيقة نحو اليسار",
    },
    8: {
        "name"               : "المرحلة 8 - متقدم (تطوير)",
        "level"              : "advanced",
        "num_balls"          : 25,
        "max_frames"         : 128,
        "target_speed_range" : (18, 35),
        "target_direction"   : "any",
        "target_zone"        : "any",
        "pass_threshold"     : 82,
        "description"        : "ضربات متقدمة متنوعة بسرعة عالية",
    },
    9: {
        "name"               : "المرحلة 9 - متقدم (إتقان)",
        "level"              : "advanced",
        "num_balls"          : 30,
        "max_frames"         : 128,
        "target_speed_range" : (20, 40),
        "target_direction"   : "any",
        "target_zone"        : "any",
        "pass_threshold"     : 88,
        "description"        : "مرحلة الاحتراف – دقة وسرعة وثبات عالي",
    },
}

# ── Level → starting stage mapping ────────────────────────────────────────────
LEVEL_START_STAGE = {
    "beginner"    : 1,
    "intermediate": 4,
    "advanced"    : 7,
}


# ── StageManager ───────────────────────────────────────────────────────────────
class StageManager:
    """
    Manages stage progression for a training session.

    Usage
    -----
    mgr = StageManager(starting_level="beginner")
    # ... run PlayerAnalyzer for current stage ...
    stats  = analyzer.get_stage_stats()
    result = mgr.evaluate_stage(stats)
    report = mgr.generate_report(stats)
    if result["passed"]:
        mgr.advance()
    else:
        mgr.retry()
    """

    def __init__(self, starting_level="beginner"):
        self.current_stage_id = LEVEL_START_STAGE.get(starting_level, 1)
        self.session_history  = []   # list of (stage_id, stats, passed)
        self.total_stages     = len(STAGES)

    # ── Stage navigation ───────────────────────────────────────────────────────

    def current_stage(self):
        """Return the config dict of the current stage."""
        return STAGES[self.current_stage_id]

    def advance(self):
        """Move to the next stage. Returns False if already at final stage."""
        if self.current_stage_id < self.total_stages:
            self.current_stage_id += 1
            return True
        return False

    def retry(self):
        """Stay at same stage (called after failure)."""
        pass

    def is_finished(self):
        return self.current_stage_id > self.total_stages

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate_stage(self, stats):
        """
        Decide pass/fail for the current stage.

        Returns
        -------
        dict:
          passed       : bool
          stage_score  : float (0-100)
          pass_threshold: float
          message      : str
        """
        stage     = self.current_stage()
        score     = stats.get("stage_score", 0)
        threshold = stage["pass_threshold"]
        passed    = score >= threshold

        message = (
            f"✅ تهانينا! تجاوزت {stage['name']} بنتيجة {score:.1f}/100"
            if passed else
            f"❌ لم تتجاوز {stage['name']} – حصلت على {score:.1f}/100 "
            f"(المطلوب {threshold}/100). أعد المحاولة."
        )

        outcome = {
            "passed"         : passed,
            "stage_score"    : score,
            "pass_threshold" : threshold,
            "message"        : message,
        }
        self.session_history.append((self.current_stage_id, stats, passed))
        return outcome

    # ── Report generation ──────────────────────────────────────────────────────

    def generate_report(self, stats):
        """Return a formatted text report for the completed stage."""
        stage    = self.current_stage()
        outcome  = self.evaluate_stage(stats)
        posture  = stats.get("posture_analysis", {})

        sep = "═" * 48
        report_lines = [
            sep,
            f"  📋 تقرير {stage['name']}",
            f"  {stage['description']}",
            sep,
            f"  🎯 عدد الكرات الكلي     : {stats.get('total_shots', 0)}",
            f"  ✅ ضربات صحيحة كاملاً  : {stats.get('fully_correct_shots', 0)}",
            f"  ✅ ضربات صحيحة (وصلت)  : {stats.get('correct_shots', 0)}",
            f"  ❌ ضربات خاطئة         : {stats.get('incorrect_shots', 0)}",
            f"     ↳ لمست الشبكة        : {stats.get('net_hits', 0)}",
            f"     ↳ فاتته الكرة (miss) : {stats.get('misses', 0)}",
            f"     ↳ خارج المنطقة/الاتجاه: {stats.get('out_shots', 0)}",
            sep,
            f"  📊 دقة الضربات         : {stats.get('shot_accuracy_pct', 0):.1f}%",
            f"  📊 الدقة الكاملة       : {stats.get('full_accuracy_pct', 0):.1f}%",
            f"  ⚡ متوسط سرعة الكرة   : {stats.get('avg_ball_speed', 0):.1f} px/frame",
            f"  ⚡ أقصى سرعة للكرة    : {stats.get('max_ball_speed', 0):.1f} px/frame",
            f"  🎯 النطاق المستهدف    : {stage['target_speed_range'][0]}"
            f"–{stage['target_speed_range'][1]} px/frame",
            sep,
            f"  🏃 وضعية اللعب        : {posture.get('posture', 'غير متاح')}",
            f"     ↳ تذبذب أفقي (X)    : {posture.get('std_x', 0):.1f} px",
            f"     ↳ تذبذب عمودي (Y)   : {posture.get('std_y', 0):.1f} px",
            sep,
            f"  🏆 نسبة النجاح        : {stats.get('stage_score', 0):.1f} / 100",
            f"  📌 الحد المطلوب       : {stage['pass_threshold']} / 100",
            sep,
            f"  {outcome['message']}",
            sep,
        ]

        # Direction breakdown
        dirs = stats.get("direction_counts", {})
        if dirs:
            report_lines.insert(-2, "  📐 توزيع الاتجاهات:")
            for direction, count in dirs.items():
                report_lines.insert(-2, f"     ↳ {direction}: {count} ضربة")

        return "\n".join(report_lines)

    def session_summary(self):
        """Print a summary of the entire session."""
        sep = "─" * 48
        lines = [
            "═" * 48,
            "  📈 ملخص الجلسة التدريبية الكاملة",
            "═" * 48,
        ]
        for stage_id, stats, passed in self.session_history:
            stage  = STAGES[stage_id]
            status = "✅ نجاح" if passed else "❌ رسوب"
            lines.append(
                f"  {stage['name'][:30]:<30} "
                f"نتيجة: {stats.get('stage_score', 0):5.1f}/100  {status}"
            )
        lines.append(sep)
        total   = len(self.session_history)
        passed  = sum(1 for _, _, p in self.session_history if p)
        lines.append(f"  إجمالي المراحل: {total} | نجاح: {passed} | رسوب: {total - passed}")
        lines.append("═" * 48)
        return "\n".join(lines)

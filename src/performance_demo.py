"""
performance_demo.py
───────────────────
Main entry point for Player Performance Analysis.

Flow
────
  1. [ASSESSMENT] Run a short trial video to determine the player's level
                  and which stage to start from.

  2. [STAGE LOOP] For each stage:
       a. Run the TTNet model on the stage video (max_frames < 130).
       b. Focus only on left-player frames (left half of segmentation).
       c. Collect per-frame outputs → PlayerAnalyzer.
       d. After all frames (or num_balls reached): evaluate the stage.
       e. Print a full Arabic report.
       f. If passed → advance; else → retry (up to max_retries).

  3. Print a session summary at the end.

Run example
───────────
  python performance_demo.py \
      --pretrained_path ../../checkpoints/ttnet/ttnet_best.pth \
      --assessment_video /path/to/trial.mp4 \
      --stage_videos /path/to/stage1.mp4 /path/to/stage2.mp4 \
      --starting_level beginner \
      --gpu_idx 0

  # To skip assessment and force a level:
  python performance_demo.py \
      --pretrained_path ../../checkpoints/ttnet/ttnet_best.pth \
      --stage_videos /path/to/stage_video.mp4 \
      --starting_level intermediate \
      --skip_assessment

Notes
─────
  - Default `--max_frames` per stage (~120–128) caps runtime; `--max_frames N` overrides that.
  - **`--max_frames 0`** or **`--full_video`** = scan **the whole clip** (all sequences); only frames
    with left-player segmentation count when that filter is on.
  - Ball XY are scaled using the opened video resolution (was hard-coded 1920×1080 before).
  - Use `--bounce_thresh 0.15` … `0.3` if you see posture OK but zero shots (model outputs weak).
  - `--skip_left_player_filter`: process frames even when left-player segmentation is empty.
"""

import os
import sys
import argparse
from collections import deque

import cv2
import numpy as np
import torch

sys.path.append(os.path.dirname(__file__))

from data_process.ttnet_video_loader import TTNet_Video_Loader
from models.model_utils import create_model, load_pretrained_model
from config.config import parse_configs
from utils.post_processing import post_processing
from utils.misc import time_synchronized
from utils.player_analysis import PlayerAnalyzer
from utils.stage_manager import StageManager, STAGES
from utils.level_assessment import LevelAssessment


def _no_frame_cap(max_frames):
    """Run until EOF when True."""
    return max_frames is None or max_frames <= 0


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run TTNet on a video and feed results to analyzer
# ─────────────────────────────────────────────────────────────────────────────

def run_model_on_video(video_path, model, configs,
                       analyzer, max_frames=128, show_image=False,
                       require_left_player=True,
                       progress_every=500):
    """
    Walk the video **from the start**. Each loader step is one new frame in the TTNet sequence.

    `max_frames` counts only frames **accepted** into the analyzer (after optional left-player
    filter). Use `None` or `<= 0` for **no limit** → whole file.

    Returns the number of analyzer frames processed.
    """
    video_loader = TTNet_Video_Loader(
        video_path, configs.input_size, configs.num_frames_sequence
    )

    middle_idx  = configs.num_frames_sequence // 2
    queue_frames = deque(maxlen=middle_idx + 1)

    w_original, h_original = video_loader.video_w, video_loader.video_h
    if w_original <= 0 or h_original <= 0:
        w_original, h_original = 1920, 1080

    analyzer.scene_width = w_original

    w_resize, h_resize = configs.input_size[0], configs.input_size[1]   # 320, 128
    w_ratio = w_original / w_resize
    h_ratio = h_original / h_resize

    frame_idx     = 0
    processed     = 0

    with torch.no_grad():
        for count, resized_imgs in video_loader:

            if (not _no_frame_cap(max_frames)) and processed >= max_frames:
                print(f"  [INFO] الحد الأقصى للفريمات ({max_frames}) تم الوصول إليه.")
                break

            # ── model inference ────────────────────────────────────────────────
            img_tensor = (torch.from_numpy(resized_imgs)
                          .to(configs.device, non_blocking=True)
                          .float()
                          .unsqueeze(0))

            pred_ball_global, pred_ball_local, pred_events, pred_seg = \
                model.run_demo(img_tensor)

            prediction_global, prediction_local, prediction_seg, prediction_events = \
                post_processing(
                    pred_ball_global, pred_ball_local,
                    pred_events, pred_seg,
                    configs.input_size[0],
                    configs.thresh_ball_pos_mask,
                    configs.seg_thresh,
                    configs.event_thresh,
                )

            # Final ball position in original resolution
            ball_final = [
                int(prediction_global[0] * w_ratio + prediction_local[0] - w_resize / 2),
                int(prediction_global[1] * h_ratio + prediction_local[1] - h_resize / 2),
            ]

            # ── optional: skip frames without left-player segmentation ────────
            seg = prediction_seg  # shape (128, 320, 3) binary
            if require_left_player:
                left_player_pixels = seg[:, : seg.shape[1] // 2, 0].sum()
                if left_player_pixels == 0:
                    frame_idx += 1
                    continue

            # ── feed to analyzer ───────────────────────────────────────────────
            shot = analyzer.update(
                ball_pos     = ball_final,
                bounce_prob  = float(prediction_events[0]),
                net_prob     = float(prediction_events[1]),
                seg_mask     = seg,
                frame_idx    = frame_idx,
            )

            if shot is not None:
                _print_shot_live(shot, frame_idx)

            # ── optional display ───────────────────────────────────────────────
            if show_image:
                disp_img = resized_imgs[3 * middle_idx: 3 * (middle_idx + 1)]
                disp_img = disp_img.transpose(1, 2, 0).astype(np.uint8)
                disp_img = cv2.cvtColor(
                    cv2.resize(disp_img, (640, 256)), cv2.COLOR_RGB2BGR
                )
                cv2.circle(disp_img,
                           (ball_final[0] // 3, ball_final[1] // 4),
                           5, (0, 255, 0), -1)
                cv2.putText(disp_img,
                            f"B:{prediction_events[0]:.2f} N:{prediction_events[1]:.2f}",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 0), 1)
                cv2.imshow("Performance Demo", disp_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1
            processed += 1
            if (_no_frame_cap(max_frames) and progress_every and processed > 0
                    and processed % progress_every == 0):
                print(f"  [INFO] تمت معالجة {processed:d} فريماً تحليلياً … (كل الفيديو)")

        if _no_frame_cap(max_frames):
            print("  [INFO] انتهى الفيديو – تم مسح كل التسلسلات المتاحة.")

    cv2.destroyAllWindows()
    return processed


def _print_shot_live(shot, frame_idx):
    result_icon = {"correct": "✅", "net": "🔴", "miss": "⚪", "out": "🟡"}.get(
        shot.result, "?"
    )
    print(f"  فريم {frame_idx:4d} │ {result_icon} {shot.result:<8} │ "
          f"سرعة={shot.speed:5.1f} px/f │ اتجاه={shot.direction}")


# ─────────────────────────────────────────────────────────────────────────────
# Assessment phase
# ─────────────────────────────────────────────────────────────────────────────

def run_assessment(assessment_video, model, configs, max_frames=120,
                   bounce_thresh=0.25, net_thresh=0.30, show_image=False,
                   require_left_player=True):
    print("\n" + "═" * 48)
    print("  🎯 مرحلة تحديد المستوى (التقييم الأولي)")
    print("═" * 48)

    assessor = LevelAssessment(max_frames=max_frames)
    analyzer = assessor.get_analyzer()
    analyzer.bounce_thresh = bounce_thresh
    analyzer.net_thresh    = net_thresh

    run_model_on_video(assessment_video, model, configs,
                       analyzer, max_frames=max_frames, show_image=show_image,
                       require_left_player=require_left_player)

    level, start_stage, report = assessor.finalise(analyzer)
    print(report)
    return level, start_stage


# ─────────────────────────────────────────────────────────────────────────────
# Stage loop
# ─────────────────────────────────────────────────────────────────────────────

def run_stage_loop(stage_videos, model, configs,
                   starting_level, starting_stage,
                   max_retries=3, bounce_thresh=0.25, net_thresh=0.30,
                   show_image=False, max_frames_override=None,
                   require_left_player=True):
    """
    `max_frames_override`: None → each stage uses its default cap in STAGES.
                           <= 0 → no cap (whole video per stage).
                           > 0  → use this many accepted frames per stage.
    """
    mgr = StageManager(starting_level=starting_level)
    mgr.current_stage_id = starting_stage

    def _get_video(stage_id):
        if isinstance(stage_videos, (list, tuple)):
            idx = stage_id - 1
            return stage_videos[idx] if idx < len(stage_videos) else stage_videos[-1]
        return stage_videos  # single video for all stages

    retry_counts = {}

    while not mgr.is_finished():
        stage_id  = mgr.current_stage_id
        stage_cfg = mgr.current_stage()
        retries   = retry_counts.get(stage_id, 0)

        if retries > max_retries:
            print(f"\n⛔ تجاوزت الحد الأقصى للمحاولات في {stage_cfg['name']}. "
                  f"انتهت الجلسة.")
            break

        attempt_label = f"(محاولة {retries + 1}/{max_retries + 1})" if retries else ""
        print(f"\n{'═'*48}")
        print(f"  ▶  {stage_cfg['name']}  {attempt_label}")
        print(f"  {stage_cfg['description']}")
        print(f"  عدد الكرات: {stage_cfg['num_balls']} │ "
              f"سرعة: {stage_cfg['target_speed_range']} px/frame │ "
              f"اتجاه: {stage_cfg['target_direction']} │ "
              f"منطقة: {stage_cfg['target_zone']}")
        print("═" * 48)

        # Create fresh analyzer with this stage's config
        analyzer = PlayerAnalyzer(stage_config=stage_cfg,
                                  bounce_thresh=bounce_thresh,
                                  net_thresh=net_thresh)
        video_path = _get_video(stage_id)

        mf = stage_cfg["max_frames"]
        if max_frames_override is not None:
            mf = None if max_frames_override <= 0 else max_frames_override

        processed = run_model_on_video(
            video_path, model, configs,
            analyzer,
            max_frames=mf,
            show_image=show_image,
            require_left_player=require_left_player,
        )
        print(f"\n  [✓] انتهت المرحلة – تم تحليل {processed} فريم")

        stats  = analyzer.get_stage_stats()
        report = mgr.generate_report(stats)
        print("\n" + report)

        outcome = mgr.evaluate_stage(stats)

        if outcome["passed"]:
            if not mgr.advance():
                print("\n🏆 أنهيت جميع المراحل! أنت محترف.")
                break
        else:
            retry_counts[stage_id] = retries + 1
            mgr.retry()

    # Session summary
    print("\n" + mgr.session_summary())


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_perf_args():
    p = argparse.ArgumentParser(
        description="TTNet Player Performance Analysis",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--pretrained_path", required=True,
                   help="Path to the pre-trained TTNet checkpoint (.pth)")
    p.add_argument("--assessment_video", default=None,
                   help="Video file for the initial level assessment")
    p.add_argument("--stage_videos", nargs="+", default=None,
                   help="Video file(s) for each stage (or one video reused for all)")
    p.add_argument("--starting_level",
                   choices=["beginner", "intermediate", "advanced"],
                   default="beginner",
                   help="Force a starting level (used when --skip_assessment is set)")
    p.add_argument("--skip_assessment", action="store_true",
                   help="Skip the assessment phase and use --starting_level directly")
    p.add_argument("--max_retries", type=int, default=3,
                   help="Maximum retries per stage before the session ends")
    p.add_argument("--gpu_idx", type=int, default=None,
                   help="GPU index (None = CPU)")
    p.add_argument("--show_image", action="store_true",
                   help="Show live video window during processing")
    p.add_argument("--bounce_thresh", type=float, default=0.25,
                   help="Min bounce event score to count a shot (match training; 0.5 is often too strict)")
    p.add_argument("--net_thresh", type=float, default=0.30,
                   help="Min net-hit score to label a shot as net")
    p.add_argument("--max_frames", type=int, default=None,
                   help="حد أقصى لإطارات التحليل (بعد فلتر اللاعب اليسار). 0 = كامل الفيديو بدون سقف.")
    p.add_argument("--full_video", action="store_true",
                   help="نفس تأثير --max_frames 0: مسح الملف من البداية إلى النهاية.")
    p.add_argument("--skip_left_player_filter", action="store_true",
                   help="Process every frame even if left-player segmentation is empty (debug / side view)")
    return p.parse_args()


def main():
    # ── parse TTNet configs (needed for model creation) ────────────────────────
    configs = parse_configs()

    # ── parse performance-specific args ───────────────────────────────────────
    perf_args = parse_perf_args()

    # ── device ────────────────────────────────────────────────────────────────
    if perf_args.gpu_idx is not None and torch.cuda.is_available():
        configs.device  = torch.device(f"cuda:{perf_args.gpu_idx}")
        configs.gpu_idx = perf_args.gpu_idx
    else:
        configs.device = torch.device("cpu")
    print(f"  [Device] {configs.device}")

    # ── build model ───────────────────────────────────────────────────────────
    model = create_model(configs)
    model.to(configs.device)
    model = load_pretrained_model(
        model, perf_args.pretrained_path,
        perf_args.gpu_idx,
        configs.overwrite_global_2_local,
    )
    model.eval()

    configs.thresh_ball_pos_mask = 0.05
    configs.seg_thresh           = 0.5
    configs.event_thresh         = 0.5

    # ── assessment ────────────────────────────────────────────────────────────
    starting_level = perf_args.starting_level
    starting_stage = {"beginner": 1, "intermediate": 4, "advanced": 7}[starting_level]

    require_left = not perf_args.skip_left_player_filter

    if perf_args.full_video:
        frame_cap = 0
    else:
        frame_cap = perf_args.max_frames

    if frame_cap is None:
        assess_max = 120
    elif frame_cap <= 0:
        assess_max = None
    else:
        assess_max = frame_cap

    if not perf_args.skip_assessment and perf_args.assessment_video:
        starting_level, starting_stage = run_assessment(
            perf_args.assessment_video, model, configs,
            max_frames=assess_max,
            bounce_thresh=perf_args.bounce_thresh,
            net_thresh=perf_args.net_thresh,
            show_image=perf_args.show_image,
            require_left_player=require_left,
        )

    print(f"\n  ▶  سيبدأ اللاعب من المرحلة {starting_stage} "
          f"(مستوى: {starting_level})\n")

    # ── stage loop ────────────────────────────────────────────────────────────
    if not perf_args.stage_videos:
        print("⚠️  لم يتم تحديد --stage_videos. انتهى البرنامج.")
        return

    run_stage_loop(
        stage_videos   = perf_args.stage_videos,
        model          = model,
        configs        = configs,
        starting_level = starting_level,
        starting_stage = starting_stage,
        max_retries    = perf_args.max_retries,
        bounce_thresh  = perf_args.bounce_thresh,
        net_thresh     = perf_args.net_thresh,
        show_image     = perf_args.show_image,
        max_frames_override=frame_cap,
        require_left_player=require_left,
    )


if __name__ == "__main__":
    main()

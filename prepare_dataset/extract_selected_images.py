"""
extract_selected_images.py
──────────────────────────
Extracts only the frames needed for TTNet training/analysis.

Additions vs original
─────────────────────
  --max_frames   : hard cap on extracted frames per video (default: 128)
  --left_player  : when set, only keep frames where the left half of the
                   frame has meaningful pixel content (proxy for left-player
                   activity). Requires the frame to have at least
                   LEFT_PLAYER_MIN_PIXELS non-dark pixels in the left half.
"""

import os
import argparse
from glob import glob
import json

import cv2
import numpy as np


# Minimum non-dark pixels in the left half to qualify as "left player active"
LEFT_PLAYER_MIN_PIXELS = 500
# Darkness threshold: pixels with mean channel value below this are ignored
DARK_THRESHOLD = 30


def make_folder(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)


def get_frame_indexes(events_annos_path, num_frames_from_event):
    json_file = open(events_annos_path)
    events_annos = json.load(json_file)
    selected_indexes = []
    main_frames = sorted(events_annos.keys())
    for main_f_idx in main_frames:
        main_f_idx = int(main_f_idx)
        for idx in range(main_f_idx - num_frames_from_event,
                         main_f_idx + num_frames_from_event + 1):
            selected_indexes.append(idx)
    return set(selected_indexes)


def _left_player_active(img, min_pixels=LEFT_PLAYER_MIN_PIXELS,
                         dark_thresh=DARK_THRESHOLD):
    """Return True if the left half of `img` contains enough non-dark pixels."""
    h, w = img.shape[:2]
    left_half = img[:, : w // 2]
    bright_mask = np.mean(left_half, axis=2) > dark_thresh
    return int(bright_mask.sum()) >= min_pixels


def extract_images_from_videos(video_path, events_annos_path, out_images_dir,
                                num_frames_from_event,
                                max_frames=128,
                                left_player_only=False):
    selected_indexes = get_frame_indexes(events_annos_path, num_frames_from_event)

    video_fn = os.path.basename(video_path)[:-4]
    sub_images_dir = os.path.join(out_images_dir, video_fn)
    make_folder(sub_images_dir)

    video_cap = cv2.VideoCapture(video_path)
    n_frames  = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_width   = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    f_height  = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f'video: {video_fn}.mp4  |  size: {f_width}×{f_height}  |  '
          f'total frames: {n_frames}  |  event frames: {len(selected_indexes)}  |  '
          f'cap: {max_frames}  |  left-only: {left_player_only}')

    frame_cnt   = -1
    saved_count = 0

    while True:
        ret, img = video_cap.read()
        if not ret:
            break

        frame_cnt += 1

        if frame_cnt not in selected_indexes:
            continue

        # ── left-player filter ─────────────────────────────────────────────────
        if left_player_only and not _left_player_active(img):
            continue

        # ── max_frames cap ─────────────────────────────────────────────────────
        if saved_count >= max_frames:
            print(f'  [cap] reached {max_frames} frames – stopping early.')
            break

        image_path = os.path.join(sub_images_dir, 'img_{:06d}.jpg'.format(frame_cnt))
        if os.path.isfile(image_path):
            print(f'  [skip] {video_fn} already extracted.')
            break

        cv2.imwrite(image_path, img)
        saved_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_cap.release()
    print(f'  done: saved {saved_count} frames from {video_fn}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract selected frames from TTNet dataset videos'
    )
    parser.add_argument('--dataset_dir', default='../dataset',
                        help='Root dataset directory')
    parser.add_argument('--num_frames_sequence', type=int, default=9,
                        help='Sequence length (paper used 9)')
    parser.add_argument('--max_frames', type=int, default=99999,
                        help='Maximum frames to extract per video (default: all frames)')
    parser.add_argument('--left_player', action='store_true',
                        help='Only extract frames where the left player is active')
    parser.add_argument('--dataset_types', nargs='+',
                        default=['training', 'test'],
                        help='Dataset splits to process')
    args = parser.parse_args()

    num_frames_from_event = int((args.num_frames_sequence - 1) / 2)

    for dataset_type in args.dataset_types:
        video_dir      = os.path.join(args.dataset_dir, dataset_type, 'videos')
        annos_dir      = os.path.join(args.dataset_dir, dataset_type, 'annotations')
        out_images_dir = os.path.join(args.dataset_dir, dataset_type, 'images')

        video_paths = glob(os.path.join(video_dir, '*.mp4'))
        print(f'\n── {dataset_type} ── {len(video_paths)} videos found')

        for video_path in video_paths:
            video_fn           = os.path.basename(video_path)[:-4]
            events_annos_path  = os.path.join(annos_dir, video_fn, 'events_markup.json')

            if not os.path.isfile(events_annos_path):
                print(f'  [warn] no annotations for {video_fn}, skipping.')
                continue

            extract_images_from_videos(
                video_path         = video_path,
                events_annos_path  = events_annos_path,
                out_images_dir     = out_images_dir,
                num_frames_from_event = num_frames_from_event,
                max_frames         = args.max_frames,
                left_player_only   = args.left_player,
            )

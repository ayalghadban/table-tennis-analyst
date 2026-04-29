import argparse
import os
from glob import glob

import cv2


def make_folder(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)


def extract_images_from_videos(video_path, out_images_dir):
    video_fn = os.path.basename(video_path)[:-4]
    sub_images_dir = os.path.join(out_images_dir, video_fn)

    make_folder(sub_images_dir)

    video_cap = cv2.VideoCapture(video_path)
    n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    f_width = video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    f_height = video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print('video_fn: {}.mp4, number of frames: {}, f_width: {}, f_height: {}'.format(video_fn, n_frames, f_width,
                                                                                     f_height))

    frame_cnt = -1
    while True:
        ret, img = video_cap.read()
        if ret:
            frame_cnt += 1
            image_path = os.path.join(sub_images_dir, 'img_{:06d}.jpg'.format(frame_cnt))
            if os.path.isfile(image_path):
                print('video {} had been already extracted'.format(video_path))
                break
            cv2.imwrite(image_path, img)
        else:
            break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    video_cap.release()
    print('done extraction: {}'.format(video_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract every frame from each video (full video → img_000000.jpg …)'
    )
    parser.add_argument(
        '--dataset_dir', type=str, default='../dataset',
        help='Root dataset directory (contains training/, test/, …)',
    )
    parser.add_argument(
        '--dataset_types', nargs='+', default=['training', 'test'],
        help='Which splits to process (e.g. training test)',
    )
    args = parser.parse_args()

    for dataset_type in args.dataset_types:
        video_dir = os.path.join(args.dataset_dir, dataset_type, 'videos')
        out_images_dir = os.path.join(args.dataset_dir, dataset_type, 'images')

        video_paths = glob(os.path.join(video_dir, '*.mp4'))
        print(f'\n── {dataset_type} ── {len(video_paths)} video(s)')

        for video_path in video_paths:
            extract_images_from_videos(video_path, out_images_dir)

import subprocess
import json
import os
import cv2
import h5py
import numpy as np
import tempfile
from pathlib import Path
import shutil
import time

MELEE_ISO = r"C:\dummy\melee.iso"
DOLPHIN_PATH = r"C:\dummy\dolphin.exe"
FRAME_DUMP_PATH = r"C:\dummy\frames"
DUMP_AVI_PATH = os.path.join(FRAME_DUMP_PATH, "framedump0.avi")

def create_slippi_config(slp_path):
    print(f"Creating Slippi config for {slp_path}")
    config = {
        "mode": "normal",
        "replay": os.path.abspath(slp_path),
        "isRealTimeMode": False,
        "outputOverlayFiles": False,
        "shouldAutomaticallyStop": True,
        "gameEndDelay": 0
    }

    temp_config = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
    json.dump(config, temp_config)
    temp_config.close()
    return temp_config.name

def record_replay(slp_path):
    print("\nStarting Dolphin recording...")
    config_path = create_slippi_config(slp_path)

    command = [
        DOLPHIN_PATH,
        f"--exec={MELEE_ISO}",
        f"--slippi-input={config_path}",
        "--batch",
        "--hide-seekbar"
    ]

    print(f"Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Dolphin: {e}")
        raise
    finally:
        os.unlink(config_path)
        print("Dolphin recording completed")

def extract_frames(target_frames):
    if not os.path.exists(DUMP_AVI_PATH):
        raise FileNotFoundError(f"No AVI file found at {DUMP_AVI_PATH}")

    print(f"\nExtracting frames from {DUMP_AVI_PATH}")
    cap = cv2.VideoCapture(DUMP_AVI_PATH)

    if not cap.isOpened():
        raise RuntimeError("Failed to open AVI file")

    frame_count = 0
    frames = []
    target_size = (96, 72)

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= target_frames:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        frames.append(frame)

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Extracted {frame_count} frames...")

    cap.release()
    print(f"Extracted {frame_count} frames total")
    return frames

def read_game_data(json_path):
    print(f"\nReading game data from {json_path}")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        p1_frames = data['players'][0]['frames']
        p2_frames = data['players'][1]['frames']
        last_frame = data['last_frame']

        print(f"Found {len(p1_frames)} frames of player data")
        print(f"Last frame in replay: {last_frame}")

        if len(p1_frames) != len(p2_frames):
            raise ValueError(f"Player frame counts don't match: P1={len(p1_frames)}, P2={len(p2_frames)}")

        target_frames = max(len(p1_frames), last_frame)
        print(f"Using target frame count: {target_frames}")

        return p1_frames, p2_frames, target_frames

    except (KeyError, IndexError) as e:
        print(f"Error parsing JSON data: {e}")
        raise

def quantize_analog(value, num_buckets, input_min, input_max):
    bucket_size = (input_max - input_min) / num_buckets
    offset = bucket_size / 2
    normalized = (value - input_min + offset) / (input_max - input_min)
    bucket = int(np.clip(normalized * num_buckets - 0.5, 0, num_buckets - 1))

    one_hot = np.zeros(num_buckets)
    one_hot[bucket] = 1
    return one_hot

def create_action_vector(frame_data):
    buttons = {
        0x0001: 'D-LEFT',
        0x0002: 'D-RIGHT',
        0x0004: 'D-DOWN',
        0x0008: 'D-UP',
        0x0010: 'Z',
        0x0020: 'R',
        0x0040: 'L',
        0x0100: 'A',
        0x0200: 'B',
        0x0400: 'Y',
        0x0800: 'X',
        0x1000: 'START'
    }

    button_vec = np.zeros(12)
    button_value = frame_data['buttons']
    for i, (bit, _) in enumerate(buttons.items()):
        if button_value & bit:
            button_vec[i] = 1

    joy_x = quantize_analog(frame_data['joy_x'], 16, -1, 1)
    joy_y = quantize_analog(frame_data['joy_y'], 16, -1, 1)
    c_x = quantize_analog(frame_data['c_x'], 16, -0.9875, 0.9875)
    c_y = quantize_analog(frame_data['c_y'], 16, -1, 1)
    trigger = quantize_analog(frame_data['trigger'], 4, 0, 1)

    return np.concatenate([
        button_vec,
        joy_x,
        joy_y,
        c_x,
        c_y,
        trigger
    ])

def create_hdf5(frames, p1_frames, p2_frames, output_path):
    print(f"\nCreating HDF5 file: {output_path}")
    max_frames = min(len(frames), len(p1_frames))

    with h5py.File(output_path, 'w') as f:
        f.attrs['num_frames'] = max_frames
        f.attrs['frame_width'] = 96
        f.attrs['frame_height'] = 72
        f.attrs['action_dims'] = 80

        for frame_idx in range(max_frames):
            if frame_idx % 100 == 0:
                print(f"Processing frame {frame_idx}/{max_frames}...")

            f.create_dataset(f'frame_{frame_idx}_x', data=frames[frame_idx])
            f.create_dataset(f'frame_{frame_idx}_p1_y', data=create_action_vector(p1_frames[frame_idx]))
            f.create_dataset(f'frame_{frame_idx}_p2_y', data=create_action_vector(p2_frames[frame_idx]))

def main(slp_path="./replays/sample.slp",
         json_path="./replays/sample.json",
         output_path="melee_data.hdf5"):

    try:
        p1_frames, p2_frames, target_frames = read_game_data(json_path)
        print("\nRecording replay...")
        record_replay(slp_path)
        print("\nExtracting frames from recording...")
        frames = extract_frames(target_frames)
        print("\nCreating HDF5 dataset...")
        create_hdf5(frames, p1_frames, p2_frames, output_path)
        print(f"\nSuccessfully created dataset at {output_path}")

    except Exception as e:
        print(f"\nError during processing: {e}")
        raise

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\nTotal processing time: {elapsed:.2f} seconds")
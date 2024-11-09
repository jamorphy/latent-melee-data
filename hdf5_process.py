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
from queue import Queue
import threading
from tqdm import tqdm

# Path Configuration
MELEE_ISO = r"C:\dummy\melee.iso"
DOLPHIN_PATH = r"C:\dummy\dolphin.exe"
SLIPPC_PATH = r"C:\dummy\slippc.exe"
FRAME_DUMP_PATH = r"C:\dummy\frames"
DUMP_AVI_PATH = os.path.join(FRAME_DUMP_PATH, "framedump0.avi")

def create_slippi_config(slp_path, last_frame):

    config = {
        "mode": "normal",
        "replay": os.path.abspath(slp_path),
        "isRealTimeMode": False,
        "outputOverlayFiles": False,
        "shouldAutomaticallyStop": True,
        "gameEndDelay": 0,
        "lastFrame": last_frame
    }

    temp_config = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
    json.dump(config, temp_config)
    temp_config.close()
    return temp_config.name

def create_frame_dump_path(replay_id):
    path = os.path.join(FRAME_DUMP_PATH, f"frames_{replay_id}")
    os.makedirs(path, exist_ok=True)
    return path

def convert_slp_to_json(slp_path, json_path):
    if os.path.exists(json_path):
        return
        
    command = [
        SLIPPC_PATH,
        "-i", slp_path,
        "-j", json_path,
        "-f"
    ]
    
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting {slp_path}: {e}")
        raise

def record_replay(slp_path, replay_id, last_frame):
    frame_dump_path = create_frame_dump_path(replay_id)
    avi_path = os.path.join(frame_dump_path, "framedump0.avi")
    
    if os.path.exists(avi_path):
        print(f"Frames already exist for replay {replay_id}, skipping recording...")
        return
        
    config_path = create_slippi_config(slp_path, last_frame)

    command = [
        DOLPHIN_PATH,
        f"--exec={MELEE_ISO}",
        f"--slippi-input={config_path}",
        f"--output-directory={frame_dump_path}",
        "--batch",
        "--hide-seekbar"
    ]

    try:
        # Try to suppress annoying sysconf error
        process = subprocess.Popen(command, stderr=subprocess.DEVNULL) 
        
        # Hacky solution for exiting Dolphin after the game ends.
        # Assuming a stable 60fps, calculate the game time, add 5s to account 
        # for dolphin loading and 15s to prevent a premature exit.
        # Doesn't need to be exact, since any garbage frames at the end are discarded
        # when compiling the dataset.
        gameplay_time = last_frame / 60
        wait_time = 5 + gameplay_time + 15
        
        time.sleep(wait_time)
        process.terminate()
        
        process.wait(timeout=5)
        if process.poll() is None:
            process.kill()
            
    finally:
        os.unlink(config_path)

def read_game_data(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        p1_frames = data['players'][0]['frames']
        p2_frames = data['players'][1]['frames']
        last_frame = data['last_frame']

        if len(p1_frames) != len(p2_frames):
            raise ValueError(f"Player frame counts don't match: P1={len(p1_frames)}, P2={len(p2_frames)}")

        target_frames = max(len(p1_frames), last_frame)

        return p1_frames, p2_frames, target_frames

    except (KeyError, IndexError) as e:
        print(f"Error parsing JSON data: {e}")
        raise

def process_replay_folder(replay_folder):
    json_folder = os.path.join(replay_folder, "json")
    os.makedirs(json_folder, exist_ok=True)

    slp_files = list(Path(replay_folder).glob("*.slp"))
    print(f"Found {len(slp_files)} .slp files to process")

    replay_configs = []
    
    for slp_path in tqdm(slp_files, desc="Converting .slp to .json"):
        base_name = slp_path.stem
        json_path = os.path.join(json_folder, f"{base_name}.json")
        
        convert_slp_to_json(str(slp_path), json_path)
        p1_frames, p2_frames, last_frame = read_game_data(json_path)
        
        replay_configs.append({
            'slp_path': str(slp_path),
            'json_path': json_path,
            'replay_id': base_name,
            'last_frame': last_frame
        })
    
    return replay_configs

def extract_frames(avi_path, target_frames):
    if not os.path.exists(avi_path):
        raise FileNotFoundError(f"No AVI file found at {avi_path}")

    cap = cv2.VideoCapture(avi_path)

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

    cap.release()
    return frames

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

def process_single_replay_to_hdf5(h5file, replay_id, frame_folder, json_path):
    avi_path = os.path.join(frame_folder, "framedump0.avi")
    if not os.path.exists(avi_path):
        print(f"Skipping replay {replay_id}: No AVI file found")
        return False
    
    replay_group = h5file.create_group(f"replay_{replay_id}")
    
    p1_frames, p2_frames, target_frames = read_game_data(json_path)
    
    frames = extract_frames(avi_path, target_frames)
    max_frames = min(len(frames), len(p1_frames))
    
    replay_group.create_dataset('frames', data=np.array(frames))
    
    p1_inputs = np.zeros((max_frames, 80))
    p2_inputs = np.zeros((max_frames, 80))
    
    for frame_idx in range(max_frames):
        p1_inputs[frame_idx] = create_action_vector(p1_frames[frame_idx])
        p2_inputs[frame_idx] = create_action_vector(p2_frames[frame_idx])
    
    replay_group.create_dataset('p1_inputs', data=p1_inputs)
    replay_group.create_dataset('p2_inputs', data=p2_inputs)
    
    replay_group.attrs['num_frames'] = max_frames
    replay_group.attrs['frame_width'] = 96
    replay_group.attrs['frame_height'] = 72

    return True  # Indicate successful processing

def create_dataset_hdf5(replay_configs, output_path):    
    with h5py.File(output_path, 'w') as f:
        f.attrs['frame_width'] = 96
        f.attrs['frame_height'] = 72
        f.attrs['action_dims'] = 80
        successful_replays = 0

        for config in tqdm(replay_configs, desc="Creating dataset"):
            frame_folder = os.path.join(FRAME_DUMP_PATH, f"frames_{config['replay_id']}")
            
            try:
                if process_single_replay_to_hdf5(
                    f,
                    config['replay_id'],
                    frame_folder,
                    config['json_path']
                ):
                    successful_replays += 1
            except Exception as e:
                print(f"Error processing replay {config['replay_id']}: {e}")
                continue

        f.attrs['num_replays'] = successful_replays

def batch_record_replays(replay_configs, max_concurrent_recordings):
    task_queue = Queue()
    active_tasks = []
    
    for config in replay_configs:
        task_queue.put(config)

    while not task_queue.empty() or active_tasks:
        active_tasks = [t for t in active_tasks if t.is_alive()]
        
        while len(active_tasks) < max_concurrent_recordings and not task_queue.empty():
            config = task_queue.get()
            thread = threading.Thread(
                target=lambda: [
                    record_replay(config['slp_path'], config['replay_id'], config['last_frame'])
                ]
            )
            active_tasks.append(thread)
            thread.start()
        
        time.sleep(1)
import h5py
import numpy as np
import cv2
import time
import torch
from dataclasses import dataclass
from typing import List

@dataclass
class CSGOAction:
    keys: List[int]
    mouse_x: float
    mouse_y: float
    l_click: bool
    r_click: bool

def decode_action_vector(y_preds):
    # Constants from your config
    N_KEYS = 11
    N_CLICKS = 2
    N_MOUSE_X = 23
    N_MOUSE_Y = 15
    MOUSE_X_POSSIBLES = [-1000, -500, -300, -200, -100, -60, -30, -20, -10, -4, -2, 0, 2, 4, 10, 20, 30, 60, 100, 200, 300, 500, 1000]
    MOUSE_Y_POSSIBLES = [-200, -100, -50, -20, -10, -4, -2, 0, 2, 4, 10, 20, 50, 100, 200]

    keys_pred = y_preds[0:N_KEYS]
    l_click_pred = y_preds[N_KEYS:N_KEYS + 1]
    r_click_pred = y_preds[N_KEYS + 1:N_KEYS + N_CLICKS]
    mouse_x_pred = y_preds[N_KEYS + N_CLICKS:N_KEYS + N_CLICKS + N_MOUSE_X]
    mouse_y_pred = y_preds[N_KEYS + N_CLICKS + N_MOUSE_X:N_KEYS + N_CLICKS + N_MOUSE_X + N_MOUSE_Y]

    keys_pressed = []
    keys_pressed_onehot = np.round(keys_pred)
    key_map = ['W', 'A', 'S', 'D', 'SPACE', 'CTRL', 'SHIFT', '1', '2', '3', 'R']
    
    for i, pressed in enumerate(keys_pressed_onehot):
        if pressed == 1:
            keys_pressed.append(key_map[i])

    l_click = bool(np.round(l_click_pred))
    r_click = bool(np.round(r_click_pred))

    mouse_x = MOUSE_X_POSSIBLES[np.argmax(mouse_x_pred)]
    mouse_y = MOUSE_Y_POSSIBLES[np.argmax(mouse_y_pred)]

    return CSGOAction(keys_pressed, mouse_x, mouse_y, l_click, r_click)

def create_input_display(action, width=280, height=150, metadata=None):
    # Create black background
    display = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)
    
    # Display metadata if available
    y_pos = 30
    if metadata:
        cv2.putText(display, f"Resolution: {metadata['resolution']}", (10, y_pos), font, font_scale, color, thickness)
        y_pos += 30
        cv2.putText(display, f"Total Frames: {metadata['total_frames']}", (10, y_pos), font, font_scale, color, thickness)
        y_pos += 30
    
    # Display keys
    if action.keys:
        cv2.putText(display, "Keys: " + " ".join(action.keys), (10, y_pos), font, font_scale, color, thickness)
    
    # Display mouse position
    y_pos += 30
    cv2.putText(display, f"Mouse: ({action.mouse_x}, {action.mouse_y})", (10, y_pos), font, font_scale, color, thickness)
    
    # Display clicks
    y_pos += 30
    clicks = []
    if action.l_click:
        clicks.append("LEFT")
    if action.r_click:
        clicks.append("RIGHT")
    if clicks:
        cv2.putText(display, "Clicks: " + " ".join(clicks), (10, y_pos), font, font_scale, color, thickness)
    
    return display

def get_metadata(f):
    # Count total frames by checking for frame_N_x keys
    frame_count = 0
    while f'frame_{frame_count}_x' in f:
        frame_count += 1
    
    # Get resolution from first frame
    first_frame = f['frame_0_x']
    resolution = f'{first_frame.shape[1]}x{first_frame.shape[0]}'
    
    return {
        'resolution': resolution,
        'total_frames': frame_count
    }

def view_frames(filepath, num_frames=60, fps=16):
    # Create windows
    cv2.namedWindow('Game View', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Inputs', cv2.WINDOW_NORMAL)
    
    # Calculate frame display time
    frame_time = 1.0 / fps
    
    with h5py.File(filepath, 'r') as f:
        # Get metadata first
        metadata = get_metadata(f)
        print(f"Dataset Info:")
        print(f"Resolution: {metadata['resolution']}")
        print(f"Total Frames: {metadata['total_frames']}")
        print(f"Playback FPS: {fps}")
        
        while True:  # Loop forever until user quits
            for i in range(num_frames):
                # Get frame data
                frame_key = f'frame_{i}_x'
                action_key = f'frame_{i}_y'
                
                if frame_key not in f or action_key not in f:
                    break
                    
                # Load frame and convert from RGB to BGR for OpenCV
                frame = f[frame_key][:]
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Load and decode action
                action_vector = f[action_key][:]
                action = decode_action_vector(action_vector)
                
                # Create input display with metadata
                input_display = create_input_display(action, metadata=metadata)
                
                # Show frames
                cv2.imshow('Game View', frame)
                cv2.imshow('Inputs', input_display)
                
                # Wait for frame time and check for quit
                key = cv2.waitKey(1)
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                    
                time.sleep(frame_time)

if __name__ == "__main__":
    file_path = r"C:\Downloads\hdf5_dm_july2021_expert_100.hdf5"
    view_frames(file_path, num_frames=1000, fps=16)
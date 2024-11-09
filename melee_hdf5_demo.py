import h5py
import numpy as np
import cv2
import time
from dataclasses import dataclass
from typing import List
from collections import deque

@dataclass
class MeleeAction:
    buttons: List[str]
    joy_x: float
    joy_y: float
    c_x: float
    c_y: float
    trigger: float

def decode_melee_vector(y_preds):
    """Decode the 80-dimensional action vector"""
    # Constants 
    N_BUTTONS = 12
    N_ANALOG = 16
    N_TRIGGER = 4

    # Button mapping
    BUTTON_MAP = [
        'D-LEFT', 'D-RIGHT', 'D-DOWN', 'D-UP',
        'Z', 'R', 'L', 'A', 'B', 'Y', 'X', 'START'
    ]

    # Split vector into components
    buttons = y_preds[:N_BUTTONS]
    joy_x = y_preds[N_BUTTONS:N_BUTTONS + N_ANALOG]
    joy_y = y_preds[N_BUTTONS + N_ANALOG:N_BUTTONS + 2*N_ANALOG]
    c_x = y_preds[N_BUTTONS + 2*N_ANALOG:N_BUTTONS + 3*N_ANALOG]
    c_y = y_preds[N_BUTTONS + 3*N_ANALOG:N_BUTTONS + 4*N_ANALOG]
    trigger = y_preds[N_BUTTONS + 4*N_ANALOG:]

    # Decode buttons
    pressed_buttons = []
    for i, pressed in enumerate(buttons):
        if pressed > 0.5:
            pressed_buttons.append(BUTTON_MAP[i])

    # Decode analog values (convert one-hot back to value)
    def decode_analog(one_hot, min_val, max_val):
        bucket = np.argmax(one_hot)
        return min_val + (max_val - min_val) * (bucket / (len(one_hot) - 1))

    return MeleeAction(
        buttons=pressed_buttons,
        joy_x=decode_analog(joy_x, -1, 1),
        joy_y=decode_analog(joy_y, -1, 1),
        c_x=decode_analog(c_x, -0.9875, 0.9875),
        c_y=decode_analog(c_y, -1, 1),
        trigger=decode_analog(trigger, 0, 1)
    )

def create_input_display(action, player_num=1, width=280, height=200, fps=None):
    """Create visualization of Melee inputs"""
    display = np.zeros((height, width, 3), dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    color = (255, 255, 255)
    
    # Title
    y_pos = 25
    cv2.putText(display, f"Player {player_num}", (10, y_pos), font, font_scale, color, thickness)
    
    # FPS Display
    if fps is not None:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(display, fps_text, (width - 100, y_pos), font, font_scale, color, thickness)
    
    # Buttons
    y_pos += 25
    if action.buttons:
        # Split into multiple lines if too many buttons
        button_text = " ".join(action.buttons)
        if len(button_text) > 30:
            chunks = [action.buttons[i:i+3] for i in range(0, len(action.buttons), 3)]
            for chunk in chunks:
                cv2.putText(display, " ".join(chunk), (10, y_pos), font, font_scale, color, thickness)
                y_pos += 20
        else:
            cv2.putText(display, button_text, (10, y_pos), font, font_scale, color, thickness)
            y_pos += 25
    
    # Analog stick positions
    y_pos += 10
    cv2.putText(display, f"Main: ({action.joy_x:.2f}, {action.joy_y:.2f})", 
                (10, y_pos), font, font_scale, color, thickness)
    
    y_pos += 25
    cv2.putText(display, f"C-Stick: ({action.c_x:.2f}, {action.c_y:.2f})", 
                (10, y_pos), font, font_scale, color, thickness)
    
    # Trigger
    y_pos += 25
    cv2.putText(display, f"Trigger: {action.trigger:.2f}", 
                (10, y_pos), font, font_scale, color, thickness)
    
    return display

def get_metadata(f):
    frame_count = 0
    while True:
        if f'frame_{frame_count}_x' not in f:
            break
        frame_count += 1
    
    first_frame = f['frame_0_x']
    resolution = f'{first_frame.shape[1]}x{first_frame.shape[0]}'
    
    return {
        'resolution': resolution,
        'total_frames': frame_count
    }

class FrameTimer:
    def __init__(self, target_fps):
        self.target_fps = target_fps
        self.target_frame_time = 1.0 / target_fps
        self.last_frame_time = time.perf_counter()
        self.frame_times = deque(maxlen=60)  # Store last 60 frame times for FPS calculation
        
    def wait_for_next_frame(self):
        current_time = time.perf_counter()
        elapsed = current_time - self.last_frame_time
        
        # If we're running behind, don't wait
        if elapsed < self.target_frame_time:
            wait_time = self.target_frame_time - elapsed
            target_wake = current_time + wait_time
            
            while time.perf_counter() < target_wake:
                # Busy-wait for more precise timing
                pass
                
        # Record frame time for FPS calculation
        actual_frame_time = time.perf_counter() - self.last_frame_time
        self.frame_times.append(actual_frame_time)
        self.last_frame_time = time.perf_counter()
        
        # Calculate current FPS
        if self.frame_times:
            current_fps = len(self.frame_times) / sum(self.frame_times)
            return current_fps
        return self.target_fps

def view_frames(filepath, target_fps=60):
    cv2.namedWindow('Game View', cv2.WINDOW_NORMAL)
    cv2.namedWindow('P1 Inputs', cv2.WINDOW_NORMAL)
    cv2.namedWindow('P2 Inputs', cv2.WINDOW_NORMAL)
    
    timer = FrameTimer(target_fps)
    
    with h5py.File(filepath, 'r') as f:
        metadata = get_metadata(f)
        print(f"Dataset Info:")
        print(f"Resolution: {metadata['resolution']}")
        print(f"Total Frames: {metadata['total_frames']}")
        
        frame_idx = 0
        
        while frame_idx < metadata['total_frames']:
            # Get frame data
            frame_key = f'frame_{frame_idx}_x'
            p1_action_key = f'frame_{frame_idx}_p1_y'
            p2_action_key = f'frame_{frame_idx}_p2_y'
            
            # Load and display frame
            frame = f[frame_key][:]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('Game View', frame)
            
            # Load and display P1 actions
            p1_action_vector = f[p1_action_key][:]
            p1_action = decode_melee_vector(p1_action_vector)
            current_fps = timer.wait_for_next_frame()
            p1_display = create_input_display(p1_action, player_num=1, fps=current_fps)
            cv2.imshow('P1 Inputs', p1_display)
            
            # Load and display P2 actions
            p2_action_vector = f[p2_action_key][:]
            p2_action = decode_melee_vector(p2_action_vector)
            p2_display = create_input_display(p2_action, player_num=2, fps=current_fps)
            cv2.imshow('P2 Inputs', p2_display)
            
            # Add progress indicator
            progress = (frame_idx + 1) / metadata['total_frames'] * 100
            print(f"\rProgress: {progress:.1f}% (FPS: {current_fps:.1f})", end='')
            
            # Check for quit
            key = cv2.waitKey(1)
            if key == ord('q'):
                print("\nPlayback stopped by user")
                break
            
            frame_idx += 1
        
        print("\nPlayback complete")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    file_path = "melee_data.hdf5"
    view_frames(file_path)  # Default 60fps
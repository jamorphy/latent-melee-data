import cv2
import json
import numpy as np
import os
from pathlib import Path

def read_game_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    p1_frames = data['players'][0]['frames']
    p2_frames = data['players'][1]['frames']
    return p1_frames, p2_frames, data['first_frame']

def get_pressed_buttons(button_value):
    """Convert button bitfield to list of pressed buttons"""
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
    
    pressed = []
    for bit, name in buttons.items():
        if button_value & bit:
            pressed.append(name)
    
    return ' + '.join(pressed) if pressed else 'NONE'

def create_input_display(frame_data):
    """Create a black box showing current frame inputs"""
    display = np.zeros((200, 250, 3), dtype=np.uint8)  # Black box
    
    # Format inputs with decoded buttons
    important_inputs = {
        'joy_x': f"{frame_data['joy_x']:.3f}",
        'joy_y': f"{frame_data['joy_y']:.3f}",
        'c_x': f"{frame_data['c_x']:.3f}",
        'c_y': f"{frame_data['c_y']:.3f}",
        'trigger': f"{frame_data['trigger']:.3f}",
        'buttons': get_pressed_buttons(frame_data['buttons'])
    }
    
    y_pos = 30
    for field, value in important_inputs.items():
        text = f"{field}: {value}"
        cv2.putText(display, text, (10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        y_pos += 30
        
    return display

def create_frame_visualization(game_frame, frame_number, p1_data, p2_data):
    """Combine frame image with input displays"""
    # Upscale game frame to 480x360 (5x original size)
    game_frame = cv2.resize(game_frame, (480, 360), interpolation=cv2.INTER_LANCZOS4)
    
    # Create input displays
    p1_display = create_input_display(p1_data)
    p2_display = create_input_display(p2_data)
    
    # Create frame number display
    frame_display = np.zeros((50, 480, 3), dtype=np.uint8)
    cv2.putText(frame_display, f"Frame: {frame_number}", 
                (180, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Combine everything into one frame
    # Final resolution will be 980x560 (250+480+250 width, max(360,200) height + 50 for frame number)
    final_frame = np.zeros((560, 980, 3), dtype=np.uint8)
    
    # Place elements
    final_frame[100:300, 0:250] = p1_display  # P1 inputs
    final_frame[0:50, 250:730] = frame_display  # Frame number
    final_frame[100:460, 250:730] = game_frame  # Game frame
    final_frame[100:300, 730:980] = p2_display  # P2 inputs
    
    return final_frame

def create_visualization_video(frames_dir, json_path, output_path, max_frames=3600):  # 3600 frames = 60 seconds at 60fps
    p1_frames, p2_frames, first_frame = read_game_data(json_path)
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 60, (980, 560))
    
    frame_count = 0
    while frame_count < max_frames:
        frame_path = os.path.join(frames_dir, f"frame_{frame_count:06d}.png")
        if not os.path.exists(frame_path):
            break
            
        # Read and process frame
        game_frame = cv2.imread(frame_path)
        if game_frame is None:
            break
            
        # Get corresponding input data
        p1_data = p1_frames[frame_count]
        p2_data = p2_frames[frame_count]
        
        # Create visualization
        vis_frame = create_frame_visualization(game_frame, frame_count, p1_data, p2_data)
        
        # Write frame
        out.write(vis_frame)
        frame_count += 1
        
        if frame_count % 60 == 0:  # Progress update every second
            print(f"Processed {frame_count} frames")
    
    out.release()
    print(f"Video created with {frame_count} frames")

if __name__ == "__main__":
    frames_dir = "frames"  # Your frames directory
    json_path = "sample.json"  # Your JSON file
    output_path = "visualization.mp4"

    
    
    create_visualization_video(frames_dir, json_path, output_path)
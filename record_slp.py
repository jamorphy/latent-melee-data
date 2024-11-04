import subprocess
import json
import tempfile
import os
import cv2

melee_iso_path = "c:/Users/ja/Desktop/melee.iso"
dolphin_path = "C:/Users/ja/AppData/Roaming/Slippi Launcher/playback/Slippi Dolphin.exe"

def create_slippi_config(slp_path):
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

def extract_frames(video_path, start_frame, end_frame, output_dir="frames"):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    target_size = (96, 72)  # width x height
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_count > end_frame:
            break
            
        resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        output_path = os.path.join(output_dir, f"frame_{frame_count:06d}.png")
        cv2.imwrite(output_path, resized_frame)
        frame_count += 1
    
    cap.release()
    return frame_count

def play_slp_segment(slp_path, start_frame=0):
    config_path = create_slippi_config(slp_path)
    
    # TODO: Stop at last frame
    command = [
        dolphin_path,
        f"--exec={melee_iso_path}",
        f"--slippi-input={config_path}",
        "--batch",
        "--hide-seekbar"
    ]
    print("Command: ", command)
    subprocess.run(command)
    
    os.unlink(config_path)

if __name__ == "__main__":
    with open("./replays/sample.json", 'r') as f:
        data = json.load(f)
        last_frame = data['last_frame']
        
    #play_slp_segment("C:/Workspace/latent-melee/replays/sample.slp")
    
    # extract individual frames, dolphin saves frame dumps in .avi
    #frame_count = extract_frames("C:/Users/ja/Desktop/frame-data-1/Frames/framedump0.avi", 0, last_frame)
    #print(f"Extracted {frame_count} frames")
import argparse
from pathlib import Path
from hdf5_process import process_replay_folder, batch_record_replays, create_dataset_hdf5
import time

def parse_args():
    parser = argparse.ArgumentParser(description='Helper script to collect and process .slp files')
    parser.add_argument(
        '--replays-folder',
        type=str,
        required=True,
        help='Folder containing .slp replay files to process'
    )
    parser.add_argument(
        '--max-dolphin-workers',
        type=int,
        default=8,
        help='Maximum number of concurrent Dolphin instances (default: 8)'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        help='Output path for the HDF5 dataset. If not provided, skip dataset creation'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    start_time = time.time()
    try:
        replay_configs = process_replay_folder(args.replays_folder)
        
        if args.max_dolphin_workers:
            print(f"Recording {len(replay_configs)} replays with {args.max_dolphin_workers} concurrent workers")
            batch_record_replays(replay_configs, max_concurrent_recordings=args.max_dolphin_workers)
        
        if args.dataset_path:
            create_dataset_hdf5(replay_configs, args.dataset_path)
        
        elapsed = time.time() - start_time
        print(f"\nTotal processing time: {elapsed:.2f} seconds")
    except Exception as e:
        print(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
import sys
from pathlib import Path
import cv2
import numpy as np

def crop_right_half(input_path: Path, output_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    
    if output_path.suffix.lower() != ".mp4":
        output_path = output_path.with_suffix(".mp4")
    
    # Open the input video
    cap = cv2.VideoCapture(str(input_path))
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Input video: {width}x{height}, {fps} fps, {frame_count} frames")
    
    if (width, height) != (512, 1024):
        print(f"Warning: expected 512x1024 input, got {width}x{height}. Cropping rightmost 512px width and top 512px height.")
    
    # Calculate crop coordinates (right half, 512x512)
    x1 = max(0, width - 512)
    y1 = 0
    x2 = width
    y2 = min(512, height)
    
    crop_width = x2 - x1
    crop_height = y2 - y1
    
    # Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (crop_width, crop_height))
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop the frame
        cropped_frame = frame[y1:y2, x1:x2]
        
        # Write the cropped frame
        out.write(cropped_frame)
        
        frame_num += 1
        if frame_num % 30 == 0:
            print(f"Processing frame {frame_num}/{frame_count}...")
    
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Cropped right 512×512 from {input_path} → {output_path}")
    print(f"Output: {crop_width}x{crop_height}, processed {frame_num} frames")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python crop_opencv.py <input.mp4> <output.mp4>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    try:
        crop_right_half(input_path, output_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
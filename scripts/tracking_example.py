import cv2
import argparse
import os
import sys
sys.path.insert(0, os.getcwd())

from lib.tracker.utracker import build_utracker

parser = argparse.ArgumentParser(description='UMatcher Detection Example')
parser.add_argument('--template_branch', type=str, default='data/umatcher_onnx/template_branch.onnx', help='Path to template_branch.onnx')
parser.add_argument('--search_branch', type=str, default='data/umatcher_onnx/search_branch.onnx', help='Path to search_branch.onnx')
parser.add_argument('--input_path', type=str, default='data/girl_dance.mp4', help='Path to search image')
parser.add_argument('--init_roi', type=int, nargs=4, default=[547, 188, 43, 57], help='ROI in cxcywh format (cx, cy, w, h)')
parser.add_argument('--output_path', type=str, default='data/tracking_result.mp4', help='Output path')
parser.add_argument('--live_mode', action='store_true', help='Enable live mode (default is off)')
args = parser.parse_args()

def select_roi_cxcywh(frame):
    """Select ROI and convert to cxcywh format"""
    print("Please select the region to track. Press SPACE or ENTER to confirm, ESC to cancel.")
    roi = cv2.selectROI("Select ROI", frame, False)
    cv2.destroyWindow("Select ROI")
    
    if roi[2] > 0 and roi[3] > 0:  # Valid selection
        # Convert from (x, y, w, h) to (cx, cy, w, h)
        cx = roi[0] + roi[2] // 2
        cy = roi[1] + roi[3] // 2
        return [cx, cy, roi[2], roi[3]]
    return None

if __name__ == "__main__":
    # Initialize tracker with ONNX models
    Tracker = build_utracker(args.template_branch, args.search_branch, True)
    
    # Open video
    cap = cv2.VideoCapture(args.input_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {args.input_path}")
        exit()
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        exit()
    
    print("Live Mode set to", args.live_mode)
    if args.live_mode == True:
        # Live mode - select ROI interactively
        bbox = select_roi_cxcywh(frame)
        if bbox is None:
            print("No ROI selected. Exiting.")
            exit()
        else:
            print(f"Selected ROI: {bbox}")
        
        # Initialize tracker
        Tracker.init(frame, bbox)
        
        # Live tracking with display
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Track
            Tracker.track(frame)
            img_track = Tracker.draw_result(frame)
            
            # Display
            cv2.imshow('Tracking Result', img_track)
            
            # Exit on ESC
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cv2.destroyAllWindows()
    
    else:
        # Non-live mode - use init_roi and save to file
        bbox = args.init_roi
        print(f"Using initial ROI: {bbox}")
        Tracker.init(frame, bbox)
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))
        
        # Process first frame
        img_track = Tracker.draw_result(frame)
        out.write(img_track)
        
        # Process remaining frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Track
            Tracker.track(frame)
            img_track = Tracker.draw_result(frame)
            
            # Write to output
            out.write(img_track)
        
        out.release()
        print(f"Tracking result saved to {args.output_path}")
    
    cap.release()


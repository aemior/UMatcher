import cv2
import argparse
import os
import sys
sys.path.insert(0, os.getcwd())

from lib.detector.udetector import build_udetector

parser = argparse.ArgumentParser(description='UMatcher Detection Example')
parser.add_argument('--template_branch', type=str, default='data/umatcher_onnx/template_branch.onnx', help='Path to template_branch.onnx')
parser.add_argument('--search_branch', type=str, default='data/umatcher_onnx/search_branch.onnx', help='Path to search_branch.onnx')
parser.add_argument('--template_img', type=str, default='data/test_1.png', help='Path to template image')
parser.add_argument('--live_mode', action='store_true', help='Enable live mode (default is off)')
parser.add_argument('--template_roi', type=int, nargs=4, default=[110, 233, 52, 99], help='ROI in cxcywh format (cx, cy, w, h)')
parser.add_argument('--search_img', type=str, default='data/test_1.png', help='Path to search image')
parser.add_argument('--output_path', type=str, default='data/detection_result.png', help='Output path')
args = parser.parse_args()

if __name__ == "__main__":
    # Initialize detector with ONNX models
    Detector = build_udetector(args.template_branch, args.search_branch, True)

    template_img = cv2.imread(args.template_img)
    # Use OpenCV to interactively select ROI
    if args.live_mode:
        print("Select a ROI on the template image and press ENTER when done (ESC to cancel)")
        x, y, w, h = cv2.selectROI("Template ROI Selection", template_img, fromCenter=False, showCrosshair=True)
        
        # Convert from (x,y,w,h) to (cx,cy,w,h) format
        cx = x + w/2
        cy = y + h/2
        template_roi = [int(cx), int(cy), w, h]
        
        print("Selected ROI in cxcywh format:", template_roi)
        cv2.destroyWindow("Template ROI Selection")
    else:
        template_roi = args.template_roi

    Detector.set_template(template_img, template_roi)
    temp_embedding = Detector.template_embedding.flatten().tolist()
    print(temp_embedding)

    search_img = cv2.imread(args.search_img)
    # all_boxes, all_scores = Detector.detect(search_img, threshold=0.3, pyramid=[0.7, 1.3])
    all_boxes, all_scores = Detector.detect(search_img, threshold=0.3)
    result = Detector.draw_bboxes(search_img, all_boxes, all_scores)

    if args.live_mode:
        cv2.imshow("Detection Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(args.output_path, result)

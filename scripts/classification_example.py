import cv2
import json
import argparse
import os
import sys
sys.path.insert(0, os.getcwd())

from lib.detector.udetector import build_udetector
import numpy as np
from collections import Counter

parser = argparse.ArgumentParser(description='UMatcher Detection Example')
parser.add_argument('--template_branch', type=str, default='data/umatcher_onnx/template_branch.onnx', help='Path to template_branch.onnx')
parser.add_argument('--search_branch', type=str, default='data/umatcher_onnx/search_branch.onnx', help='Path to search_branch.onnx')
parser.add_argument('--data_set', type=str, default='data/classfication_dataset.json', help='Path to classfication dataset (json annotated file)')
parser.add_argument('--live_mode', action='store_true', help='Enable live mode (default is off)')
parser.add_argument('--target_roi', type=int, nargs=4, default=[320, 266, 57, 49], help='ROI in cxcywh format (cx, cy, w, h)')
parser.add_argument('--search_img', type=str, default='data/test_4.png', help='Path to search image')
parser.add_argument('--output_path', type=str, default='data/classification_result.png', help='Output path')
args = parser.parse_args()

def build_classification_dataset(detector, data_set, live_mode=False):
    """
    Build classification dataset from json file
    """
    with open(data_set, 'r') as f:
        dataset = json.load(f)
    template_embeddings = {}
    for single_image in dataset:
        img_path = single_image['image_path']
        annotations = single_image['annotations']
        img = cv2.imread(img_path)
        for annotation in annotations:
            detector.set_template(img, annotation['bbox'])
            template_embedding = detector.get_template_embedding()
            if annotation['label'] not in template_embeddings.keys():
                template_embeddings[annotation['label']] = []
            template_embeddings[annotation['label']].append(template_embedding)
        if live_mode:
            for annotation in annotations:
                cx, cy, w, h = annotation['bbox']
                cv2.rectangle(img, (int(cx-w/2), int(cy-h/2)), (int(cx+w/2), int(cy+h/2)), (0, 255, 0), 2)
                cv2.putText(img, annotation['label'], (int(cx-w/2), int(cy-h/2)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            # cv2.imwrite("debug.png", img)
            print("Press any key to continue...")
            cv2.imshow("Label Image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return template_embeddings

if __name__ == "__main__":
    # Initialize detector with ONNX models
    Detector = build_udetector(args.template_branch, args.search_branch, True)

    # Build classification dataset
    dataset = build_classification_dataset(Detector, args.data_set, args.live_mode)

    if args.live_mode:
        # Load search image for ROI selection
        search_img_for_roi = cv2.imread(args.search_img)
        print("Please select the target ROI in the image window...")
        
        # Use selectROI to get bounding box in (x, y, w, h) format
        bbox = cv2.selectROI("Select Target ROI", search_img_for_roi, False)
        cv2.destroyAllWindows()
        
        # Convert from (x, y, w, h) to (cx, cy, w, h) format
        x, y, w, h = bbox
        cx = x + w // 2
        cy = y + h // 2
        target_roi = [cx, cy, w, h]
        
        print(f"Selected ROI (cxcywh): {target_roi}")
    else:
        target_roi = args.target_roi
    # Uncomment the following line to use other target
    # target_roi = [737, 205, 41, 110]
    # target_roi = [336, 268, 51, 126]
    search_img = cv2.imread(args.search_img)
    Detector.set_template(search_img, target_roi)
    target_embedding = Detector.get_template_embedding()

    # Define k for KNN
    k = 3

    # Calculate similarities and store label-distance pairs
    similarities = []
    for label, embeddings in dataset.items():
        for embedding in embeddings:
            # Calculate cosine similarity (the shape of embedding is (1, n, 1, 1))
            target_flat = target_embedding.reshape(1, -1)[0]
            embedding_flat = embedding.reshape(1, -1)[0]
            sim = np.dot(target_flat, embedding_flat) / (np.linalg.norm(target_flat) * np.linalg.norm(embedding_flat))
            similarities.append((label, sim))

    # Sort similarities in descending order (higher cosine similarity = more similar)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Take top k neighbors
    top_k_neighbors = similarities[:k]
    neighbor_labels = [neighbor[0] for neighbor in top_k_neighbors]

    # Find the most common label
    predicted_label = Counter(neighbor_labels).most_common(1)[0][0]
    
    # Get the highest score for the predicted label
    predicted_score = max([score for label, score in top_k_neighbors if label == predicted_label])

    # Print results
    print(f"Predicted label: {predicted_label}")
    print(f"Top {k} neighbors: {top_k_neighbors}")

    # Draw shadow for the target ROI (offset by 2 pixels bottom-right)
    cx, cy, w, h = target_roi
    cv2.rectangle(search_img, (int(cx-w/2)+2, int(cy-h/2)+2), (int(cx+w/2)+2, int(cy+h/2)+2), (255, 150, 150), 2)
    
    # Draw the target ROI on the search image
    cv2.rectangle(search_img, (int(cx-w/2), int(cy-h/2)), (int(cx+w/2), int(cy+h/2)), (76, 87, 244), 2)  # Material Design Red 500
    
    # Display k neighbors with scores, best on top
    for i, (label, score) in enumerate(top_k_neighbors):
        text = f"{i+1}. {label} ({score:.3f})"
        y_offset = int(cy+h/2) + 25 + (i * 20)  # Stack vertically below ROI
        
        # Get text size for background box
        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Material Design Red colors
        bg_color = (92, 107, 239)  # Material Design Red 400 (BGR)
        shadow_color = (255, 150, 150)  # Custom shadow color (BGR)
        text_color = (255, 255, 255)  # White text
        
        # Calculate box coordinates
        box_x1 = int(cx-w/2)
        box_y1 = y_offset - text_height - 2
        box_x2 = box_x1 + text_width + 8
        box_y2 = y_offset + baseline + 2
        
        # Draw shadow (offset by 2 pixels bottom-right)
        cv2.rectangle(search_img, (box_x1 + 2, box_y1 + 2), (box_x2 + 2, box_y2 + 2), shadow_color, -1)
        
        # Draw background box
        cv2.rectangle(search_img, (box_x1, box_y1), (box_x2, box_y2), bg_color, -1)
        
        # Draw text
        cv2.putText(search_img, text, (box_x1 + 4, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

    # Display the result if in live mode
    if args.live_mode:
        cv2.imshow("Detection Result", search_img)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # Save the result
        cv2.imwrite(args.output_path, search_img)
        print(f"Result saved to {args.output_path}")


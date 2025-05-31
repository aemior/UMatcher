import os
import numpy as np
import cv2
import onnxruntime as ort

from lib.utils.imgproc import center_crop

class UDetector:
    def __init__(self, template_branch_path, search_branch_path, half=False, template_scale=2, template_size=128, search_scale=4, search_size=256, stride=16):
        self.half = half
        
        # Setup ONNX session options
        sess_options = ort.SessionOptions()
        
        # Set providers with CUDA if available
        providers = ['CPUExecutionProvider']
        
        # Load ONNX models
        self.template_branch = ort.InferenceSession(template_branch_path, sess_options, providers=providers)
        self.search_branch = ort.InferenceSession(search_branch_path, sess_options, providers=providers)
        
        # Get input and output names
        self.template_input_name = self.template_branch.get_inputs()[0].name
        self.template_output_name = self.template_branch.get_outputs()[0].name
        
        self.search_input_names = [input.name for input in self.search_branch.get_inputs()]
        self.search_output_names = [output.name for output in self.search_branch.get_outputs()]
        
        # Default sizes
        self.template_size = template_size
        self.search_size = search_size
        self.template_scale = template_scale
        self.search_scale = search_scale
        self.stride = stride
        self.feat_sz = search_size // stride
        
        self.template_embedding = None

    def set_template(self, template, bbox):
        bbox = [int(x) for x in bbox] 
        template_img = center_crop(template, bbox, self.template_scale)
        self.scale_factor = self.template_size / template_img.shape[0]
        template_img = cv2.resize(template_img, (self.template_size, self.template_size))
        self.template_image = template_img.copy()
        
        # Prepare input for ONNX
        template_img = template_img.astype(np.float32).transpose(2, 0, 1) / 255.0
        if self.half:
            template_img = template_img.astype(np.float16)
        
        # Add batch dimension
        template_img = np.expand_dims(template_img, axis=0)
        
        # Run ONNX inference
        self.template_embedding = self.template_branch.run(
            [self.template_output_name], 
            {self.template_input_name: template_img}
        )[0]

        # Normalize to unit vector
        norm = np.linalg.norm(self.template_embedding)
        self.template_embedding = self.template_embedding / (norm + 1e-12)
        
        # Convert to float16 if half precision is enabled
        if self.half:
            self.template_embedding = self.template_embedding.astype(np.float16)

    def detect(self, input_img, threshold=0.5, pyramid=[0.7, 1.0, 1.3], overlap=0.5, debug=True):
        
        all_boxes = []
        all_scores = []
        original_h, original_w = input_img.shape[:2]

        # Create debug directory only if debug is enabled
        if debug:
            debug_dir = "debug_windows"
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)

        for scale_idx, scale in enumerate(pyramid):
            scale = self.scale_factor * scale

            scaled_w = int(original_w * scale)
            scaled_h = int(original_h * scale)

            # Skip if both dimensions are smaller than search size
            if scaled_w < self.search_size and scaled_h < self.search_size:
                continue

            scaled_img = cv2.resize(input_img, (scaled_w, scaled_h))

            # Pad the image to ensure full coverage
            pad_w = max(0, self.search_size - scaled_w)
            pad_h = max(0, self.search_size - scaled_h)
            
            if pad_w > 0 or pad_h > 0:
                # Pad with black pixels
                padded_img = np.zeros((scaled_h + pad_h, scaled_w + pad_w, scaled_img.shape[2]), dtype=scaled_img.dtype)
                padded_img[:scaled_h, :scaled_w] = scaled_img
                scaled_img = padded_img
                scaled_w += pad_w
                scaled_h += pad_h

            # Create folder for this pyramid scale only if debug is enabled
            if debug:
                scale_dir = os.path.join(debug_dir, f"scale_{scale_idx}_{scale:.2f}")
                if not os.path.exists(scale_dir):
                    os.makedirs(scale_dir)

            step = int(self.search_size * (1 - overlap))
            step = max(1, step)

            x_starts = []
            current_x = 0
            while current_x <= scaled_w - self.search_size:
                x_starts.append(current_x)
                current_x += step
            if x_starts and x_starts[-1] + self.search_size < scaled_w:
                x_starts.append(scaled_w - self.search_size)
            elif not x_starts and scaled_w >= self.search_size:
                x_starts.append(0)

            y_starts = []
            current_y = 0
            while current_y <= scaled_h - self.search_size:
                y_starts.append(current_y)
                current_y += step
            if y_starts and y_starts[-1] + self.search_size < scaled_h:
                y_starts.append(scaled_h - self.search_size)
            elif not y_starts and scaled_h >= self.search_size:
                y_starts.append(0)

            window_idx = 0
            for x_start in x_starts:
                for y_start in y_starts:
                    window = scaled_img[y_start:y_start+self.search_size, x_start:x_start+self.search_size, :]
                    
                    # Save window image for debug only if debug is enabled
                    if debug:
                        window_filename = os.path.join(scale_dir, f"window_{window_idx}_x{x_start}_y{y_start}.jpg")
                        cv2.imwrite(window_filename, window)
                    window_idx += 1
                    
                    pred_bboxes, scores = self.search(window)

                    for bbox, score in zip(pred_bboxes, scores):
                        cx, cy, w, h = bbox
                        scaled_cx = cx + x_start
                        scaled_cy = cy + y_start

                        # Account for padding when converting back to original coordinates
                        original_cx = scaled_cx / scale
                        original_cy = scaled_cy / scale
                        original_w_bbox = w / scale
                        original_h_bbox = h / scale

                        x1 = original_cx - original_w_bbox / 2
                        y1 = original_cy - original_h_bbox / 2
                        x2 = original_cx + original_w_bbox / 2
                        y2 = original_cy + original_h_bbox / 2

                        # Clamp to original image boundaries
                        x1 = max(0, min(original_w, x1))
                        y1 = max(0, min(original_h, y1))
                        x2 = max(0, min(original_w, x2))
                        y2 = max(0, min(original_h, y2))

                        if x2 <= x1 or y2 <= y1:
                            continue

                        all_boxes.append([x1, y1, x2, y2])
                        all_scores.append(score)

        if not all_boxes:
            return np.empty((0, 4)), np.empty((0,))

        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)

        mask = all_scores >= threshold
        filtered_boxes = all_boxes[mask]
        filtered_scores = all_scores[mask]

        if len(filtered_boxes) == 0:
            return np.empty((0, 4)), np.empty((0,))

        # Convert to the format expected by cv2.dnn.NMSBoxes
        boxes_for_nms = filtered_boxes.astype(np.float32)
        scores_for_nms = filtered_scores.astype(np.float32)
        
        # Apply NMS using OpenCV
        indices = cv2.dnn.NMSBoxes(boxes_for_nms, scores_for_nms, threshold, 0.5)
        
        # OpenCV >= 4.5.4 returns a flat array
        if len(indices.shape) == 1:
            indices = indices
        else:
            indices = indices.flatten()
            
        final_boxes = filtered_boxes[indices]
        final_scores = filtered_scores[indices]

        return final_boxes, final_scores

    def get_all_bbox(self, score_map, size_map, offset_map, thr=0.1):
        # Flatten the score map
        flat_score_map = score_map.flatten()
        
        # Reshape size_map and offset_map for indexing
        size_map_flat = size_map.reshape(size_map.shape[0], size_map.shape[1], -1)
        offset_map_flat = offset_map.reshape(offset_map.shape[0], offset_map.shape[1], -1)
        
        bboxes = []
        scores = []
        
        # Find indices where scores are above threshold
        above_threshold = np.where(flat_score_map > thr)[0]
        
        for idx in above_threshold:
            # Get size and offset for current detection
            size = size_map_flat[0, :, idx]
            offset = offset_map_flat[0, :, idx]
            
            # Compute bbox coordinates (cx, cy, w, h)
            idx_y = idx // self.feat_sz
            idx_x = idx % self.feat_sz
            cx = (idx_x + offset[0]) / self.stride
            cy = (idx_y + offset[1]) / self.stride
            w = size[0]
            h = size[1]
            
            bboxes.append([float(cx), float(cy), float(w), float(h)])
            scores.append(float(flat_score_map[idx]))
            
        return np.array(bboxes) if bboxes else np.empty((0, 4)), np.array(scores)

    def search(self, search_img):
        # Process search image
        search_img = search_img.astype(np.float32).transpose(2, 0, 1) / 255.0
        if self.half:
            search_img = search_img.astype(np.float16)
        
        # Add batch dimension
        search_img = np.expand_dims(search_img, axis=0)
        
        # Prepare inputs for ONNX model
        inputs = {
            self.search_input_names[0]: search_img,
            self.search_input_names[1]: self.template_embedding
        }
        
        # Run inference
        outputs = self.search_branch.run(self.search_output_names, inputs)

        # Extract bboxes
        pred_bboxes, scores = self.get_all_bbox(outputs[0], outputs[1], outputs[2], thr=0.1)
        
        return pred_bboxes * self.search_size, scores

    def draw_bboxes(self, img, bboxes, scores):
        for box, score in zip(bboxes, scores):
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f"{score:.2f}", (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img

    def get_template_embedding(self):
        return self.template_embedding

def build_udetector(template_branch_path, search_branch_path, half=False):
    return UDetector(template_branch_path, search_branch_path, half)
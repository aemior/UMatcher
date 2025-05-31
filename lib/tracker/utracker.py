import numpy as np
import cv2
import onnxruntime as ort

from lib.utils.imgproc import center_crop

class UTracker:
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

        self.alpha_kf = 0.5           # weight for KF-IoU
        self.tau_kf = 10               # threshold frames to use KF
        self.success_count = 0             # consecutive successful updates

        # Initialize OpenCV KalmanFilter: 8 state dims, 4 measurement dims
        self.kf = cv2.KalmanFilter(8, 4)
        # State: [x, y, w, h, vx, vy, vw, vh]
        # Measurement: [x, y, w, h]
        # Transition matrix F
        dt = 1.0
        self.kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.kf.transitionMatrix[i, i+4] = dt
        # Measurement matrix H
        self.kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        self.kf.measurementMatrix[0, 0] = 1
        self.kf.measurementMatrix[1, 1] = 1
        self.kf.measurementMatrix[2, 2] = 1
        self.kf.measurementMatrix[3, 3] = 1
        # Process and measurement noise
        cv2.setIdentity(self.kf.processNoiseCov, 1e-2)
        cv2.setIdentity(self.kf.measurementNoiseCov, 1e-1)
        cv2.setIdentity(self.kf.errorCovPost, 1.0)

    def init(self, img, bbox):
        self.update_template(img, bbox)
        self.last_pos = bbox
        self.last_score = 1.0
        cx, cy, w, h = bbox
        # Initialize Kalman state
        self.kf.statePost = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], dtype=np.float32)
        self.kf.statePre = self.kf.statePost.copy()
        self.success_count = 0

    def update_template(self, template, bbox):
        bbox = [int(x) for x in bbox] 
        template_img = center_crop(template, bbox, self.template_scale)
        template_img = cv2.resize(template_img, (self.template_size, self.template_size))
        self.template_image = template_img.copy()
        
        # Prepare input for ONNX
        template_img = template_img.astype(np.float32).transpose(2, 0, 1) / 255.0
        if self.half:
            template_img = template_img.astype(np.float16)
        
        # Add batch dimension
        template_img = np.expand_dims(template_img, axis=0)
        
        # Run ONNX inference
        template_embedding = self.template_branch.run(
            [self.template_output_name], 
            {self.template_input_name: template_img}
        )[0]
        
        # Normalize to unit vector
        norm = np.linalg.norm(template_embedding)
        template_embedding = template_embedding / (norm + 1e-12)



        if self.template_embedding is None:
            self.template_embedding = template_embedding
        else:
            # Combine and normalize embeddings
            combined_embedding = self.template_embedding + template_embedding
            # Normalize to unit vector
            norm = np.linalg.norm(combined_embedding)
            if norm > 0:
                self.template_embedding = combined_embedding / norm
            else:
                self.template_embedding = combined_embedding

        # self.template_embedding = self.template_branch.run(
            # [self.template_output_name], 
            # {self.template_input_name: template_img}
        # )[0]
        
        # Convert to float16 if half precision is enabled
        if self.half:
            self.template_embedding = self.template_embedding.astype(np.float16)

    def trans_pose(self, pred_bbox, w_i, h_i):
        cx, cy, w, h = pred_bbox
        offset_x, offset_y, w, h = (cx-0.5) * w_i, (cy-0.5) * h_i, w * w_i, h * h_i
        return [self.last_pos[0] + offset_x, self.last_pos[1] + offset_y, w, h]

    def track(self, search):
        search_img = center_crop(search, self.last_pos, self.search_scale)
        w_i, h_i = search_img.shape[1], search_img.shape[0]
        search_img = cv2.resize(search_img, (self.search_size, self.search_size))
        search_img = search_img.astype(np.float32).transpose(2, 0, 1) / 255.0
        if self.half:
            search_img = search_img.astype(np.float16)
        search_img = np.expand_dims(search_img, axis=0)
        inputs = {
            self.search_input_names[0]: search_img,
            self.search_input_names[1]: self.template_embedding
        }
        outputs = self.search_branch.run(self.search_output_names, inputs)
        pred_bboxes, scores = self.get_all_bbox(outputs[0], outputs[1], outputs[2], thr=0.1)

        self.last_candidates = []
        for box, score in zip(pred_bboxes, scores):
            self.last_candidates.append([score] + self.trans_pose(box, w_i, h_i))

        self.last_pos, self.last_score = self.match_pos(self.last_candidates)

        return self.last_pos

    def match_pos(self, detections):
        """
        Choose the most confident detection among candidates using KF-IoU fusion.
        Detections: list of tuples (score, cx, cy, w, h)
        Returns: best_bbox [score, cx, cy, w, h]
        """
        if not detections:
            self.success_count = 0
            return self.last_pos, 0.0

        # Compute IoU between prediction and candidates
        def iou(boxA, boxB):
            # box = [cx, cy, w, h]
            xA1, yA1 = boxA[0] - boxA[2]/2, boxA[1] - boxA[3]/2
            xA2, yA2 = boxA[0] + boxA[2]/2, boxA[1] + boxA[3]/2
            xB1, yB1 = boxB[0] - boxB[2]/2, boxB[1] - boxB[3]/2
            xB2, yB2 = boxB[0] + boxB[2]/2, boxB[1] + boxB[3]/2
            ix1, iy1 = max(xA1, xB1), max(yA1, yB1)
            ix2, iy2 = min(xA2, xB2), min(yA2, yB2)
            iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
            inter = iw * ih
            union = boxA[2]*boxA[3] + boxB[2]*boxB[3] - inter
            return inter/union if union>0 else 0

        use_kf = (self.success_count >= self.tau_kf)
        if use_kf:
            # Predict next state
            pred = self.kf.predict()
            px, py, pw, ph = pred[0,0], pred[1,0], pred[2,0], pred[3,0]
        best_score = -1
        best_det = None
        for det in detections:
            score, cx, cy, w, h = det
            if use_kf:
                kf_iou = iou((px,py,pw,ph), (cx,cy,w,h))
                combined = self.alpha_kf * kf_iou + (1 - self.alpha_kf) * score
            else:
                combined = score
            if combined > best_score:
                best_score = combined
                best_det = det


        # Update success counter
        if best_det and best_score > 0.2:
            self.success_count += 1
            # Correct with measurement
            _, cx, cy, w, h = best_det
            measurement = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
            self.kf.correct(measurement)
        else:
            self.success_count = 0
            return self.last_pos, 0.0

        return best_det[1:], best_score
        

        

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

    def get_template_embedding(self):
        return self.template_embedding

    def draw_result(self, img):
        """
        Draw bounding box and score/MISS indicator on the image.
        - Material Design style with shadows
        - Score shown with colored background and white text
        - Only corner lines with shadows, no complete rectangle
        """
        if not hasattr(self, 'last_pos'):
            return img

        x, y, w, h = self.last_pos
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)

        # Choose colors based on tracking status
        if hasattr(self, 'last_score') and self.last_score > 0:
            box_color = (76, 175, 80)  # Material Green 500 (BGR)
            bg_color = (76, 175, 80)   # Green background
        else:
            box_color = (67, 56, 202)  # Material Red 500 (BGR)
            bg_color = (67, 56, 202)   # Red background

        shadow_color = (255, 150, 150)  # Black shadow
        text_color = (255, 255, 255)  # White text

        # Draw corner lines with shadows
        corner_length = int(min(30, w // 6, h // 6))
        thickness = 3
        shadow_offset = 2

        # Shadow for corners (offset to bottom-right)
        # Top-left corner shadow
        cv2.line(img, (x1 + shadow_offset, y1 + shadow_offset), 
                 (x1 + corner_length + shadow_offset, y1 + shadow_offset), shadow_color, thickness)
        cv2.line(img, (x1 + shadow_offset, y1 + shadow_offset), 
                 (x1 + shadow_offset, y1 + corner_length + shadow_offset), shadow_color, thickness)
        
        # Top-right corner shadow
        cv2.line(img, (x2 + shadow_offset, y1 + shadow_offset), 
                 (x2 - corner_length + shadow_offset, y1 + shadow_offset), shadow_color, thickness)
        cv2.line(img, (x2 + shadow_offset, y1 + shadow_offset), 
                 (x2 + shadow_offset, y1 + corner_length + shadow_offset), shadow_color, thickness)
        
        # Bottom-left corner shadow
        cv2.line(img, (x1 + shadow_offset, y2 + shadow_offset), 
                 (x1 + corner_length + shadow_offset, y2 + shadow_offset), shadow_color, thickness)
        cv2.line(img, (x1 + shadow_offset, y2 + shadow_offset), 
                 (x1 + shadow_offset, y2 - corner_length + shadow_offset), shadow_color, thickness)
        
        # Bottom-right corner shadow
        cv2.line(img, (x2 + shadow_offset, y2 + shadow_offset), 
                 (x2 - corner_length + shadow_offset, y2 + shadow_offset), shadow_color, thickness)
        cv2.line(img, (x2 + shadow_offset, y2 + shadow_offset), 
                 (x2 + shadow_offset, y2 - corner_length + shadow_offset), shadow_color, thickness)

        # Main corner lines
        # Top-left corner
        cv2.line(img, (x1, y1), (x1 + corner_length, y1), box_color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + corner_length), box_color, thickness)
        
        # Top-right corner
        cv2.line(img, (x2, y1), (x2 - corner_length, y1), box_color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + corner_length), box_color, thickness)
        
        # Bottom-left corner
        cv2.line(img, (x1, y2), (x1 + corner_length, y2), box_color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - corner_length), box_color, thickness)
        
        # Bottom-right corner
        cv2.line(img, (x2, y2), (x2 - corner_length, y2), box_color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - corner_length), box_color, thickness)

        # Display score or MISS with background
        if hasattr(self, 'last_score') and self.last_score is not None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_thickness = 2
            
            if self.last_score > 0:
                text = f"Score: {self.last_score:.3f}"
            else:
                text = "MISS"
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
            
            # Background rectangle coordinates
            bg_x1, bg_y1 = 10, 10
            bg_x2, bg_y2 = bg_x1 + text_width + 10, bg_y1 + text_height + 10
            
            # Draw background shadow (offset to bottom-right)
            shadow_offset = 3
            cv2.rectangle(img, (bg_x1 + shadow_offset, bg_y1 + shadow_offset), 
                         (bg_x2 + shadow_offset, bg_y2 + shadow_offset), shadow_color, -1)
            
            # Draw background rectangle
            cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
            
            # Draw white text on colored background
            text_x = bg_x1 + 5
            text_y = bg_y1 + text_height + 5
            cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, text_thickness)

        return img

def build_utracker(template_branch_path, search_branch_path, half=False):
    return UTracker(template_branch_path, search_branch_path, half)
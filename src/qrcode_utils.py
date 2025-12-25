import cv2
import numpy as np
import os
from typing import Optional, Tuple, List

class QRCodeRegistrator:
    def __init__(self, model_dir: str = "wqmodules"):
        """
        Initialize WeChat QR Code Detector.
        
        Args:
            model_dir: Directory containing the WeChat QR code models.
                       Expected files: detect.prototxt, detect.caffemodel, sr.prototxt, sr.caffemodel
        """
        self.detector = self._load_detector(model_dir)
        
    def _load_detector(self, model_dir: str):
        try:
            detect_proto = os.path.join(model_dir, "detect.prototxt")
            detect_caffe = os.path.join(model_dir, "detect.caffemodel")
            sr_proto = os.path.join(model_dir, "sr.prototxt")
            sr_caffe = os.path.join(model_dir, "sr.caffemodel")
            
            # check if files exist
            if not all(os.path.exists(p) for p in [detect_proto, detect_caffe, sr_proto, sr_caffe]):
                print(f"Warning: WeChat QR models not found in {model_dir}. QR detection will fail.")
                return None
                
            detector = cv2.wechat_qrcode_WeChatQRCode(
                detect_proto, detect_caffe, sr_proto, sr_caffe
            )
            return detector
        except Exception as e:
            print(f"Failed to load WeChat QR detector: {e}")
            return None

    def detect(self, img: np.ndarray) -> Tuple[List[str], List[np.ndarray]]:
        """
        Detect QR codes in an image.
        
        Returns:
            res: List of decoded strings
            points: List of point arrays (4 corner points per QR code)
        """
        if self.detector is None:
            return [], []
        
        res, points = self.detector.detectAndDecode(img)
        return res, points

    def get_fourth_channel(self, original_img: np.ndarray, register_dir: str) -> np.ndarray:
        """
        Generate the 4th channel by registering a reference image.
        
        Args:
            original_img: Input image (H, W, 3) in BGR or RGB format (OpenCV defaults BGR)
            register_dir: Directory containing reference images
            
        Returns:
            registered_channel: (H, W) single channel image (grayscale), 
                                matched and warped from reference, or black if failed.
        """
        h, w = original_img.shape[:2]
        
        # Default to black channel
        black_channel = np.zeros((h, w), dtype=np.uint8)
        
        # 1. Detect QR code in original image
        res, points = self.detect(original_img)
        
        if not res or len(res) == 0:
            return black_channel
        
        # Assume the main QR code is the first one found or the one we care about.
        # Use the first valid one.
        qr_content = res[0]
        qr_points_orig = points[0]
        
        # 2. Find reference image
        ref_path = os.path.join(register_dir, f"{qr_content}.jpg")
        if not os.path.exists(ref_path):
            # Try to handle potential filename issues (unsafe chars) if needed, 
            # but for now assume direct mapping
            # print(f"Reference image not found: {ref_path}")
            return black_channel
            
        ref_img = cv2.imread(ref_path)
        if ref_img is None:
            return black_channel
            
        # 3. Detect QR code in reference image
        res_ref, points_ref_list = self.detect(ref_img)
        
        if not res_ref:
            return black_channel
            
        # Find which detected QR in ref matches the content (though usually ref only has one)
        qr_points_ref = None
        for i, r_content in enumerate(res_ref):
            if r_content == qr_content:
                qr_points_ref = points_ref_list[i]
                break
        
        if qr_points_ref is None:
            # Fallback: if content didn't match perfectly, maybe just use the first one 
            # if we trust the filename matching
            qr_points_ref = points_ref_list[0]
            
        # 4. Calculate Homography and Warp
        try:
            # points are lists of numpy arrays. Need float32 for warp
            # WeChat detector returns points as list of [array([[x,y], [x,y]...])] 
            # or simply list of arrays. Let's verify shape. 
            # points[0] is usually (4, 2) float32
            
            src_pts = qr_points_ref.reshape(-1, 2).astype(np.float32)
            dst_pts = qr_points_orig.reshape(-1, 2).astype(np.float32)
            
            if len(src_pts) != 4 or len(dst_pts) != 4:
                return black_channel
                
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            
            # Warp reference image to match original image perspective
            # Use nearest neighbor or linear. Linear is better for visual, 
            # but we want structure. Linear is fine.
            warped_ref = cv2.warpPerspective(ref_img, M, (w, h))
            
            # 5. Convert to Grayscale to be the 4th channel
            warped_gray = cv2.cvtColor(warped_ref, cv2.COLOR_BGR2GRAY)
            
            return warped_gray
            
        except Exception as e:
            print(f"Registration failed: {e}")
            return black_channel

"""
PRODUCTION-GRADE QATM - PROPERLY FIXED VERSION
- Stricter confidence thresholds
- Better NMS to remove duplicates
- Cleaner visualization
- Accurate tracking
"""

import cv2
import numpy as np
from glob import glob
from math import atan, degrees
from datetime import datetime
from inria_qatm_pytorch_v2_copy import *
import argparse
from skimage.draw import line
from noise import noise, blur
from cam_sim import SimCamera
import torchvision
from time import time, sleep
import warnings
warnings.filterwarnings('ignore')

FPS = 30
source_image_scale_factor = 1
select_points = True

def mouse_click(event, x, y, flags, param):
    """Handle mouse clicks for path selection"""
    global scaled_image, points_array, source_image_scale_factor, line_draw_array, select_points
    if select_points:
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(scaled_image, (x, y), 5, (0, 0, 255), 4)
            points_array.append([x * source_image_scale_factor, y * source_image_scale_factor])
            line_draw_array.append([x, y])
            if len(points_array) > 1:
                delta_x = line_draw_array[-2][0] - line_draw_array[-1][0]
                delta_y = line_draw_array[-2][1] - line_draw_array[-1][1]
                if delta_x != 0:
                    ang = degrees(atan(delta_y / delta_x))
                else:
                    ang = 90.0
                cv2.line(scaled_image, tuple(line_draw_array[-2]), tuple(line_draw_array[-1]), (0, 255, 0), 2)
            cv2.imshow("point_selector", scaled_image)

def extract_roi_around_detection(image_raw, center_x, center_y, roi_size, margin):
    """Extract ROI with safety checks"""
    h, w = image_raw.shape[:2]
    half_roi = roi_size // 2
    
    x1 = max(0, int(center_x - half_roi - margin))
    y1 = max(0, int(center_y - half_roi - margin))
    x2 = min(w, int(center_x + half_roi + margin))
    y2 = min(h, int(center_y + half_roi + margin))
    
    if (x2 - x1) < 100 or (y2 - y1) < 100:
        x1 = max(0, center_x - 400)
        y1 = max(0, center_y - 400)
        x2 = min(w, center_x + 400)
        y2 = min(h, center_y + 400)
    
    roi_image = image_raw[y1:y2, x1:x2]
    return roi_image, (x1, y1), (x1, y1, x2, y2)

def map_boxes_to_global(boxes, offset_x, offset_y):
    """Map boxes to global coordinates"""
    if boxes is None or len(boxes) == 0:
        return boxes
    boxes_global = boxes.copy()
    boxes_global[:, :, 0] += offset_x
    boxes_global[:, :, 1] += offset_y
    return boxes_global

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Production QATM - Fixed')
    
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--resize', '-r', type=int, default=100)
    parser.add_argument('--crop_size', '-cs', type=int, default=150)
    parser.add_argument('--alpha', '-a', type=float, default=25)
    parser.add_argument('--fps', '-f', type=int, default=3)
    parser.add_argument('--scale_factor', '-sf', type=int, default=1)
    # ✅ FIX 1: Higher threshold to reject false positives
    parser.add_argument('--thres', '-t', type=float, default=0.90)
    parser.add_argument('--source', '-s', type=str, required=True)
    parser.add_argument('--noise', '-n', type=str, default='none')
    parser.add_argument('--blur', '-b', type=str, default='none')
    parser.add_argument('--blur_filter', '-bf', type=int, default=5)
    parser.add_argument('--use_localization', '-ul', action='store_true', default=True)
    parser.add_argument('--roi_size', '-rs', type=int, default=500)
    parser.add_argument('--roi_margin', '-rm', type=int, default=150)
    
    args = parser.parse_args()
    
    print("\n" + "="*75)
    print(" PRODUCTION QATM - PROPERLY FIXED")
    print("="*75)
    print(f" Threshold:       {args.thres} (Strict confidence)")
    print(f" Localization:    {'ENABLED' if args.use_localization else 'DISABLED'}")
    print("="*75 + "\n")

    width, height = args.crop_size, args.crop_size
    np.random.seed(123)
    
    source_image_scale_factor = args.scale_factor
    points_array = []
    line_draw_array = []

    # Load image
    print("[1/6] Loading image...")
    source_image = cv2.imread(args.source)
    if source_image is None:
        print(f"❌ ERROR: Cannot read {args.source}")
        exit(1)
        
    src_h, src_w = source_image.shape[:2]
    print(f"✓ {src_w}×{src_h} pixels")
    
    scaled_image = cv2.resize(source_image, (src_w * source_image_scale_factor, 
                                             src_h * source_image_scale_factor))
    clean_image = scaled_image.copy()

    # Path selection
    print("\n[2/6] Path selection")
    cv2.imshow("point_selector", scaled_image)
    cv2.setMouseCallback('point_selector', mouse_click)
    print("Click 2+ points along path. Press any key when done.")
    cv2.waitKey()
    select_points = False
    cv2.destroyWindow("point_selector")
    
    if len(points_array) < 2:
        print("❌ Need 2+ points")
        exit(1)
    print(f"✓ {len(points_array)} points selected\n")

    # VGG19
    print("[3/6] Loading VGG19...")
    try:
        model = torchvision.models.vgg19(pretrained=True)
        print("✓ Pretrained weights loaded")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        exit(1)
    
    model = model.eval()
    
    if args.cuda:
        if torch.cuda.is_available():
            model = model.cuda()
            print("✓ GPU enabled")
        else:
            print("⚠ CUDA unavailable, using CPU")
            args.cuda = False

    # Data loader
    print("\n[4/6] Initializing QATM...")
    data_loader = ImageData(source_img=args.source, half=False, thres=args.thres)
    
    if args.cuda and data_loader.image is not None:
        data_loader.image = data_loader.image.cuda()
    
    new_size = (int(width * args.resize / 100.0), int(height * args.resize / 100.0))

    # ✅ FIX 2: Pre-compute full image model ONCE
    print("  Computing source features...")
    full_image_model = CreateModel_2(model=model.features, alpha=args.alpha, 
                                     use_cuda=args.cuda, image=data_loader.image)
    print("✓ QATM ready")

    # Camera
    print("\n[5/6] Starting camera...")
    camera_fps = args.fps
    camera = SimCamera(points_array=np.array(points_array), image=args.source, 
                      fps=camera_fps, skip_pts=3)
    camera.start()
    print(f"✓ {camera_fps} FPS")

    # Tracking
    prev_detection_center = None
    frame_count = 0
    successful_detections = 0
    rejected_detections = 0
    total_error = 0

    print("\n[6/6] PROCESSING STARTED")
    print("="*75)
    print("Press 'q' to quit | 's' to save screenshot")
    print("="*75 + "\n")

    while True:
        if camera.frame_q.empty():
            sleep(1 / camera_fps)
            if not camera.running.empty():
                continue
            else:
                print("\n✓ Stream ended")
                break

        ret = camera.frame_q.get()
        if ret is None:
            break
            
        fid, frame_x, frame_y, crop = ret
        print(f"\n{'─'*75}")
        print(f"FRAME {fid:03d} | Camera GT: ({frame_x}, {frame_y})")
        print(f"{'─'*75}")
        frame_count += 1
        frame_start = time()

        # Apply degradations
        crop = noise(crop, args.noise)
        crop = blur(crop, args.blur, (args.blur_filter, args.blur_filter))
        if args.resize != 100:
            crop = cv2.resize(crop, new_size)

        # Load template
        data = data_loader.load_template(crop)
        if args.cuda:
            data['template'] = data['template'].cuda()
        
        roi_offset_x, roi_offset_y = 0, 0
        roi_bounds = None
        
        # ✅ FIX 3: Proper localization logic
        if args.use_localization and prev_detection_center is not None:
            center_x, center_y = prev_detection_center
            print(f"[MODE] LOCALIZED | Prev: ({center_x}, {center_y})")
            
            roi_image, (roi_offset_x, roi_offset_y), roi_bounds = extract_roi_around_detection(
                data_loader.image_raw, center_x, center_y, 
                args.roi_size, args.roi_margin
            )
            
            print(f"  ROI: {roi_bounds}")
            
            # Create ROI-specific model
            roi_tensor = data_loader.transform(roi_image).unsqueeze(0)
            if args.cuda:
                roi_tensor = roi_tensor.cuda()
            
            model_qatm = CreateModel_2(model=model.features, alpha=args.alpha, 
                                       use_cuda=args.cuda, image=roi_tensor)
            search_image = roi_tensor
            
        else:
            print(f"[MODE] FULL IMAGE")
            model_qatm = full_image_model
            search_image = data_loader.image

        # ✅ FIX 4: Run matching
        print("[MATCHING] Running QATM...")
        score = run_one_sample_2(model_qatm, template=data['template'], image=search_image)
        
        # ✅ FIX 5: Stricter NMS - only keep BEST detection
        scores = np.array([score])
        w_array = np.array([[data['template_w']]])
        h_array = np.array([[data['template_h']]])
        
        # Use HIGHER threshold for NMS
        adaptive_thresh = max(args.thres, 0.85)
        thresh_list = [adaptive_thresh]

        try:
            mb_boxes, mb_indices = nms_multi(scores, w_array, h_array, thresh_list, multibox=False)
        except Exception as e:
            print(f"  ⚠ NMS error: {e}")
            mb_boxes, mb_indices = np.array([]), np.array([])

        # ✅ FIX 6: Verify detection quality
        max_score = float(np.max(score))
        
        display_image = clean_image.copy()

        # Only accept HIGH confidence detections
        if len(mb_indices) > 0 and max_score >= args.thres:
            successful_detections += 1
            print(f"[RESULT] ✓ VALID | Score: {max_score:.4f}")
            
            # Map to global coordinates
            if args.use_localization and prev_detection_center is not None:
                mb_boxes_global = map_boxes_to_global(mb_boxes, roi_offset_x, roi_offset_y)
            else:
                mb_boxes_global = mb_boxes
            
            best_box = mb_boxes_global[0]
            center_x = int((best_box[0][0] + best_box[1][0]) / 2)
            center_y = int((best_box[0][1] + best_box[1][1]) / 2)
            
            # Calculate error
            error = np.sqrt((center_x - frame_x)**2 + (center_y - frame_y)**2)
            total_error += error
            
            print(f"  Detection: ({center_x}, {center_y})")
            print(f"  Error: {error:.1f} pixels")
            
            # Update tracking
            prev_detection_center = (center_x, center_y)
            
            # ✅ FIX 7: Clean visualization
            # Draw detection box (GREEN, thin)
            pt1 = tuple(best_box[0].astype(int))
            pt2 = tuple(best_box[1].astype(int))
            cv2.rectangle(display_image, pt1, pt2, (0, 255, 0), 2)
            
            # Draw center dot (RED)
            cv2.circle(display_image, (center_x, center_y), 4, (0, 0, 255), -1)
            
            # Draw ground truth (BLUE)
            cv2.circle(display_image, (frame_x, frame_y), 6, (255, 0, 0), 2)
            cv2.line(display_image, (frame_x-10, frame_y), (frame_x+10, frame_y), (255, 0, 0), 2)
            cv2.line(display_image, (frame_x, frame_y-10), (frame_x, frame_y+10), (255, 0, 0), 2)
            
            # Frame label
            label_pos = (center_x + 15, center_y - 15)
            cv2.putText(display_image, f"F{fid:03d}", label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw ROI box (CYAN, dashed effect)
            if roi_bounds is not None:
                x1, y1, x2, y2 = roi_bounds
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(display_image, "ROI", (x1 + 10, y1 + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
        else:
            # Reject low confidence
            rejected_detections += 1
            if len(mb_indices) > 0:
                print(f"[RESULT] ✗ REJECTED | Score: {max_score:.4f} < {args.thres}")
            else:
                print(f"[RESULT] ✗ NO DETECTION | Max: {max_score:.4f}")
            
            print(f"  → Reset to FULL search")
            prev_detection_center = None
            
            # Draw ground truth only
            cv2.circle(display_image, (frame_x, frame_y), 6, (255, 0, 0), 2)
            cv2.line(display_image, (frame_x-10, frame_y), (frame_x+10, frame_y), (255, 0, 0), 2)
            cv2.line(display_image, (frame_x, frame_y-10), (frame_x, frame_y+10), (255, 0, 0), 2)

        # Add legend
        legend_y = 30
        cv2.putText(display_image, "GREEN: Detection", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_image, "BLUE: Ground Truth", (10, legend_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(display_image, "CYAN: Search ROI", (10, legend_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Result", display_image)
        cv2.imshow("Template", crop)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[USER] Quit")
            break
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"screenshot_{timestamp}.jpg", display_image)
            print(f"  📸 Saved: screenshot_{timestamp}.jpg")

        frame_time = time() - frame_start
        print(f"[TIME] {frame_time:.3f}s")

    # Statistics
    print("\n" + "="*75)
    print(" FINAL STATISTICS")
    print("="*75)
    print(f" Total Frames:      {frame_count}")
    print(f" Successful:        {successful_detections} ({successful_detections/max(1,frame_count)*100:.1f}%)")
    print(f" Rejected:          {rejected_detections}")
    if successful_detections > 0:
        print(f" Average Error:     {total_error/successful_detections:.1f} pixels")
    print("="*75 + "\n")

    camera.stop = True
    cv2.destroyAllWindows()
    print("✓ Complete\n")

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

FPS = 30
source_image_scale_factor = 1
video_file = f'inria_video/{datetime.now().strftime("_%d%m%Y_%H%M%S")}.avi'
select_points = True

def mouse_click(event, x, y, flags, param):
    global scaled_image, points_array, source_image_scale_factor, line_draw_array, select_points
    if select_points:
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(scaled_image, (x, y), 5, (0, 0, 255), 4)
            points_array.append([x * source_image_scale_factor, y * source_image_scale_factor])
            line_draw_array.append([x, y])
            if len(points_array) > 1:
                delta_x, delta_y = line_draw_array[-2][0] - line_draw_array[-1][0], line_draw_array[-2][1] - line_draw_array[-1][1]
                if delta_x != 0:
                    ang = degrees(atan(delta_y / delta_x))
                else:
                    ang = 90.0
                print(line_draw_array[-2], line_draw_array[-1], ang)
                cv2.line(scaled_image, tuple(line_draw_array[-2]), tuple(line_draw_array[-1]), (0, 255, 255), 2)
            cv2.imshow("point_selector", scaled_image)


def extract_roi_around_detection(image_raw, center_x, center_y, roi_size, margin):
    """
    Extract Region of Interest (ROI) around previous detection.
    
    This implements SPATIAL LOCALIZATION by cropping a small region
    around where the target was found in the previous frame.
    
    Args:
        image_raw: Full source satellite image (H, W, 3)
        center_x, center_y: Center coordinates of previous detection
        roi_size: Base size of ROI (e.g., 500 pixels)
        margin: Safety margin around ROI (e.g., 150 pixels)
        
    Returns:
        roi_image: Cropped region (numpy array)
        (offset_x, offset_y): Top-left corner position in full image
    """
    h, w = image_raw.shape[:2]
    half_roi = roi_size // 2
    
    # Calculate ROI boundaries
    x1 = max(0, int(center_x - half_roi - margin))
    y1 = max(0, int(center_y - half_roi - margin))
    x2 = min(w, int(center_x + half_roi + margin))
    y2 = min(h, int(center_y + half_roi + margin))
    
    # Extract ROI using array slicing
    roi_image = image_raw[y1:y2, x1:x2]
    
    print(f"  [LOCALIZATION] ROI extracted: ({x1},{y1}) to ({x2},{y2})")
    print(f"  [OPTIMIZATION] Search area reduced: {w}×{h} → {roi_image.shape[1]}×{roi_image.shape[0]}")
    reduction = ((w * h) - (roi_image.shape[1] * roi_image.shape[0])) / (w * h) * 100
    print(f"  [SPEEDUP] Computation reduced by {reduction:.1f}%")
    
    return roi_image, (x1, y1)


def map_boxes_to_global(boxes, offset_x, offset_y):
    """
    Map detection boxes from ROI coordinate system to global image coordinates.
    
    Args:
        boxes: Detection boxes in ROI local coordinates
        offset_x, offset_y: ROI position in full image
        
    Returns:
        boxes_global: Boxes in full image coordinate system
    """
    if boxes is None or len(boxes) == 0:
        return boxes
    
    boxes_global = boxes.copy()
    # Add offset to convert local → global
    boxes_global[:, :, 0] += offset_x  # X coordinates
    boxes_global[:, :, 1] += offset_y  # Y coordinates
    
    return boxes_global


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='QATM with Spatial Localization')
    
    # Standard arguments
    parser.add_argument('--cuda', action='store_true', default=False,
                        help="Use GPU acceleration")
    parser.add_argument('--resize', '-r', type=int, default=100,
                        help="Template resize percentage")
    parser.add_argument('--crop_size', '-cs', type=int, default=150,
                        help="Template crop size")
    parser.add_argument('--alpha', '-a', type=float, default=25,
                        help="QATM alpha parameter")
    parser.add_argument('--fps', '-f', type=int, default=3,
                        help="Virtual camera FPS")
    parser.add_argument('--scale_factor', '-sf', type=int, default=1,
                        help="Display scale factor")
    parser.add_argument('--thres', '-t', type=float, default=0.7,
                        help="Detection threshold")
    parser.add_argument('--source', '-s', type=str, required=True,
                        help="Path to source satellite image")
    parser.add_argument('--noise', '-n', type=str, default='none',
                        help="Noise type")
    parser.add_argument('--blur', '-b', type=str, default='none',
                        help="Blur type")
    parser.add_argument('--blur_filter', '-bf', type=int, default=5,
                        help="Blur filter size")
    
    # LOCALIZATION arguments
    parser.add_argument('--use_localization', '-ul', action='store_true', default=True,
                        help="Enable spatial localization (RECOMMENDED)")
    parser.add_argument('--roi_size', '-rs', type=int, default=500,
                        help="ROI base size in pixels")
    parser.add_argument('--roi_margin', '-rm', type=int, default=150,
                        help="Safety margin around ROI")
    
    args = parser.parse_args()
    print("\n" + "="*70)
    print("QATM Template Matching with Spatial Localization")
    print("="*70)
    print(f"Localization: {'ENABLED ✓' if args.use_localization else 'DISABLED'}")
    print(f"ROI Size: {args.roi_size}×{args.roi_size} + {args.roi_margin}px margin")
    print("="*70 + "\n")

    width, height = args.crop_size, args.crop_size
    np.random.seed(123)
    
    source_image_scale_factor = args.scale_factor
    points_array = []
    line_draw_array = []

    # Load source image
    source_image = cv2.imread(args.source)
    if source_image is None:
        print(f"ERROR: Could not read image from {args.source}")
        exit(1)
        
    src_h, src_w = source_image.shape[:2]
    print(f"Source image loaded: {src_w}×{src_h} pixels ({src_w * src_h / 1e6:.1f}M pixels)")
    scaled_image = cv2.resize(source_image, (src_w * source_image_scale_factor, 
                                             src_h * source_image_scale_factor))

    # User path selection
    cv2.imshow("point_selector", scaled_image)
    cv2.setMouseCallback('point_selector', mouse_click)
    print("\n[USER INPUT] Click points to define camera path. Press any key when done.")
    cv2.waitKey()
    select_points = False
    cv2.destroyWindow("point_selector")
    
    if len(points_array) < 2:
        print("ERROR: Need at least 2 points to define a path!")
        exit(1)
    
    print(f"✓ Selected {len(points_array)} points\n")

    # Initialize VGG19
    print("[INITIALIZATION] Loading VGG19 model...")
    model = torchvision.models.vgg19(weights=None)
    model = model.eval()
    print("✓ VGG19 initialized\n")

    # Load data
    data_loader = ImageData(source_img=args.source, half=False, thres=args.thres)
    
    from seaborn import color_palette
    color_list = color_palette("hls", 1)
    color_list = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), color_list))
    
    d_img = data_loader.image_raw.copy()
    new_size = (int(width * args.resize / 100.0), int(height * args.resize / 100.0))

    print("[FEATURE EXTRACTION] Creating QATM model with full source image...")
    model_qatm = CreateModel_2(model=model.features, alpha=args.alpha, 
                               use_cuda=args.cuda, image=data_loader.image)

    # Start virtual camera
    camera_fps = args.fps
    print(f"\n[CAMERA] Starting virtual camera at {camera_fps} FPS...")
    camera = SimCamera(points_array=points_array, image=args.source, 
                      fps=camera_fps, skip_pts=3)
    camera.start()

    # =================================================================
    # TRACKING STATE - Core of localization
    # =================================================================
    prev_detection_center = None  # Stores (x, y) of last detection
    frame_count = 0
    total_time_full = 0
    total_time_roi = 0
    tic = time()

    print("\n" + "="*70)
    print("STARTING TEMPLATE MATCHING")
    print("="*70)
    print("Controls: Press 'q' to quit")
    print("="*70 + "\n")

    while True:
        # Check camera queue
        if camera.frame_q.empty():
            sleep(1 / camera_fps)
            if not camera.running.empty() and camera.process.is_alive():
                continue
            else:
                print("\n[CAMERA] Stream ended.")
                break

        # Get frame from camera
        fid, frame_x, frame_y, crop = camera.frame_q.get()
        print(f"\n{'='*70}")
        print(f"FRAME {fid}")
        print(f"{'='*70}")
        frame_count += 1
        frame_start = time()

        # Apply augmentations
        crop = noise(crop, args.noise)
        crop = blur(crop, args.blur, (args.blur_filter, args.blur_filter))
        if args.resize != 100:
            crop = cv2.resize(crop, new_size)

        # Prepare template
        data = data_loader.load_template(crop)

        # Initialize offset variables
        roi_offset_x, roi_offset_y = 0, 0
        
        # =================================================================
        # LOCALIZATION DECISION LOGIC
        # =================================================================
        if args.use_localization and prev_detection_center is not None:
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # LOCALIZED SEARCH (Tracking-by-detection)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            center_x, center_y = prev_detection_center
            print(f"[MODE] LOCALIZED SEARCH around previous detection ({center_x}, {center_y})")
            
            # Extract ROI around previous detection
            roi_image, (roi_offset_x, roi_offset_y) = extract_roi_around_detection(
                data_loader.image_raw, center_x, center_y, 
                args.roi_size, args.roi_margin
            )
            
            # Transform ROI to tensor
            roi_tensor = data_loader.transform(roi_image).unsqueeze(0)
            
            # KEY OPTIMIZATION: Create QATM model with SMALL ROI instead of full image
            print(f"  [PROCESSING] Running VGG19 on ROI...")
            model_qatm = CreateModel_2(model=model.features, alpha=args.alpha, 
                                       use_cuda=args.cuda, image=roi_tensor)
            
            # Update data to use ROI
            data['image'] = roi_tensor
            data['image_raw'] = roi_image
            
        else:
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # FULL IMAGE SEARCH (First frame or tracking lost)
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            print(f"[MODE] FULL IMAGE SEARCH (frame 0 or tracking lost)")
            roi_offset_x, roi_offset_y = 0, 0
            
            # Use full source image
            print(f"  [PROCESSING] Running VGG19 on full image ({src_w}×{src_h})...")
            model_qatm = CreateModel_2(model=model.features, alpha=args.alpha, 
                                       use_cuda=args.cuda, image=data_loader.image)
            data['image'] = data_loader.image
            data['image_raw'] = data_loader.image_raw

        # Run QATM template matching
        print(f"  [MATCHING] Computing similarity scores...")
        score = run_one_sample_2(model_qatm, template=data['template'], image=data['image'])
        scores = np.squeeze(np.array([score]), axis=1)

        w_array = np.array([data['template_w']])
        h_array = np.array([data['template_h']])
        thresh_list = [data['thresh']]

        # Apply NMS
        mb_boxes, mb_indices = nms_multi(scores, w_array, h_array, thresh_list, multibox=True)

        # Process detection results
        if len(mb_indices) > 0:
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # DETECTION SUCCESSFUL
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            print(f"  [RESULT] ✓ Detection found! {len(mb_indices)} box(es)")
            
            # Map boxes from ROI coordinates to global image coordinates
            mb_boxes_global = map_boxes_to_global(mb_boxes, roi_offset_x, roi_offset_y)
            
            # Get best detection box
            best_box = mb_boxes_global[0]
            center_x = int((best_box[0][0] + best_box[1][0]) // 2)
            center_y = int((best_box[0][1] + best_box[1][1]) // 2)
            
            # UPDATE TRACKING STATE for next frame
            prev_detection_center = (center_x, center_y)
            
            print(f"  [TRACKING] Updated target center: ({center_x}, {center_y})")
            
            # Visualization
            d_img = plot_result(scaled_image, mb_boxes_global, text=f"F{fid}", 
                              text_loc=(frame_x - 20, frame_y - 20))
            scaled_image = d_img
            
            # Draw ROI rectangle (blue box)
            if args.use_localization and prev_detection_center is not None:
                roi_x1 = max(0, center_x - args.roi_size // 2 - args.roi_margin)
                roi_y1 = max(0, center_y - args.roi_size // 2 - args.roi_margin)
                roi_x2 = min(src_w, center_x + args.roi_size // 2 + args.roi_margin)
                roi_y2 = min(src_h, center_y + args.roi_size // 2 + args.roi_margin)
                cv2.rectangle(d_img, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
                cv2.putText(d_img, f"ROI", (roi_x1 + 5, roi_y1 + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            cv2.imshow("result", cv2.resize(d_img, (d_img.shape[1] // source_image_scale_factor, 
                                                    d_img.shape[0] // source_image_scale_factor)))
        else:
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # DETECTION FAILED
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            print(f"  [RESULT] ✗ No detection")
            print(f"  [TRACKING] Reset → will use FULL IMAGE search next frame")
            prev_detection_center = None  # Reset tracking

        # Display template
        cv2.imshow("template", crop)

        # Check for quit
        k = cv2.waitKey(1)
        if k == ord('q'):
            print("\n[USER] Quit requested.")
            break

        # Timing statistics
        frame_time = time() - frame_start
        if prev_detection_center is None:
            total_time_full += frame_time
        else:
            total_time_roi += frame_time
            
        print(f"  [TIMING] Frame processed in {frame_time:.3f}s")
        print(f"{'='*70}")

    # Final statistics
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    if total_time_full > 0 and total_time_roi > 0:
        avg_full = total_time_full / max(1, frame_count - sum(1 for _ in range(frame_count) if prev_detection_center))
        avg_roi = total_time_roi / sum(1 for _ in range(frame_count) if prev_detection_center)
        speedup = avg_full / avg_roi if avg_roi > 0 else 1
        print(f"Average time (Full image): {avg_full:.3f}s/frame")
        print(f"Average time (ROI):        {avg_roi:.3f}s/frame")
        print(f"Speedup achieved:          {speedup:.1f}×")
    print("="*70 + "\n")

    cv2.waitKey()
    camera.stop_thread()
    print("Program terminated.\n")
    
'''
# Basic usage with localization (recommended)
python inria_test_v2.py --source satellite.jpg

# Custom ROI parameters
python inria_test_v2.py --source satellite.jpg --roi_size 600 --roi_margin 200

# Disable localization (compare performance)
python inria_test_v2.py --source satellite.jpg --use_localization=False

# With GPU
python inria_test_v2.py --source satellite.jpg --cuda
'''
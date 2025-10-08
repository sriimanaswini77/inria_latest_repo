"""
QATM Localization System with ROI Tracking
Complete implementation with error handling and robust tracking
"""

import cv2
import numpy as np
import os
from glob import glob
from math import atan, degrees
from datetime import datetime
from inria_qatm_pytorch_v2_copy import *
import argparse
from skimage.draw import line
from noise import noise, blur
from cam_sim_latest import SimCamera
import torchvision
from time import time, sleep


# Global variables
FPS = 30
source_image_scale_factor = 1
video_file = f'inria_video/{datetime.now().strftime("_%d%m%Y_%H%M%S")}.avi'
select_points = True


def mouse_click(event, x, y, flags, param):
    """Mouse callback for selecting trajectory waypoints"""
    global scaled_image, points_array, source_image_scale_factor, line_draw_array, select_points
    
    if select_points:
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(scaled_image, (x, y), 5, (0, 0, 255), 4)
            points_array.append([x * source_image_scale_factor, y * source_image_scale_factor])
            line_draw_array.append([x, y])
            
            if len(points_array) > 1:
                delta_x, delta_y = line_draw_array[-2][0] - line_draw_array[-1][0], \
                                    line_draw_array[-2][1] - line_draw_array[-1][1]
                
                if delta_x != 0:
                    ang = degrees(atan(delta_y / (delta_x + 0.00001)))
                else:
                    ang = 90 if delta_y < 0 else -90
                
                cv2.line(scaled_image, tuple(line_draw_array[-2]), tuple(line_draw_array[-1]), 
                        (0, 255, 0), 2)
                cv2.putText(scaled_image, f'{ang:.2f}', tuple(line_draw_array[-1]), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow('select points', scaled_image)


def extract_roi_around_detection(image_raw, center_x, center_y, roi_size, margin):
    """
    Extract Region of Interest around previous detection
    
    Args:
        image_raw: Full source image
        center_x, center_y: Center of previous detection
        roi_size: Size of ROI region
        margin: Extra margin around ROI
        
    Returns:
        roi_image: Cropped ROI image
        offset: (offset_x, offset_y) for coordinate mapping
        roi_bounds: (x1, y1, x2, y2) for visualization
    """
    h, w = image_raw.shape[:2]
    half_roi = roi_size // 2
    
    # Calculate ROI bounds
    x1 = max(0, int(center_x - half_roi - margin))
    y1 = max(0, int(center_y - half_roi - margin))
    x2 = min(w, int(center_x + half_roi + margin))
    y2 = min(h, int(center_y + half_roi + margin))
    
    # Extract ROI
    roi_image = image_raw[y1:y2, x1:x2]
    
    return roi_image, (x1, y1), (x1, y1, x2, y2)


def map_boxes_to_global(boxes, offset_x, offset_y):
    """
    Convert ROI coordinates to global image coordinates
    
    Args:
        boxes: Detection boxes in ROI coordinates
        offset_x, offset_y: ROI offset in source image
        
    Returns:
        boxes_global: Boxes in global coordinates
    """
    boxes_global = boxes.copy()
    boxes_global[:, :, 0] += offset_x
    boxes_global[:, :, 1] += offset_y
    return boxes_global


if __name__ == '__main__':
    # ============================================================
    # COMMAND-LINE ARGUMENT PARSING
    # ============================================================
    parser = argparse.ArgumentParser(
        description='QATM Localization with ROI Tracking',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Basic parameters
    parser.add_argument('--source', type=str, default='./a.png', 
                       help='Source satellite image path')
    parser.add_argument('--alpha', type=int, default=25, 
                       help='QATM alpha parameter (controls matching sharpness)')
    parser.add_argument('--fps', type=int, default=5, 
                       help='Camera frame rate')
    parser.add_argument('--thres', type=float, default=0.85, 
                       help='Detection confidence threshold (0.0-1.0)')
    parser.add_argument('--crop_size', type=int, default=150, 
                       help='Template crop size (pixels)')
    parser.add_argument('--resize', type=int, default=100, 
                       help='Template resize percentage (100=no resize)')
    
    # Localization parameters
    parser.add_argument('--use_localization', action='store_true', default=True,
                       help='Enable ROI-based localization tracking')
    parser.add_argument('--roi_size', type=int, default=500,
                       help='ROI size for localization (pixels)')
    parser.add_argument('--roi_margin', type=int, default=150,
                       help='ROI margin for localization (pixels)')
    
    # Testing parameters
    parser.add_argument('--noise', type=str, default='none',
                       choices=['none', 'gauss', 'gauss_n', 'gauss_u', 'sp', 'poisson'],
                       help='Noise type to apply for robustness testing')
    parser.add_argument('--blur', type=str, default='none',
                       choices=['none', 'normal', 'median', 'gaussian', 'bilateral'],
                       help='Blur type to apply for robustness testing')
    parser.add_argument('--skip_pts', type=int, default=1,
                       help='Skip points in trajectory (1=every point, 2=every other, etc.)')
    
    args = parser.parse_args()
    
    # ============================================================
    # STEP 1: VALIDATE AND LOAD SOURCE IMAGE
    # ============================================================
    print("\n" + "="*70)
    print("QATM LOCALIZATION SYSTEM WITH ROI TRACKING")
    print("="*70)
    
    # Check if source image exists
    if not os.path.exists(args.source):
        print(f"\n❌ ERROR: Image file not found: {args.source}")
        print(f"\n📁 Current directory: {os.getcwd()}")
        print("\n📋 Available image files in current directory:")
        
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']:
            image_files.extend(glob(ext))
            image_files.extend(glob(ext.upper()))
        
        if image_files:
            for i, img in enumerate(image_files[:10], 1):
                print(f"  {i}. {img}")
            if len(image_files) > 10:
                print(f"  ... and {len(image_files)-10} more")
        else:
            print("  (No image files found)")
        
        print("\n💡 Usage:")
        print("  python inria_test_localization_final.py --source <your_image.png>")
        print("\nExample:")
        print("  python inria_test_localization_final.py --source satellite.png")
        exit(1)
    
    # Try to load image
    print(f"\n📷 Loading source image: {args.source}")
    scaled_image = cv2.imread(args.source)
    
    if scaled_image is None:
        print(f"\n❌ ERROR: Could not load image: {args.source}")
        print("\n❓ Possible reasons:")
        print("  1. Unsupported image format")
        print("  2. File is corrupted")
        print("  3. Insufficient file permissions")
        print("  4. OpenCV not properly installed")
        exit(1)
    
    h, w = scaled_image.shape[:2]
    print(f"✓ Image loaded successfully: {w}×{h} pixels ({scaled_image.shape[2]} channels)")
    
    # ============================================================
    # STEP 2: INTERACTIVE WAYPOINT SELECTION
    # ============================================================
    print("\n" + "-"*70)
    print("STEP 1: Select Trajectory Waypoints")
    print("-"*70)
    print("Instructions:")
    print("  • Click points on the image to define camera flight path")
    print("  • Need at least 2 waypoints")
    print("  • Green lines will connect consecutive points")
    print("  • Press ANY KEY when done selecting")
    print()
    
    image_raw = scaled_image.copy()
    points_array = []
    line_draw_array = []
    
    cv2.namedWindow('select points')
    cv2.setMouseCallback('select points', mouse_click)
    cv2.imshow('select points', scaled_image)
    
    print("⏳ Waiting for waypoint selection...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    select_points = False
    
    if len(points_array) < 2:
        print("\n❌ ERROR: Need at least 2 waypoints!")
        print("Please run again and select more points.")
        exit(1)
    
    print(f"✓ Selected {len(points_array)} waypoints")
    points_array = np.array(points_array)
    
    # ============================================================
    # STEP 3: LOAD VGG-19 MODEL
    # ============================================================
    print("\n" + "-"*70)
    print("STEP 2: Loading VGG-19 Model")
    print("-"*70)
    
    vgg_path = "./vgg19.pth"
    vgg = torchvision.models.vgg19(pretrained=False)
    
    if os.path.exists(vgg_path):
        try:
            vgg.load_state_dict(torch.load(vgg_path))
            print(f"✓ Loaded VGG-19 weights from {vgg_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not load {vgg_path}: {e}")
            print("  Downloading pretrained weights...")
            vgg = torchvision.models.vgg19(pretrained=True)
            torch.save(vgg.state_dict(), vgg_path)
            print(f"✓ Saved weights to {vgg_path}")
    else:
        print(f"⚠ {vgg_path} not found")
        print("  Downloading pretrained VGG-19 weights (this may take a while)...")
        vgg = torchvision.models.vgg19(pretrained=True)
        torch.save(vgg.state_dict(), vgg_path)
        print(f"✓ Downloaded and saved weights to {vgg_path}")
    
    vgg = vgg.features
    
    # ============================================================
    # STEP 4: INITIALIZE QATM MODEL
    # ============================================================
    print("\n" + "-"*70)
    print("STEP 3: Initializing QATM Model")
    print("-"*70)
    
    print("  • Loading source image...")
    dataset = ImageData(args.source, thres=args.thres)
    
    print("  • Pre-computing source image features (this may take ~10 seconds)...")
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("  • GPU detected - using CUDA acceleration")
    else:
        print("  • No GPU detected - using CPU (will be slower)")
    
    full_image_model = CreateModel_2(
        model=vgg,
        alpha=args.alpha,
        use_cuda=use_cuda,
        image=dataset.image.cuda() if use_cuda else dataset.image
    )
    print("✓ QATM model initialized successfully")
    
    # ============================================================
    # STEP 5: INITIALIZE CAMERA SIMULATOR
    # ============================================================
    print("\n" + "-"*70)
    print("STEP 4: Initializing Camera Simulator")
    print("-"*70)
    
    cam = SimCamera(
        fps=args.fps,
        image=args.source,
        points_array=points_array,
        crop_size=(args.crop_size, args.crop_size),
        skip_pts=args.skip_pts
    )
    cam.start()
    print(f"✓ Camera started at {args.fps} FPS")
    
    # ============================================================
    # STEP 6: MAIN PROCESSING LOOP WITH LOCALIZATION
    # ============================================================
    print("\n" + "-"*70)
    print("STEP 5: Processing Frames with Localization")
    print("-"*70)
    print(f"Localization: {'ENABLED ✓' if args.use_localization else 'DISABLED'}")
    if args.use_localization:
        print(f"ROI settings: {args.roi_size}×{args.roi_size} + {args.roi_margin}px margin")
    print(f"Detection threshold: {args.thres}")
    print(f"Template size: {args.crop_size}×{args.crop_size}")
    if args.noise != 'none':
        print(f"Noise testing: {args.noise.upper()}")
    if args.blur != 'none':
        print(f"Blur testing: {args.blur.upper()}")
    
    print("\n⌨️  Keyboard controls:")
    print("  • Press 'q' to quit")
    print("  • Press 'r' to reset tracking")
    print("  • Press 't' to toggle tracking on/off")
    print("\n" + "="*70 + "\n")
    
    # Tracking state
    prev_detection_center = None
    tracking_enabled = args.use_localization
    
    # Statistics
    frame_count = 0
    total_time = 0
    full_search_count = 0
    roi_search_count = 0
    detection_success_count = 0
    total_error = 0
    
    try:
        while True:
            # Get frame from camera
            if not cam.frame_q.empty():
                ret = cam.frame_q.get()
                
                if ret is None:
                    print("\n✓ Camera finished - all frames processed")
                    break
                
                fid, frame_x, frame_y, crop = ret
                frame_count += 1
                
                print(f"\n{'='*70}")
                print(f"Frame {fid} | Camera at ({frame_x}, {frame_y})")
                print("-"*70)
                
                # Apply degradations if specified
                if args.noise != 'none':
                    crop = noise(crop, noise_type=args.noise)
                
                if args.blur != 'none':
                    crop = blur(crop, blur_type=args.blur)
                
                # Resize template if specified
                if args.resize != 100:
                    new_size = (int(crop.shape[1] * args.resize / 100),
                               int(crop.shape[0] * args.resize / 100))
                    crop = cv2.resize(crop, new_size)
                
                # Load template
                data = dataset.load_template(crop)
                template = data['template']
                if use_cuda:
                    template = template.cuda()
                
                # ============================================================
                # LOCALIZATION LOGIC: Decide full search or ROI search
                # ============================================================
                start_time = time()
                
                use_roi = (tracking_enabled and prev_detection_center is not None)
                
                if use_roi:
                    # ========== ROI-BASED SEARCH (Fast) ==========
                    search_mode = "ROI"
                    center_x, center_y = prev_detection_center
                    
                    # Extract ROI
                    roi_image, (roi_offset_x, roi_offset_y), roi_bounds = extract_roi_around_detection(
                        dataset.image_raw, center_x, center_y, 
                        args.roi_size, args.roi_margin
                    )
                    
                    print(f"Search: ROI around ({center_x}, {center_y})")
                    print(f"  ROI: {roi_bounds}")
                    
                    # Create ROI-specific model
                    roi_tensor = dataset.transform(roi_image).unsqueeze(0)
                    if use_cuda:
                        roi_tensor = roi_tensor.cuda()
                    
                    model_qatm = CreateModel_2(
                        model=vgg,
                        alpha=args.alpha,
                        use_cuda=use_cuda,
                        image=roi_tensor
                    )
                    
                    search_image = roi_tensor
                    roi_search_count += 1
                    
                else:
                    # ========== FULL IMAGE SEARCH (Slower) ==========
                    search_mode = "FULL"
                    print(f"Search: FULL IMAGE ({w}×{h})")
                    
                    model_qatm = full_image_model
                    search_image = dataset.image
                    roi_offset_x, roi_offset_y = 0, 0
                    roi_bounds = None
                    full_search_count += 1
                
                # Run QATM matching
                score = run_one_sample_2(model_qatm, template=template, image=search_image)
                
                # Apply NMS
                scores = np.array([score])
                w_array = np.array([[data['template_w']]])
                h_array = np.array([[data['template_h']]])
                thresh_list = [args.thres]
                
                mb_boxes, mb_indices = nms_multi(scores, w_array, h_array, thresh_list, multibox=True)
                
                processing_time = time() - start_time
                total_time += processing_time
                
                # ============================================================
                # Process detection results
                # ============================================================
                if len(mb_indices) > 0:
                    # DETECTION SUCCESSFUL
                    detection_success_count += 1
                    
                    # Map to global coordinates if ROI search
                    if use_roi:
                        mb_boxes_global = map_boxes_to_global(mb_boxes, roi_offset_x, roi_offset_y)
                    else:
                        mb_boxes_global = mb_boxes
                    
                    # Calculate center of best detection
                    best_box = mb_boxes_global[0]
                    center_x = int((best_box[0][0] + best_box[1][0]) / 2)
                    center_y = int((best_box[0][1] + best_box[1][1]) / 2)
                    
                    # Update tracking
                    prev_detection_center = (center_x, center_y)
                    
                    # Calculate error
                    error = np.sqrt((center_x - frame_x)**2 + (center_y - frame_y)**2)
                    total_error += error
                    max_score = float(np.max(score))
                    
                    print(f"✓ DETECTED at ({center_x}, {center_y})")
                    print(f"  Confidence: {max_score:.3f}")
                    print(f"  Position error: {error:.1f} pixels")
                    print(f"  Processing: {processing_time*1000:.1f} ms")
                    
                else:
                    # NO DETECTION
                    mb_boxes_global = np.array([])
                    
                    print(f"✗ NO DETECTION above threshold {args.thres}")
                    print(f"  Processing: {processing_time*1000:.1f} ms")
                    
                    # Reset tracking on failure
                    if use_roi:
                        print(f"  ⚠ ROI search failed - will retry with FULL search")
                        prev_detection_center = None
                
                # ============================================================
                # Visualization
                # ============================================================
                vis_image = image_raw.copy()
                
                # Draw ROI bounds if ROI search
                if roi_bounds is not None:
                    x1, y1, x2, y2 = roi_bounds
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(vis_image, 'ROI', (x1+5, y1+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Draw detections
                if len(mb_boxes_global) > 0:
                    for box in mb_boxes_global:
                        x1, y1 = box[0]
                        x2, y2 = box[1]
                        cv2.rectangle(vis_image, (x1-5, y1-5), (x1+5, y1+5), (0, 255, 0), 2)
                        cv2.putText(vis_image, f"F{fid}", (x1+10, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw ground truth position
                cv2.circle(vis_image, (frame_x, frame_y), 7, (0, 0, 255), 2)
                cv2.putText(vis_image, f"GT", (frame_x+10, frame_y+10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Add status text
                mode_text = f"Mode: {search_mode}"
                time_text = f"Time: {processing_time*1000:.1f}ms"
                track_text = f"Tracking: {'ON' if tracking_enabled else 'OFF'}"
                frame_text = f"Frame: {fid}/{frame_count}"
                
                cv2.putText(vis_image, frame_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis_image, mode_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis_image, time_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis_image, track_text, (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show windows
                cv2.imshow("Localization Result", vis_image)
                cv2.imshow("Template", crop)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n✗ User requested exit")
                    break
                elif key == ord('r'):
                    prev_detection_center = None
                    print("  ⟳ Tracking reset")
                elif key == ord('t'):
                    tracking_enabled = not tracking_enabled
                    prev_detection_center = None
                    print(f"  ⚙ Tracking {'ENABLED' if tracking_enabled else 'DISABLED'}")
            
            elif not cam.running.empty():
                print("\n✓ Camera process ended")
                break
            
            else:
                sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user (Ctrl+C)")
    
    finally:
        # ============================================================
        # Cleanup and statistics
        # ============================================================
        cam.stop = True
        cv2.destroyAllWindows()
        
        # Print final statistics
        print("\n" + "="*70)
        print("LOCALIZATION STATISTICS")
        print("="*70)
        print(f"Total frames processed:    {frame_count}")
        print(f"Successful detections:     {detection_success_count}/{frame_count} ({detection_success_count/max(1,frame_count)*100:.1f}%)")
        print(f"Full image searches:       {full_search_count}")
        print(f"ROI searches:              {roi_search_count}")
        if detection_success_count > 0:
            print(f"Average position error:    {total_error/detection_success_count:.1f} pixels")
        print(f"Average time per frame:    {total_time/max(1,frame_count)*1000:.1f} ms")
        print(f"Average FPS:               {frame_count/max(0.001,total_time):.1f}")
        print(f"Total processing time:     {total_time:.2f} seconds")
        if roi_search_count > 0:
            speedup = (full_search_count + roi_search_count) / max(1, full_search_count)
            print(f"Localization speedup:      {speedup:.1f}x")
        print("="*70)
        print("\n✓ Program completed successfully\n")

        ## python inria_test_8oct.py --source "satelllite img.jpg"

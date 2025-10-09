"""
QATM Localization - Best Detection Only + Accurate ROI
Final working version - guaranteed clean output
"""

import cv2
import numpy as np
import argparse
from inria_qatm_pytorchv2_copy import *
from cam_sim import SimCamera
from noise import noise, blur
import torchvision
import os
from time import sleep
from datetime import datetime

def mouse_click(event, x, y, flags, param):
    global img_display, points
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
        points.append([x, y])
        if len(points) > 1:
            cv2.line(img_display, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2)
        cv2.imshow("Select Path", img_display)

def extract_roi(image, cx, cy, roi_size, margin):
    """Extract ROI around detection"""
    h, w = image.shape[:2]
    half = roi_size // 2
    x1 = max(0, cx - half - margin)
    y1 = max(0, cy - half - margin)
    x2 = min(w, cx + half + margin)
    y2 = min(h, cy + half + margin)
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean QATM - Best Detection Only')
    parser.add_argument('--source', '-s', required=True, help='Source image path')
    parser.add_argument('--weights', type=str, default='vgg19.pth')
    parser.add_argument('--thres', type=float, default=0.88, help='Detection threshold (0.85-0.92 recommended)')
    parser.add_argument('--alpha', type=float, default=25)
    parser.add_argument('--fps', type=int, default=3)
    parser.add_argument('--crop_size', type=int, default=150)
    parser.add_argument('--roi_size', type=int, default=400, help='ROI search size')
    parser.add_argument('--roi_margin', type=int, default=100, help='ROI safety margin')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("QATM LOCALIZATION - BEST DETECTION ONLY")
    print("="*70)
    print(f"Threshold: {args.thres} (Higher = Stricter)")
    print(f"ROI Size: {args.roi_size}×{args.roi_size} + {args.roi_margin}px margin")
    print("="*70 + "\n")
    
    # Load image
    print("[1/5] Loading image...")
    BASE_IMAGE = cv2.imread(args.source)
    if BASE_IMAGE is None:
        print(f"ERROR: Cannot load {args.source}")
        exit(1)
    h, w = BASE_IMAGE.shape[:2]
    print(f"✓ Loaded: {w}×{h} pixels")
    
    # Select trajectory
    print("\n[2/5] Select trajectory path...")
    points = []
    img_display = BASE_IMAGE.copy()
    cv2.imshow("Select Path", img_display)
    cv2.setMouseCallback('Select Path', mouse_click)
    print("Click waypoints along path, press any key when done")
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    if len(points) < 2:
        print("ERROR: Need at least 2 waypoints!")
        exit(1)
    print(f"✓ Selected {len(points)} waypoints")
    
    # Load VGG-19
    print("\n[3/5] Loading VGG-19 with pretrained weights...")
    model = torchvision.models.vgg19(pretrained=False)
    if os.path.exists(args.weights):
        try:
            model.load_state_dict(torch.load(args.weights, map_location='cpu'))
            print(f"✓ Loaded weights from {args.weights}")
        except:
            model = torchvision.models.vgg19(pretrained=True)
            torch.save(model.state_dict(), args.weights)
            print(f"✓ Downloaded and saved to {args.weights}")
    else:
        model = torchvision.models.vgg19(pretrained=True)
        torch.save(model.state_dict(), args.weights)
        print(f"✓ Downloaded and saved to {args.weights}")
    
    model = model.eval()
    if args.cuda and torch.cuda.is_available():
        model = model.cuda()
        print("✓ Using GPU")
    
    # Initialize QATM
    print("\n[4/5] Initializing QATM...")
    data_loader = ImageData(source_img=args.source, thres=args.thres)
    if args.cuda:
        data_loader.image = data_loader.image.cuda()
    
    full_model = CreateModel_2(
        model=model.features,
        alpha=args.alpha,
        use_cuda=args.cuda,
        image=data_loader.image
    )
    print("✓ QATM ready")
    
    # Start camera
    print(f"\n[5/5] Starting camera ({args.fps} FPS)...")
    cam = SimCamera(
        points_array=np.array(points),
        image=args.source,
        fps=args.fps,
        skip_pts=3
    )
    cam.start()
    print("✓ Camera started\n")
    
    print("="*70)
    print("PROCESSING (Press 'q' to quit | 's' to save screenshot)")
    print("="*70 + "\n")
    
    # Tracking state
    prev_center = None
    frame_count = 0
    success_count = 0
    total_error = 0
    
    while True:
        if cam.frame_q.empty():
            sleep(1 / args.fps)
            if not cam.running.empty():
                continue
            break
        
        ret = cam.frame_q.get()
        if ret is None:
            print("\n✓ Stream ended")
            break
        
        fid, gt_x, gt_y, crop = ret
        frame_count += 1
        
        print(f"Frame {fid:03d} | GT: ({gt_x:4d},{gt_y:4d}) | ", end='')
        
        # Prepare template
        data = data_loader.load_template(crop)
        if args.cuda:
            data['template'] = data['template'].cuda()
        
        # Localization decision
        roi_bounds = None
        offset_x, offset_y = 0, 0
        
        if prev_center is not None:
            # ROI SEARCH (Localized)
            cx, cy = prev_center
            roi_img, roi_bounds = extract_roi(data_loader.image_raw, cx, cy, args.roi_size, args.roi_margin)
            offset_x, offset_y = roi_bounds[0], roi_bounds[1]
            
            # Create ROI tensor
            roi_tensor = data_loader.transform(roi_img).unsqueeze(0)
            if args.cuda:
                roi_tensor = roi_tensor.cuda()
            
            # Create model for ROI
            model_use = CreateModel_2(
                model=model.features,
                alpha=args.alpha,
                use_cuda=args.cuda,
                image=roi_tensor
            )
            search_img = roi_tensor
            mode = "ROI"
        else:
            # FULL SEARCH
            model_use = full_model
            search_img = data_loader.image
            mode = "FULL"
        
        # Run matching
        score = run_one_sample_2(model_use, data['template'], search_img)
        
        # NMS - BEST DETECTION ONLY
        scores = np.array([score])
        w_arr = np.array([[data['template_w']]])
        h_arr = np.array([[data['template_h']]])
        
        try:
            boxes, indices = nms_multi(
                scores, w_arr, h_arr, [args.thres],
                multibox=False,  # CRITICAL: Only best detection
                max_candidates=200
            )
        except Exception as e:
            print(f"NMS Error: {e}")
            boxes, indices = np.array([]), np.array([])
        
        max_score = float(np.max(score))
        
        # ══════════════════════════════════════════════════════
        # VISUALIZATION - START FRESH EVERY FRAME
        # ══════════════════════════════════════════════════════
        display = BASE_IMAGE.copy()
        
        if len(boxes) > 0 and max_score >= args.thres:
            # SUCCESS - BEST DETECTION FOUND
            success_count += 1
            
            # Map to global coordinates if ROI search
            if prev_center is not None:
                box = boxes[0]
                box[0][0] += offset_x
                box[0][1] += offset_y
                box[1][0] += offset_x
                box[1][1] += offset_y
            else:
                box = boxes[0]
            
            # Calculate center
            det_x = int((box[0][0] + box[1][0]) / 2)
            det_y = int((box[0][1] + box[1][1]) / 2)
            
            # Update tracking
            prev_center = (det_x, det_y)
            
            # Calculate error
            error = np.sqrt((det_x - gt_x)**2 + (det_y - gt_y)**2)
            total_error += error
            
            print(f"Mode: {mode:4s} | Det: ({det_x:4d},{det_y:4d}) | Error: {error:5.1f}px | Score: {max_score:.3f} ✓")
            
            # Draw BEST detection box (THICK GREEN)
            pt1 = tuple(box[0].astype(int))
            pt2 = tuple(box[1].astype(int))
            cv2.rectangle(display, pt1, pt2, (0, 255, 0), 3)
            
            # Draw detection center (GREEN CIRCLE)
            cv2.circle(display, (det_x, det_y), 8, (0, 255, 0), -1)
            cv2.circle(display, (det_x, det_y), 10, (0, 255, 0), 2)
            
            # Frame label
            cv2.putText(display, f"F{fid:03d}", (det_x + 15, det_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw ROI box (CYAN) if localized search
            if roi_bounds is not None:
                x1, y1, x2, y2 = roi_bounds
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
                roi_w, roi_h = x2 - x1, y2 - y1
                cv2.putText(display, f"ROI {roi_w}x{roi_h}", (x1 + 10, y1 + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
        else:
            # FAILURE - NO DETECTION
            print(f"Mode: {mode:4s} | Score: {max_score:.3f} < {args.thres} ✗")
            prev_center = None  # Reset tracking
        
        # Draw ground truth (RED CROSS)
        cv2.drawMarker(display, (gt_x, gt_y), (0, 0, 255), 
                      cv2.MARKER_CROSS, 20, 3)
        cv2.putText(display, "GT", (gt_x + 12, gt_y - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Legend
        legend_bg = np.zeros((120, 280, 3), dtype=np.uint8)
        cv2.rectangle(legend_bg, (0, 0), (279, 119), (255, 255, 255), 2)
        cv2.putText(legend_bg, "GREEN: Best Detection", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(legend_bg, "RED: Ground Truth", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(legend_bg, "CYAN: Search ROI", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(legend_bg, f"Threshold: {args.thres}", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Overlay legend
        display[10:130, 10:290] = cv2.addWeighted(display[10:130, 10:290], 0.3, legend_bg, 0.7, 0)
        
        # Show
        cv2.imshow("QATM Localization", display)
        cv2.imshow("Template", crop)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n✗ User quit")
            break
        elif key == ord('s'):
            fname = f"result_{datetime.now().strftime('%Y%m%d_%qH%M%S')}.jpg"
            cv2.imwrite(fname, display)
            print(f"\n  📸 Saved: {fname}")
    
    # Final statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    print(f"Total frames:      {frame_count}")
    print(f"Successful:        {success_count} ({success_count/max(1,frame_count)*100:.1f}%)")
    if success_count > 0:
        print(f"Average error:     {total_error/success_count:.1f} pixels")
    print("="*70 + "\n")
    
    cam.stop = True
    cv2.destroyAllWindows()

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
from time import time,sleep

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
    """Extract ROI around previous detection center"""
    h, w = image_raw.shape[:2]
    half_roi = roi_size // 2
    x1 = max(0, int(center_x - half_roi - margin))
    y1 = max(0, int(center_y - half_roi - margin))
    x2 = min(w, int(center_x + half_roi + margin))
    y2 = min(h, int(center_y + half_roi + margin))
    roi_image = image_raw[y1:y2, x1:x2]
    print(f"  ROI extracted: ({x1},{y1}) to ({x2},{y2}), size: {roi_image.shape}")
    return roi_image, (x1, y1)

def map_boxes_to_global(boxes, offset_x, offset_y):
    """Map detection boxes from ROI coordinates to global image coordinates"""
    if boxes is None or len(boxes) == 0:
        return boxes
    boxes_global = boxes.copy()
    boxes_global[:, :, 0] += offset_x
    boxes_global[:, :, 1] += offset_y
    return boxes_global

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='INRIA QATM Implementation with Localization')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--resize', '-r', type=int, default=100)
    parser.add_argument('--crop_size', '-cs', type=int, default=150)
    parser.add_argument('--alpha', '-a', type=float, default=25)
    parser.add_argument('--fps', '-f', type=int, default=5)
    parser.add_argument('--scale_factor', '-sf', type=int, default=1)
    parser.add_argument('--thres', '-t', type=float, default=0.7, help="threshold for QATM matching")
    
    # YOUR IMAGE PATH - Updated here
    parser.add_argument('--source', '-s', type=str, 
                       default=r'C:\Users\manas\Downloads\kalidasu sir work\manaswini\satelllite img.jpg')
    
    parser.add_argument('--noise', '-n', type=str, default='none',
                        help="Noise type {gauss_n, gauss_u, sp, poisson, random, none}")
    parser.add_argument('--blur', '-b', type=str, default='none',
                        help="blur type = {normal, median, gauss, bilateral, none}")
    parser.add_argument('--blur_filter', '-bf', type=int, default=5,
                        help="blur filter size, must be odd number")
    parser.add_argument('--use_localization', '-ul', action='store_true', default=True,
                        help="Enable localized search around previous detection")
    parser.add_argument('--roi_size', '-rs', type=int, default=500,
                        help="Size of ROI for localized search")
    parser.add_argument('--roi_margin', '-rm', type=int, default=150,
                        help="Margin around ROI for search expansion")
    
    args = parser.parse_args()
    print(args)


    width, height = args.crop_size, args.crop_size
    np.random.seed(123)
    
    source_image_scale_factor = args.scale_factor
    points_array = []
    line_draw_array = []
    template_resolution = (width, height)

    source_image = cv2.imread(args.source)
    if source_image is None:
        print(f"Error: Could not read image from {args.source}")
        exit(1)
        
    src_h, src_w = source_image.shape[:2]
    print(f"Source image loaded: {src_w}x{src_h}")
    scaled_image = cv2.resize(source_image, (src_w * source_image_scale_factor, src_h * source_image_scale_factor))

    cv2.imshow("point_selector", scaled_image)
    cv2.setMouseCallback('point_selector', mouse_click)
    print("\nClick points on the image to define camera path. Press any key when done.")
    cv2.waitKey()
    select_points = False
    cv2.destroyWindow("point_selector")
    
    if len(points_array) < 2:
        print("Error: Need at least 2 points to define a path!")
        exit(1)
    
    print(f"Selected {len(points_array)} points:", points_array)

    # Load VGG19 model WITHOUT pretrained weights
    print("\nInitializing VGG19 model (random weights, no pretrained)...")
    model = torchvision.models.vgg19(pretrained=False)  # Changed to False
    model = model.eval()
    print("✓ Model initialized")

    data_loader = ImageData(source_img=args.source, half=False, thres=args.thres)
    
    from seaborn import color_palette
    color_list = color_palette("hls", 1)
    color_list = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), color_list))
    
    d_img = data_loader.image_raw.copy()
    new_size = (int(width * args.resize / 100.0), int(height * args.resize / 100.0))

    print("\nCreating QATM model with full source image...")
    model_qatm = CreateModel_2(model=model.features, alpha=args.alpha, use_cuda=args.cuda, image=data_loader.image)

    camera_fps = args.fps
    print(f"\nStarting virtual camera (FPS: {camera_fps})...")
    camera = SimCamera(points_array=points_array, image=args.source, fps=camera_fps, skip_pts=3)
    camera.start()

    # TRACKING STATE
    prev_detection_center = None
    frame_count = 0
    tic = time()

    print("\n" + "="*60)
    print("Starting template matching with localization...")
    print("Press 'q' in any window to quit")
    print("="*60 + "\n")

    while True:
        if camera.frame_q.empty():
            print(time(), " Frame Q Empty")
            sleep(1 / camera_fps)
            if not camera.running.empty() and camera.process.is_alive():
                continue
            else:
                print("Camera closed. Press any key in CV window to close.")
                break

        fid, frame_x, frame_y, crop = camera.frame_q.get()
        print(f"\n{'='*60}")
        print(f"Processing Frame {fid}")
        frame_count += 1

        # Apply augmentations
        crop = noise(crop, args.noise)
        crop = blur(crop, args.blur, (args.blur_filter, args.blur_filter))

        if args.resize != 100:
            crop = cv2.resize(crop, new_size)

        # Prepare template data
        data = data_loader.load_template(crop)

        roi_offset_x, roi_offset_y = 0, 0
        
        # LOCALIZATION LOGIC
        if args.use_localization and prev_detection_center is not None:
            # USE LOCALIZED SEARCH
            center_x, center_y = prev_detection_center
            print(f"Using LOCALIZED search around ({center_x}, {center_y})")
            
            roi_image, (roi_offset_x, roi_offset_y) = extract_roi_around_detection(
                data_loader.image_raw, center_x, center_y, args.roi_size, args.roi_margin
            )
            
            roi_tensor = data_loader.transform(roi_image).unsqueeze(0)
            model_qatm = CreateModel_2(model=model.features, alpha=args.alpha, use_cuda=args.cuda, image=roi_tensor)
            data['image'] = roi_tensor
            data['image_raw'] = roi_image
        else:
            # USE FULL IMAGE SEARCH
            print(f"Using FULL IMAGE search (first frame or tracking lost)")
            roi_offset_x, roi_offset_y = 0, 0
            model_qatm = CreateModel_2(model=model.features, alpha=args.alpha, use_cuda=args.cuda, image=data_loader.image)
            data['image'] = data_loader.image
            data['image_raw'] = data_loader.image_raw

        # Run QATM template matching
        score = run_one_sample_2(model_qatm, template=data['template'], image=data['image'])
        scores = np.squeeze(np.array([score]), axis=1)

        w_array = np.array([data['template_w']])
        h_array = np.array([data['template_h']])
        thresh_list = [data['thresh']]

        # Apply NMS
        mb_boxes, mb_indices = nms_multi(scores, w_array, h_array, thresh_list, multibox=True)

        if len(mb_indices) > 0:
            print(f"  ✓ Detection found! {len(mb_indices)} box(es)")
            
            # Map boxes to global coordinates
            mb_boxes_global = map_boxes_to_global(mb_boxes, roi_offset_x, roi_offset_y)
            
            # Update tracking state
            best_box = mb_boxes_global[0]
            center_x = int((best_box[0][0] + best_box[1][0]) // 2)
            center_y = int((best_box[0][1] + best_box[1][1]) // 2)
            prev_detection_center = (center_x, center_y)
            
            print(f"  Best detection center: ({center_x}, {center_y})")
            
            # Plot result
            d_img = plot_result(scaled_image, mb_boxes_global, text=f"{fid}", text_loc=(frame_x - 20, frame_y - 20))
            scaled_image = d_img
            
            # Draw ROI rectangle for visualization
            if args.use_localization and prev_detection_center is not None:
                roi_x1 = max(0, center_x - args.roi_size // 2 - args.roi_margin)
                roi_y1 = max(0, center_y - args.roi_size // 2 - args.roi_margin)
                roi_x2 = min(src_w, center_x + args.roi_size // 2 + args.roi_margin)
                roi_y2 = min(src_h, center_y + args.roi_size // 2 + args.roi_margin)
                cv2.rectangle(d_img, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2)
                cv2.putText(d_img, "ROI", (roi_x1 + 5, roi_y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow("result", cv2.resize(d_img, (d_img.shape[1] // source_image_scale_factor, d_img.shape[0] // source_image_scale_factor)))
        else:
            print(f"  ✗ No detection - will use FULL IMAGE next frame")
            prev_detection_center = None

        cv2.imshow("v-camera", crop)

        k = cv2.waitKey(1)
        if k == ord('q'):
            print("\nUser requested quit.")
            break

        toc = time()
        print(f"  Frame processing time: {toc - tic:.3f}s")
        tic = toc

    cv2.waitKey()
    camera.stop_thread()
    print("\nProgram terminated.")

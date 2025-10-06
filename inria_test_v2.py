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
from cam_sim import SimCamera

#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#width = 100
#height = 100
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
                delta_x, delta_y = line_draw_array[-2][0] - line_draw_array[-1][0], line_draw_array[-2][1] - \
                                   line_draw_array[-1][1]
                ang = degrees(atan(delta_y / delta_x))
                print(line_draw_array[-2], line_draw_array[-1], ang)
                cv2.line(scaled_image, tuple(line_draw_array[-2]), tuple(line_draw_array[-1]), (0, 255, 255), 2)
            cv2.imshow("point_selector", scaled_image)

def rect_bbox2(rect):
    (center_x, center_y), (width, height), _ = rect
    x, y, w, h = int(center_x - width / 2), int(center_y - height / 2), int(width), int(height)
    return (x, y, x + w, y + h)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='INRIA QATM  Implementation')
    parser.add_argument('--cuda', action='store_true', default=False)
    #parser.add_argument('--select', action='store_true', default=False)
    parser.add_argument('--resize', '-r', type=int, default=100)
    parser.add_argument('--crop_size', '-cs', type=int, default=150)
    parser.add_argument('--alpha', '-a', type=float, default=25)
    parser.add_argument('--fps', '-f', type=int, default=5)
    parser.add_argument('--scale_factor', '-sf', type=int, default=1)
    parser.add_argument('--thres', '-t', type=float, default=0.7, help="threshold for QATM matching")
    parser.add_argument('--source', '-s', type=str)
    parser.add_argument('--noise', '-n', type=str, default='none',
                        help="Noise type {gauss_n, gauss_u, sp, poisson, random, none}")
    parser.add_argument('--blur', '-b', type=str, default='none',
                        help="blur type = {normal, median, gauss, bilateral, none}")
    parser.add_argument('--blur_filter', '-bf', type=int, default=5,
                        help="blur filter size, must be odd number")
    parser.add_argument('--local', '-l', action='store_true', default=False)
    parser.add_argument('--local_size', '-ls', type=int, default=300)
    #parser.add_argument('--half',  action='store_true', default=False)
    args = parser.parse_args()
    print(args)
    width, height = args.crop_size, args.crop_size
    #outVideo = cv2.VideoWriter(video_file, fourcc, FPS, (width, height))
    # random seed to reproduce the result
    np.random.seed(123)
    source_image_scale_factor = args.scale_factor

    points_array = []
    line_draw_array = []
    template_resolution = (width, height)
    # combined = combine_tiles('dataset/chicago*.tif')
    source_image = cv2.imread(args.source)
    src_h, src_w = source_image.shape[:2]
    scaled_image = cv2.resize(source_image, (src_h * source_image_scale_factor, src_w * source_image_scale_factor))
    # scaled image is for display purpose only.

    cv2.imshow("point_selector", scaled_image)
    cv2.setMouseCallback('point_selector', mouse_click)
    cv2.waitKey()
    select_points = False
    cv2.destroyWindow("point_selector")

    print(points_array)



    model = torchvision.models.vgg19()
    model.load_state_dict(torch.load("./vgg19.pth"))
    model = model.eval()  #.half()
    #model = CreateModel(model=model.features, alpha=args.alpha, use_cuda=args.cuda)

    data_loader = ImageData(source_img=args.source, half=False, thres=args.thres)

    color_list = color_palette("hls", 1)
    color_list = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), color_list))

    color = 0
    d_img = data_loader.image_raw.copy()
    new_size = (int(width * args.resize / 100.0), int(height * args.resize / 100.0))

    model = CreateModel_2(model=model.features, alpha=args.alpha, use_cuda=args.cuda, image=data_loader.image)

    #scaled_image = cv2.resize(source_image, (comb_h // source_image_scale_factor, comb_w // source_image_scale_factor))
    camera_fps=args.fps
    camera = SimCamera(points_array=points_array, image=args.source, fps=camera_fps, skip_pts=3)
    camera.start()
    tic=time()

    while True:
        if camera.frame_q.empty():
            print(time(), " Frame Q Empty")
            sleep(1/camera_fps)
            print(camera.running)

            if not camera.running.empty() and camera.process.is_alive():
                continue
            else:
                print("Camera closed. press any key in CV window to close.")
                break
        fid, frame_x, frame_y, crop = camera.frame_q.get()
        print(fid, " Frame used")
        # apply noise
        crop = noise(crop, args.noise)

        # apply blur
        crop = blur(crop, args.blur, (args.blur_filter, args.blur_filter))

        # apply resize on input video
        if args.resize != 100:
            crop = cv2.resize(crop, new_size)

        w_array = []
        h_array = []
        thresh_list = []

        # pass the template to data loader
        data = data_loader.load_template(crop)

        score = run_one_sample_2(model, template=data['template'], image=data['image'])
        scores = np.squeeze(np.array([score]), axis=1)
        w_array.append(data['template_w'])
        w_array = np.array(w_array)
        h_array.append(data['template_h'])
        h_array = np.array(h_array)
        thresh_list.append(data['thresh'])

        # apply nms and threshold on the results of QATM
        mb_boxes, mb_indices = nms_multi(scores, w_array, h_array, thresh_list, multibox=True)

        if len(mb_indices) > 0:
            # d_img = plot_result(d_img, mb_boxes[0][None, :, :].copy())
            d_img = plot_result(scaled_image, mb_boxes[0][None, :, :],text=f"{fid}",text_loc=(frame_x-20,frame_y-20))
            scaled_image = d_img

        # cv2.circle(scaled_image, (x // source_image_scale_factor, y // source_image_scale_factor), 3,
        #           (255, 0, 0), 3)
        cv2.imshow("result", cv2.resize(d_img, (d_img.shape[1] // source_image_scale_factor,
                                                d_img.shape[0] // source_image_scale_factor)))
        # cv2.imshow("expected_path", scaled_image)
        cv2.imshow("v-camera", crop)
        # cv2.imshow("original", combined)
        # outVideo.write(crop)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        toc = time()
        print("Time taken: ", toc - tic)
        tic = toc
    cv2.waitKey()
    
    
 


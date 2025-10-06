import cv2
import numpy as np
from multiprocessing import Queue,Process
from time import sleep
from skimage.draw import line


class SimCamera:
    def __init__(self, fps=30, video=None, image=None, points_array=None,crop_size=(150,150),skip_pts=1):
        if fps<1:
            fps=1
        self.fps = fps
        self.frame_q=Queue(1)
        self.stop=False
        self.frame=None
        self.video=None
        self.image=None
        self.points=None
        self.frame_ctr=-1
        self.crop_size=crop_size
        self.process=None
        self.running=Queue()
        if video:
            self.video=cv2.VideoCapture(video)
            if not self.video.isOpened():
                self.video=None
        if image:
            self.image=cv2.imread(image)
            self.points=[]
            x1y1=points_array[0]
            for x2y2 in points_array[1:]:
                points=self.line_points(x1y1, x2y2,skip_pts=skip_pts)
                self.points.extend(points)
                x1y1=x2y2
            print("Total points to trace: ",len(self.points))

        self.margin = 50
        self.width,self.height = self.crop_size

        self.dst_pts = np.array([
            [0, 0],  # top-left
            [self.width - 1, 0],  # top-right
            [self.width - 1, self.height - 1],  # bottom-right
            [0, self.height - 1]  # bottom-left
        ], dtype=np.float32)




    def line_points(self,x1y1, x2y2, skip_pts):
        pts = list(zip(*line(*x1y1, *x2y2)))
        pts = pts[::skip_pts]
        return pts

    def start(self):

        self.process = Process(target=self.get_frame)
        self.process.daemon=True
        #self.stop = False
        self.running.put(True)
        self.process.start()
        print(self.process, " process status")


    def get_frame(self):
        self.ctr=-1
        while not self.running.empty():
            self.ctr += 1
            if self.video is not None:
                ret,img=self.video.read()
                if ret:
                    if self.frame_q.full():
                        _=self.frame_q.get()
                    self.frame_q.put(img)
                    self.frame_ctr += 1
                else:
                    self.stop=True
                    print("Reached end of video", self.frame_ctr)
            elif self.image is not None and self.points is not None:
                if self.frame_ctr>=len(self.points):
                    print("End of frames. closing")
                    if not self.running.empty():
                        _=self.running.get()
                    break

                else:
                    #crop = self.image[self.points[self.frame_ctr][1]:self.points[self.frame_ctr][1]+self.crop_size[1],self.points[self.frame_ctr][0]:self.points[self.frame_ctr][0]+self.crop_size[0]]
                    #crop = self.image[self.points[self.frame_ctr][1]:self.points[self.frame_ctr][1]+self.crop_size[1],self.points[self.frame_ctr][0]:self.points[self.frame_ctr][0]+self.crop_size[0]]
                    crop = self.get_perspective(self.points[self.frame_ctr][0],self.points[self.frame_ctr][1])
                    #cv2.imshow("v-camera", crop)
                    #cv2.imshow(1)
                    self.frame_ctr += 1
                    if self.frame_q.full():
                        _=self.frame_q.get()
                        print(self.ctr,(self.points[self.frame_ctr][0].item(),self.points[self.frame_ctr][1].item()), " Frame Drop")
                    self.frame_q.put([self.frame_ctr,self.points[self.frame_ctr][0],self.points[self.frame_ctr][1],crop])
                    print(self.ctr," Frame added")
            sleep(1/self.fps)
        if not self.running.empty():
            _ = self.running.get()
        print("Closing thread")
        # if self.frame_q.empty():
        #     self.frame_q.put(None)

    def get_perspective(self,x,y):
        src_pts = np.array([
            [x,y],  # top-left
            [x+self.width, y-self.margin],  # top-right
            [x+self.width, y+self.height+self.margin],  # bottom-right
            [x,y+self.height+self.margin]  # bottom-left
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, self.dst_pts)
        warped = cv2.warpPerspective(self.image, M, self.crop_size)
        return warped

    def stop_thread(self):
        if not self.running.empty():
            #self.stop=True
            _ = self.running.get()
            if self.process:
                self.process.join()

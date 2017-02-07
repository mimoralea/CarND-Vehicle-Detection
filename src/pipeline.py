from __future__ import print_function
from collections import deque
from sys import argv
from moviepy.editor import VideoFileClip
from scipy.misc import imread, imresize
from skimage import data, feature

from camera import Camera
from vehicle_detection import VehicleDetection
from vehicle_tracking import VehicleTracking

import matplotlib.pyplot as plt
import logging as log
import numpy as np

import argparse
import time
import random
import cv2
import os


class Pipeline:
    def __init__(self, reset=False):
        """
        """
        self.frame_counter = -1
        self.logs_directory = 'logs'
        self.camera = None
        self.vehicle_detection = None
        self.vehicle_tracking = None
        self.last_detection = None

        if reset:
            log.info('cleaning up directories')
            import shutil
            shutil.rmtree(self.logs_directory)

        if not os.path.exists(self.logs_directory): os.makedirs(self.logs_directory)

        log.debug(vars(self))

    def set_camera(self, camera):
        self.camera = camera

    def set_vehicle_detection(self, vehicle_detection):
        self.vehicle_detection = vehicle_detection

    def set_vehicle_tracking(self, vehicle_tracking):
        self.vehicle_tracking = vehicle_tracking

    def __draw_raw_boxes(self, img, squares, color=(0, 0, 255), thick=6):
        log.debug('drawing ' + str(len(squares)) + ' vehicles boxes')

        # make a copy of the image
        draw_img = np.copy(img)

        # iterate through the bounding boxes
        for bbox in squares:
            # draw a rectangle given bbox coordinates
            cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)

        # return the image copy with boxes drawn
        return draw_img

    def __draw_filtered_boxes(self, img, labels, color=(0, 255, 0), thick=6):
        # Iterate through all detected cars
        if len(labels) == 0:
            return img

        for car_number in range(1, labels[1]+1):
            log.debug('car number ' + str(car_number))
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            x_dist = np.max(nonzerox) - np.min(nonzerox)
            y_dist = np.max(nonzeroy) - np.min(nonzeroy)
            log.debug('distances of x=%d and y=%d'.format(x_dist, y_dist))
            # only draw squares and rectangles - but not too disproportionate
            if x_dist > y_dist * 4 or y_dist > x_dist * 2:
                continue
            cv2.rectangle(img, bbox[0], bbox[1], color, thick)
        # Return the image
        return img

    def __overlay_heatmap(self, img, heatmap):
        # dark = np.zeros_like(heatmap).astype(np.uint8)
        heatmap *= 255.0/heatmap.max()
        overlay = np.dstack((heatmap, heatmap, heatmap)).astype(np.uint8)
        log.debug('heatmap values')
        log.debug(np.mean(heatmap))
        log.debug(np.median(heatmap))
        log.debug(np.min(heatmap))
        log.debug(np.max(heatmap))
        log.debug('overlaying heatmap on top of image')
        log.debug('image shape ' + str(img.shape))
        log.debug('heatmap shape ' + str(heatmap.shape))
        log.debug('overlay shape ' + str(overlay.shape))
        img = cv2.addWeighted(img, 1, overlay, 0.999, 0)
        return img

    def process_frame(self, img_in):
        self.frame_counter += 1
        log.info('processing frame ' + str(self.frame_counter))

        log.info('copying image for backup')
        img_out = np.copy(img_in)
        img_out = imresize(img_out, (720, 1280))

        log.info('undistort image')
        img_out = self.camera.undistort(img_out)

        log.info('detect vehicles on frame')
        if self.frame_counter % 5 != 0:
            filtered, squares = self.last_detection
        else:
            filtered, squares = self.vehicle_detection.detect_vehicles(img_out)
            self.last_detection = (filtered, squares)

        log.info('tracking vehicles by cleaning false positives')
        labels, heatmap = self.vehicle_tracking.remove_false_positives(
            img_out.shape[:2], filtered)

        log.info('draw vehicle raw boxes')
        # img_out = self.__draw_raw_boxes(img_out, squares, (0, 255, 0), 1)

        log.info('draw vehicle filtered boxes')
        # img_out = self.__draw_raw_boxes(img_out, filtered, (0, 0, 255), 2)

        log.info('draw vehicle labels')
        img_out = self.__draw_filtered_boxes(img_out, labels, (255, 0, 0), 3)

        log.info('overlay heatmap')
        img_out = self.__overlay_heatmap(img_out, heatmap)

        log.info('return processed frame')
        return img_out

    def process_video(self, in_path, out_path):
        log.info('processing video ' + in_path)
        log.info('will save output at ' + out_path)

        log.info('processing video clip')
        clip = VideoFileClip(in_path)
        output = clip.fl_image(self.process_frame)
        output.write_videofile(out_path, audio=False)

        log.info('done with process_video method')



def main(args):
    """
    """
    log.info('Verbose output enabled ' + str(log.getLogger().getEffectiveLevel()))
    log.debug(args)

    camera = Camera(args.calibration_path)
    camera.calibrate()

    vehicle_detection = VehicleDetection()
    vehicle_detection.fit('datasets/', force=False)
    vehicle_tracking = VehicleTracking()

    pipeline = Pipeline()
    pipeline.set_camera(camera)
    pipeline.set_vehicle_detection(vehicle_detection)
    pipeline.set_vehicle_tracking(vehicle_tracking)

    if args.image_path:
        img = imread(args.image_path)
        result = pipeline.process_frame(img)
        plt.imshow(result)
        plt.show()
        log.info('end of image processing')
    else:
        pipeline.process_video(args.source, args.destination)
        log.info('end of video processing check output at ' + args.destination)
    log.info('end of processing')

if __name__ == '__main__':
    """
    Loads the script and parses the arguments
    """
    parser = argparse.ArgumentParser(
        description='A Deep Reinforcement Learning agent that plays pong like a King.'
    )
    parser.add_argument(
        '-v',
        help='logging level set to ERROR',
        action='store_const', dest='loglevel', const=log.ERROR,
    )
    parser.add_argument(
        '-vv',
        help='logging level set to INFO',
        action='store_const', dest='loglevel', const=log.INFO,
    )
    parser.add_argument(
        '-vvv',
        help='logging level set to DEBUG',
        action='store_const', dest='loglevel', const=log.DEBUG,
    )
    parser.add_argument(
        '-i', '--input-video',
        help='Path to the video to be processed',
        dest='source', type=str, required=False,
    )
    parser.add_argument(
        '-o', '--output-video',
        help='Desired path of output video',
        dest='destination', type=str, default='output.mp4',
    )
    parser.add_argument(
        '-f', '--input-frame',
        help='Path to the image to be processed',
        dest='image_path', type=str, required=False,
    )
    parser.add_argument(
        '-l', '--calibration-path',
        help='Desired path of output video',
        dest='calibration_path', type=str,
    )
    """
    parser.add_argument(
        '-g', '--games',
        help='number of games for the agent to play. (default: 100) '
        dest='ngames', type=int, default=100,
    )
    """
    parser.add_argument(
        '-c', '--clear',
        help='clears log and debug folders the folders'
             'NOTE: use this to start from scratch',
        dest='reset', action='store_true')

    args = parser.parse_args()
    if args.loglevel:
        log.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=args.loglevel)
    else:
        log.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p', level=log.CRITICAL)

    main(args)

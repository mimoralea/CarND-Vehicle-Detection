from scipy.misc import imread, imresize

import logging as log
import numpy as np
import os
import cv2
import glob
import pickle

class Camera:

    def __init__(self, calibration_path):
        root_dir = os.path.dirname(calibration_path + '/*.jpg')
        self.mtx_pkl = root_dir + '/mtx.pkl'
        self.dist_pkl = root_dir + '/dist.pkl'
        log.debug('pickle files ' + self.mtx_pkl + ' and ' + self.dist_pkl)

        self.mtx = None
        self.dist = None
        if os.path.exists(self.mtx_pkl) and os.path.exists(self.dist_pkl):
            log.info('loading camera mtx and dist from pickle files')
            with open(self.mtx_pkl, 'rb') as input:
                self.mtx = pickle.load(input)

            with open(self.dist_pkl, 'rb') as input:
                self.dist = pickle.load(input)

        self.calibration_path = calibration_path

    def is_calibrated(self):
        log.debug('mtx == None? ' + str(self.mtx == None) +
                  '; dist == None? ' + str(self.dist == None))
        return self.mtx != None and self.dist != None

    def __calibrate_from_path(self, imshape):
        log.debug('calibration from path with shape ' + str(imshape))
        objpoints, imgpoints, _ = self.__get_calibration_coef()
        self.__calibrate(imshape, objpoints, imgpoints)

    def __get_calibration_coef(self, nx=9, ny=6, plot=False):
        log.debug('getting calibration coefficients')
        objpoints = []
        imgpoints = []
        drawnimgs = []

        objp = np.zeros((nx*ny, 3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        fnames = glob.glob(self.calibration_path)

        for fname in fnames:
            img = imread(fname)

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

            # if chessboard corners were not found, continue to next image
            if not ret:
                log.info('corners were not found on image, continuing to next')
                continue

            # save the points to calibrate later
            imgpoints.append(corners)
            objpoints.append(objp)

            # no need to waste cycles if do not want plotting
            if not plot:
                log.debug('not plotting this function call')
                continue

            # draw points in the img and save a copy
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            drawnimgs.append(img)
        log.info('returning a total of ' + str(len(objpoints)) + ' object points and ' +
                 str(len(imgpoints)) + ' image points')
        return objpoints, imgpoints, drawnimgs

    def __calibrate(self, img_shape, objpoints, imgpoints):
        log.debug('calling OpenCV calibrateCamera method')
        _, self.mtx, self.dist, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints, img_shape, None, None)

        with open(self.mtx_pkl, 'wb') as output:
            log.info('saving ' + self.mtx_pkl)
            pickle.dump(self.mtx, output, pickle.HIGHEST_PROTOCOL)

        with open(self.dist_pkl, 'wb') as output:
            log.info('saving ' + self.dist_pkl)
            pickle.dump(self.dist, output, pickle.HIGHEST_PROTOCOL)

    def calibrate(self, force=False):
        log.debug('calibrating camera public method')

        if self.is_calibrated() and force == False:
            log.info('camera already calibrated and no force flag. Getting out early')
            return

        fnames = glob.glob(self.calibration_path)
        log.debug('got a total of ' + str(len(fnames)) + ' in the calibration directory: ' + self.calibration_path)
        img = imread(fnames[0])
        self.__calibrate_from_path(img.shape[:2])

    def undistort(self, img):
        log.debug('calling OpenCV undistort method')
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)

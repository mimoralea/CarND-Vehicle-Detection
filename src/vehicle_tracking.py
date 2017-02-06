from collections import deque

from scipy.ndimage.measurements import label

import matplotlib.pyplot as plt
import numpy as np
import logging as log


class VehicleTracking:
    def __init__(self):
        self.labels_queue = deque()
        self.nlabels = 15
        self.boxes_queue = deque()
        self.nboxes_lists = 30

    def __add_heat(self, heatmap, boxlist):
        # Iterate through list of bboxes
        for box in boxlist:
            # Add += 1 for all pixels inside each bbox
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap

    def __apply_threshold(self, heatmap, threshold = 10):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def __filter_non_continuous(self, labels):
        label_mesh, label_count = labels
        label_mesh = label_mesh.astype(np.uint8)

        new_mesh = np.zeros_like(label_mesh, dtype=np.uint8)
        for label in range(1, label_count + 1):
            tmp_mesh = np.zeros_like(label_mesh, dtype=np.uint8)
            # regardless of label start with 1
            tmp_mesh[label_mesh == label] = 1
            for prev_lab in self.labels_queue:
                # reset all values to one and start adding
                prev_lab[prev_lab > 0] = 1
                tmp_mesh += prev_lab
            log.debug('tmp mesh looks like')
            log.debug(np.max(tmp_mesh))
            log.debug(np.min(tmp_mesh))
            log.debug(np.mean(tmp_mesh))
            if np.max(tmp_mesh) > self.nlabels:
                new_mesh[label_mesh == label] = np.max(new_mesh) + 1
        return new_mesh, np.max(new_mesh)

    def remove_false_positives(self, shape, raw_boxes):
        heatmap = np.zeros(shape, dtype=np.float)
        log.debug(heatmap.shape)
        if len(self.boxes_queue) < self.nboxes_lists:
            self.boxes_queue.append(raw_boxes)
            log.debug('not enough boxes history ' + str(len(self.boxes_queue)))
            return [], heatmap

        log.debug('removing false positives')
        self.boxes_queue.popleft()
        self.boxes_queue.append(raw_boxes)

        for boxlist in self.boxes_queue:
            heatmap = self.__add_heat(heatmap, boxlist)

        heatmap = self.__apply_threshold(heatmap)
        labels = label(heatmap)

        log.debug(labels[0].shape)
        log.debug(np.max(labels[0]))
        log.debug(np.mean(labels[0]))
        log.debug(np.median(labels[0]))

        self.labels_queue.append(labels[0].astype(np.uint8))
        if len(self.labels_queue) > self.nlabels:
            self.labels_queue.popleft()
        labels = self.__filter_non_continuous(labels)

        log.debug('ended with ' + str(labels[1]) + ' cars')
        return labels, heatmap

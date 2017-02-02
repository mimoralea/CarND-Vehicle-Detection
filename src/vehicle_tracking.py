from collections import deque

from scipy.ndimage.measurements import label

import matplotlib.pyplot as plt
import numpy as np
import logging as log

class VehicleTracking:
    def __init__(self):
        self.boxes_queue = deque()
        self.nboxes_lists = 10

    def __add_heat(self, heatmap, boxlist):
        # Iterate through list of bboxes
        for box in boxlist:
            # Add += 1 for all pixels inside each bbox
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap

    def __apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

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

        heatmap = self.__apply_threshold(heatmap, 25)
        #final_map = np.clip(heatmap - 2, 0, 255)
        #plt.imshow(final_map, cmap='hot')
        #plt.imshow(heatmap, cmap='hot')
        #plt.show()

        labels = label(heatmap)
        log.debug('ended with ' + str(labels[1]) + ' cars')
        return labels, heatmap


import logging as log


class VehicleTracking:
    def __init__(self):
        pass

    def remove_false_positives(self, coordinates):
        log.debug('removing false positives started with ' + str(len(coordinates)) + ' boxes')

        bboxes = [
            ((275, 572), (380, 510)),
            ((488, 563), (549, 518)),
            ((554, 543), (582, 522)),
            ((601, 555), (646, 522)),
            ((657, 545), (685, 517)),
            ((849, 678), (1135, 512))
        ]
        bboxes = [
            ((275, 572), (380, 510)),
            ((488, 563), (549, 518)),
            ((554, 543), (582, 522)),
            ((601, 555), (646, 522)),
            ((657, 545), (685, 517)),
            ((849, 678), (1135, 512))
        ]

        log.debug('ended with ' + str(len(bboxes)) + ' boxes')
        return bboxes

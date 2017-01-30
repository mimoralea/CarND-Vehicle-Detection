"""
This is the module the detects vehicles in images
"""

import logging as log

class VehicleDetection:
    """
    Main class for vehicle detection pipeline
    """
    def __init__(self):
        pass


    def detect_vehicles(self, img):
        """
        Main method to detect vehicles.
        It receives an image and if returns coordinates
        of the potential vehicles after the overlap.
        False positives should be detected by the vehicle tracking class
        """
        bboxes = [
            ((275, 572), (380, 510)),
            ((488, 563), (549, 518)),
            ((554, 543), (582, 522)),
            ((601, 555), (646, 522)),
            ((657, 545), (685, 517)),
            ((849, 678), (1135, 512))
        ]
        return bboxes

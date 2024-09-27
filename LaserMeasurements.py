import CameraMeasurements
import cv2 
from fulladjustment import detect_aruco
import MeasurementTool

class LaserMeasurements(MeasurementTool):
    def __init__(self, camera, alignment_threshold: float = 0.00001):
        """
        Parameters
        ----------
        alignment_threshold : float, optional
            The threshold, in radians, to consider the laser as aligned.
            The default is 0.001.
        """
        self.alignment_threshold = alignment_threshold

    def measureAngleDeviation(self, source_id,  target_id):
        cam_measure = CameraMeasurements.CameraMeasurements()
        rvec_source, tvec_source = cam_measure.find_pose(source_id)
        rvec_target, tvec_target = cam_measure.find_pose(target_id)

        scale = 0.1
        tvec_between = scale * tvec_source + (1 - scale) * tvec_target

        Center1 = cam_measure.measureBeamCenter(tvec_between)

        print("Alignment Deviation: " + Center1)

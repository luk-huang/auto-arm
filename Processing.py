import cv2
import numpy as np

class Processing:
    def __init__(self):
        """
        Initialize the Processing class.
        """
        pass
    
    @staticmethod
    def estimate_pose(self, corners, mtx, dist, marker_length):
        """
        Estimate the pose of an ArUco marker.

        Parameters
        ----------
        corners : list of numpy.ndarray
            Detected corners of the ArUco marker.
        mtx : numpy.ndarray
            Camera matrix from calibration.
        dist : numpy.ndarray
            Distortion coefficients from calibration.
        marker_length : float
            Side length of the marker in meters.

        Returns
        -------
        rvec : numpy.ndarray
            Rotation vector representing the orientation of the marker.
        tvec : numpy.ndarray
            Translation vector representing the position of the marker.
        """
        obj_points = np.array([
            [-marker_length / 2, marker_length / 2, 0],
            [marker_length / 2, marker_length / 2, 0],
            [marker_length / 2, -marker_length / 2, 0],
            [-marker_length / 2, -marker_length / 2, 0]
        ])
        success, rvec, tvec = cv2.solvePnP(obj_points, corners, mtx, dist)
        return rvec, tvec

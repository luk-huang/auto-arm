import pyrealsense2 as rs
import cv2
import numpy as np
from MeasurementTool import MeasurementTool
from fulladjustment import detect_aruco, estimate_pose, average_rotation_vectors

class CameraMeasurements(MeasurementTool):
    def __init__(self):
        """
        Constructor for CameraMeasurements class.

        Initializes the camera pipeline and sets up the projection matrices,
        ArUco dictionary and parameters, and captures the video streams from both cameras.

        :param:
            None
        :return:
            None
        """

        print(cv2.__version__)
        # Load calibration data
        self.calibration_data = np.load('stereo_calibration2.npz')
        self.mtx1 = self.calibration_data['mtx1']
        self.dist1 = self.calibration_data['dist1']
        self.mtx2 = self.calibration_data['mtx2']
        self.dist2 = self.calibration_data['dist2']
        self.R = self.calibration_data['R']
        self.T = self.calibration_data['T']
        self.actual_distance = 0.23  # 23 cm
        self.calibrated_distance = 0.22  # 22 cm

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        pipeline.start(config)

        # Calculate the scaling factor
        scale_factor = self.actual_distance / self.calibrated_distance
        # print(f"Scaling factor: {scale_factor}")

        # Adjust the translation vector
        T = T * scale_factor
        # print(f"Scaled translation vector:\n{T}")

        # Compute projection matrices
        proj1 = self.mtx1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        proj2 = self.mtx2 @ np.hstack((R, T))

        # Define the ArUco dictionary and parameters
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()

        # Main loop for real-time detection
        self.cap1 = cv2.VideoCapture(1, cv2.CAP_MSMF)
        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        self.cap2 = cv2.VideoCapture(2, cv2.CAP_MSMF)
        self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

    def __del__(self):
        """
        Release all resources used by the CameraMeasurements instance.

        This is called automatically when the object is garbage collected.
        """
        self.pipeline.stop()

    def find_pose(self, target_id):
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()
        if (not ret1 or not ret2):
            raise RuntimeError("Failed to capture frames from both cameras.")
        
        corners1, id1 = detect_aruco(frame1, target_id)
        corners2, id2 = detect_aruco(frame2, target_id)

        rvec1, tvec1 = estimate_pose(corners1, self.mtx1, self.dist1, self.marker_length)
        rvec2, tvec2 = estimate_pose(corners2, self.mtx2, self.dist2, self.marker_length)
        avg_rvec = average_rotation_vectors([rvec1, rvec2])
        avg_tvec = np.mean([tvec1, tvec2], axis=0)

        return avg_rvec, avg_tvec
    
    def measureBeamCenter(self, tvec):
        """
        Measure the beam center at the translation vector (tvec) 
        The beam center is the position of the 
        beam relative to the camera. The units of the beam center are 
        the same as the units of the translation vector.

        Parameters
        ----------
        tvec : numpy.ndarray
            The translation vector returned by find_pose.

        Returns
        -------
        beam_center : numpy.ndarray
            The position of the beam relative to the camera in the same
            units as the translation vector.
        """
        pass

    def validatePosition(self, tvec):
        """
        Validate the translation vector (tvec) by seeing if it will hit any optical elements

        Parameters
        ----------
        tvec : numpy.ndarray
            The translation vector 

        Returns
        -------
        valid : bool
            True if the translation vector is valid, False otherwise.
        """
        pass

    def measureBeamWidth(tvec):
        """
        Measure the beam width at the translation vector (tvec)

        Parameters
        ----------
        tvec : numpy.ndarray
            The translation vector returned by find_pose.

        Returns
        -------
        beam_width : float
            The width of the beam in the same units as the translation vector
        """
        pass
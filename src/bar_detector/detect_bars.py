#!/usr/bin/env python
# Author: Hrushikesh
__author__ = "Hrushikesh"

# from flytware_libs.capabilities.drone_state_manager.drone_state_manager import DroneStateManager
from bar_detector.hbar_detector import HBar_Detector
from bar_detector.vbar_detector import VBar_Detector
from bar_detector.srv import UpdateCkpt, UpdateCkptResponse, BD_State, BD_StateResponse
from bar_detector.srv import GimbalSet, GimbalSetRequest, GimbalSetResponse

from nav_controller.srv import PositionSetRequest, PositionSet, PositionSetResponse
from kittiseg_ros.msg import Result
from kittiseg_ros.srv import InitModel, InitModelRequest
from bar_detector.msg import BarDetectionResult, LineData, FakeImageResult
from geometry_msgs.msg import Point, TwistStamped
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import rospy
import message_filters
import tf.transformations
import numpy as np
import time

# done: create service to initialize bar_detection
# done: Internal function to set necessary rack facing and gimbal angle based on pose from init srv.
# done: provide service to pause and resume bar_detection
# done: Add rack class
# done: Add service call to start DL inference
# done: Make necessary changes to support resume after rack switching.
# done: Make vbar class independent of hbar class
# done: Increasing delta_yaw after switching gimbal or rack
# done: subscribe to checkpoints being published and switch gimbal and flags accordingly
# done: verify bd_yaw being published after rack switching
# done: update toggle_dir callback


class PseudoDronePose:
    def __init__(self, x, y, z, yaw=None, drone_facing=None):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw          # camera angle from vanishing point (-ve if vp at right)
        self.beta = np.pi/2     # camera angle from downward vector

        self.LEFT_RACK = -1
        self.AISLE = 0
        self.RIGHT_RACK = 1
        self.drone_facing = drone_facing

        self.GIMBAL_LEFT = -1
        self.GIMBAL_FRONT = 0
        self.GIMBAL_RIGHT = 1
        self.gimbal_direction = self.GIMBAL_RIGHT

        # change in successive frames
        self.delta_x = 0.3
        self.delta_y = 0.45
        self.delta_z = 0.3
        self.delta_yaw = np.deg2rad(0.1)
        self.delta_beta = np.deg2rad(0.1)

        # camera related attributes
        self.camera_matrix = self.__create_camera_matrix()
        self.Ki = np.linalg.inv(self.camera_matrix)
        self.Rotm = None
        self.vp_x = None
        self.vp_y = None

    def __create_camera_matrix(self):
        new_matrix = np.array(
            [[1541.210403, 0.0, 995.309011],
             [0.0, 1583.927544, 555.48413],
             [0, 0, 1]], dtype="double"
        )
        new_matrix *= 0.667
        new_matrix[2, 2] = 1
        return new_matrix

    def print_state(self, rack):
        print(" drone_facing: ", self.drone_facing, " gimbal_direction: ", self.gimbal_direction, " rack: ", rack)


class RackStructure:
    def __init__(self):
        self.aisle_width = 3.0
        self.rack_depth = 0
        self.level_height = 1.9
        # self.levels = np.arange(0, 9) * self.level_height
        self.hbar_ticks = np.array([0.3, 2.22, 4.46, 6.8, 8.55])
        # self.hbar_ticks = np.hstack((self.hbar_ticks[:-1], self.hbar_ticks[-1] + self.levels))
        self.bay_width = 1.02
        self.bay_count = 14
        self.vbar_ticks = np.arange(0, self.bay_count+1) * self.bay_width
        self.aisle_length = self.vbar_ticks[-1]   # only used for gimbal switching


class BarDetector:
    def __init__(self):

        self.PAUSED = 0
        self.RUNNING = 1
        self.IDLE = 2
        self.working_state = self.IDLE

        self.RS = RackStructure()
        self.NEAR = 0.25 * self.RS.aisle_length
        self.FAR = 0.75 * self.RS.aisle_length

        self.GIMBAL_ANGLE = np.radians(50)

        self.hbar_dl_model_name = 'skf_hbar_aview.pb'
        self.vbar_dl_model_name = 'kittiseg_vbar_aisle_view_updated.pb'
        ###########
        # self.pdp = PseudoDronePose(2.0, -0.0, 0.05, np.radians(-40))
        # self.pdp = PseudoDronePose(1.0, -0.6, 0.8, np.radians(-40))
        # self.pdp = PseudoDronePose(1.5, 4.0, 0.8, np.radians(-40))
        # self.pdp = PseudoDronePose(1.6, -0.2, 0, np.radians(0))
        self.pdp = None

        self.hbar_detector = None
        self.vbar_detector = None

        self.bridge = CvBridge()
        self.img_sub = None
        self.pub_for_vmask = None
        self.pub_for_hmask = None
        self.results_pub = None
        self.pose_pub = None
        self.debug_pub = None
        self.detection_img_pub = None
        self.image_header = None
        self.drone_yaw_pub = None
        self.toggle_dir_srv = None
        self.update_ckpt_srv = None

        self.debug = True
        self.compressed = False
        self.processed = True   # use without dl
        self.ckpt_y = 0

        self.state_change_srv = rospy.Service('/flytware/bar_detection/change_state', BD_State, self.state_change_cb)

        # self.init_subs_pubs_srvs()
        # self.init_subs_pubs_srvs_dl()

    def state_change_cb(self, req):
        if self.working_state == self.IDLE and req.Start_Resume is True:
            ''' ##### Start Bar Detection #####
            '''
            rospy.loginfo("[Bar_Detector] Starting bar detection")
            try:
                self.pdp = PseudoDronePose(req.X, req.Y, -req.Z)
                self.pdp.drone_facing = self.get_drone_facing(req.Yaw)
                if self.pdp.drone_facing is None:
                    return BD_StateResponse(success=False, message="[Bar_Detector] Invalid drone yaw")
                resp = self.gimbal_set()
                if not resp.success:
                    return BD_StateResponse(success=False, message=resp.message)
                self.pdp.yaw = self.get_bd_yaw(req.Yaw)

                success = self.init_subs_pubs_srvs_dl()
                if not success:
                    return BD_StateResponse(success=False, message="[Bar_Detector] Failed to load DL models")
                self.hbar_detector = HBar_Detector(self.pdp, self.RS)
                self.vbar_detector = VBar_Detector(self.pdp, self.RS)
                self.working_state = self.RUNNING
                return BD_StateResponse(success=True, message="[Bar_Detector] Bar Detection started")
            except rospy.ServiceException, e:
                rospy.logerr("gimbal set service call failed %s", e)
                return BD_StateResponse(success=False, message="[Bar_Detector] Failed to start Bar Detection")
        elif self.working_state == self.RUNNING and req.Start_Resume is False:
            ''' ##### Pause Bar Detection #####
            '''
            self.working_state = self.PAUSED
            rospy.loginfo("[Bar_Detector] Pausing Bar Detection")
            rospy.sleep(2.0)
            return BD_StateResponse(success=True, message="[Bar_Detector] Bar Detection Paused")
        elif self.working_state == self.PAUSED and req.Start_Resume is True:
            ''' ##### Resume Bar Detection #####
            '''
            rospy.loginfo("[Bar_Detector] Resuming Bar Detection")
            if req.RackSwitch:
                if self.pdp.drone_facing == self.pdp.LEFT_RACK:
                    self.pdp.drone_facing = self.pdp.RIGHT_RACK
                    self.pdp.delta_yaw = np.radians(180)
                    self.pdp.delta_beta = np.radians(5)
                elif self.pdp.drone_facing == self.pdp.RIGHT_RACK:
                    self.pdp.drone_facing = self.pdp.LEFT_RACK
                    self.pdp.delta_yaw = np.radians(180)
                    self.pdp.delta_beta = np.radians(5)
                else:
                    return BD_StateResponse(success=False, message="[Bar_Detector] Cannot switch Rack. Wrong condition")

            if req.GimbalSwitch:
                resp = self.gimbal_set(switch_gimbal=True)
            else:
                resp = self.gimbal_set()

            if resp.success:
                self.working_state = self.RUNNING
            else:
                return BD_StateResponse(success=False, message=resp.message)
            return BD_StateResponse(success=True, message="[Bar_Detector] Bar Detection Resumed")

        elif req.GimbalSwitch is True:
            ''' ##### Switch Gimbal Direction #####
            '''
            rospy.loginfo("[Bar_Detector] Switching gimbal direction")
            if self.working_state == self.RUNNING:
                self.working_state = self.PAUSED
            resp = self.gimbal_set(switch_gimbal=True)
            if resp.success:
                self.working_state = self.RUNNING
                return BD_StateResponse(success=True,
                                        message="[Bar_Detector] Gimbal switch complete, resuming bar detection")
            else:
                return BD_StateResponse(success=False,
                                        message="[Bar_Detector] Failed to Switch gimbal direction, bar detection paused")

        elif req.Stop is True:
            ''' ##### Stopping Bar Detection #####
            '''
            rospy.loginfo("[Bar_Detector] Stopping bar detection")
            self.working_state = self.IDLE
            # add dl de-init service here if required
            return BD_StateResponse(success=True, message="[Bar_Detector] Bar Detection Stopped")


        else:
            ''' Wrong condition
            '''
            rospy.loginfo("[Bar_Detector] Invalid request argument received")
            return BD_StateResponse(success=False, message="[Bar_Detector] Invalid request argument")

    def init_subs_pubs_srvs(self):      # without DL
        if self.compressed:
            self.img_sub = rospy.Subscriber('/flytos/flytcam/image_raw/compressed', CompressedImage, self.get_inference)
        else:
            self.img_sub = rospy.Subscriber('/flytos/flytcam/image_raw', Image, self.get_inference)

        ####### Start Pubs ######
        self.results_pub = rospy.Publisher('/bar_detector/lines', BarDetectionResult, queue_size=1)
        self.pose_pub = rospy.Publisher('mavros/local_position/local_tag', TwistStamped, queue_size=1)
        self.drone_yaw_pub = rospy.Publisher('yaw_publisher/yaw', TwistStamped, queue_size=1)
        self.debug_pub = rospy.Publisher('bar_detector/other', Pose, queue_size=1)
        self.detection_img_pub = rospy.Publisher('bar_detector/detection_overlay', Image, queue_size=1)

        self.update_ckpt_srv = rospy.Service('/flytware/bar_detection/update_ckpt', UpdateCkpt, self.update_ckpt_cb)
        # self.toggle_dir_srv = rospy.Service('/flytware/bar_detection/toggle_bd_dir', ToggleDir, self.switch_direction)

    def init_subs_pubs_srvs_dl(self):      # with DL

        ####### Start Inference ######
        rospy.loginfo("[bar_detection] Waiting for kittiseg_ros services")
        rospy.wait_for_service('/kittiseg_hbar_aisle_view/kittiseg_ros_manager/init_model')
        rospy.wait_for_service('/kittiseg_vbar_aisle_view/kittiseg_ros_manager/init_model')
        init_hbar_dl = rospy.ServiceProxy('/kittiseg_hbar_aisle_view/kittiseg_ros_manager/init_model', InitModel)
        req = InitModelRequest(model_name=self.hbar_dl_model_name, load_model=True)
        resp = init_hbar_dl(req)
        if not resp.success and not (resp.message == "model already loaded"):
            return False
        init_vbar_dl = rospy.ServiceProxy('/kittiseg_vbar_aisle_view/kittiseg_ros_manager/init_model', InitModel)
        req = InitModelRequest(model_name=self.vbar_dl_model_name, load_model=True)
        resp = init_vbar_dl(req)
        if not resp.success and not (resp.message == "model already loaded"):
            return False

        ####### Start Pubs Subs ######
        if self.img_sub is not None:    # Bar detection was stopped
            return True

        if self.compressed:
            self.img_sub = rospy.Subscriber('/flytos/flytcam/image_raw/compressed', CompressedImage, self.get_inference_dl)
        else:
            self.img_sub = rospy.Subscriber('/flytos/flytcam/image_raw', Image, self.get_inference_dl)
            # self.img_sub = rospy.Subscriber('/video_stream/input', Image, self.get_inference_dl)

        self.pub_for_vmask = rospy.Publisher('/kittiseg_vbar_aisle_view/video_stream/input', Image, queue_size=1)
        self.pub_for_hmask = rospy.Publisher('/kittiseg_hbar_aisle_view/video_stream/input', Image, queue_size=1)

        vbar_sub = message_filters.Subscriber('/kittiseg_vbar_aisle_view/kittiseg_ros_manager/results', Result)
        hbar_sub = message_filters.Subscriber('/kittiseg_hbar_aisle_view/kittiseg_ros_manager/results', Result)

        ts = message_filters.TimeSynchronizer([vbar_sub, hbar_sub], 20)
        # ts = message_filters.ApproximateTimeSynchronizer([vbar_sub, hbar_sub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.detect_lines)

        self.results_pub = rospy.Publisher('/bar_detector/lines', BarDetectionResult, queue_size=1)
        self.pose_pub = rospy.Publisher('/flytos/mavros/local_position/local_tag', TwistStamped, queue_size=1)
        self.drone_yaw_pub = rospy.Publisher('/flytos/yaw_publisher/yaw', TwistStamped, queue_size=1)
        self.debug_pub = rospy.Publisher('bar_detector/other', Pose, queue_size=1)
        self.detection_img_pub = rospy.Publisher('bar_detector/detection_overlay', Image, queue_size=1)

        self.update_ckpt_srv = rospy.Service('/flytware/bar_detection/update_ckpt', UpdateCkpt, self.update_ckpt_cb)
        return True

    def get_inference_dl(self, input_img_data):       # with DL
        if self.working_state == self.PAUSED:
            self.publish_pose()     # keep publishing previous pose
            return
        elif self.working_state == self.RUNNING:
            if self.compressed:
                # rospy.loginfo("======")
                image_data = self.bridge.compressed_imgmsg_to_cv2(input_img_data)
                image_data = cv2.resize(image_data, (1280, 720))   # 0.6667
                # image_data = cv2.resize(image_data, (1152, 648))   # 0.6
                # image_data = cv2.resize(image_data, (960, 540))  # 0.5
                # image_data = cv2.resize(image_data, (640, 360))  # 0.3334
                # image_data = cv2.resize(image_data, (576, 324))  # 0.3
                # image_data = cv2.resize(image_data, (0, 0), fx=0.5, fy=0.5)
                img_data = self.bridge.cv2_to_imgmsg(image_data, "bgr8")
                img_data.header = input_img_data.header
            else:
                image = self.bridge.imgmsg_to_cv2(input_img_data)
                image = cv2.resize(image, (1280, 720))  # 0.6667
                # image = cv2.resize(image, (640, 360))
                img_data = self.bridge.cv2_to_imgmsg(image, "bgr8")
                img_data.header = input_img_data.header
                # img_data = input_img_data
            img_data.header.stamp.nsecs = 0
            self.pub_for_vmask.publish(img_data)
            self.pub_for_hmask.publish(img_data)
        # else:
        #     rospy.loginfo("[bar_detection] bar detection in idle state]")

    def get_inference(self, input_img_data):      # without DL
        if self.working_state == self.PAUSED:
            self.publish_pose()  # keep publishing previous pose
            return
        if self.processed:
            self.processed = False
            self.image_header = input_img_data.header

            if self.compressed:
                image_data = self.bridge.compressed_imgmsg_to_cv2(input_img_data)
            else:
                image_data = self.bridge.imgmsg_to_cv2(input_img_data, "bgr8")

            vmask = self.get_mask(image_data, color='blue')
            hmask = self.get_mask(image_data, color='red')
            # rospy.sleep(0.35)
            self.detect_lines(image_data, vmask, hmask)

    def detect_lines(self, vbar_data, hbar_data):
    # def detect_lines(self, image_data, v_mask, h_mask):
        # rospy.loginfo(".")

        # st = str(hbar_data.header.stamp) + ", " + str(vbar_data.header.stamp) + ', ' + str(hbar_data.header.stamp - vbar_data.header.stamp)
        # rospy.loginfo(st)

        # with DL
        self.image_header = vbar_data.mask.header
        image_data = self.bridge.imgmsg_to_cv2(hbar_data.image)
        h_mask = self.bridge.imgmsg_to_cv2(hbar_data.mask)  # , "bgr8")
        v_mask = self.bridge.imgmsg_to_cv2(vbar_data.mask)  # , "bgr8")

        overlay_img = image_data.copy()
        # overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_GRAY2BGR)

        left_h_lines, left_h_conf, left_h_hts, right_h_lines, right_h_conf, right_h_hts, overlay_img = \
            self.hbar_detector.detect_lines(h_mask, overlay_img)
        results_msg = BarDetectionResult()

        for i, line in enumerate(left_h_lines):
            line_msg = LineData()

            img_coord = Point()
            img_coord.x = line[0][0]
            img_coord.y = line[0][1]
            line_msg.point1 = img_coord

            img_coord = Point()
            img_coord.x = line[1][0]
            img_coord.y = line[1][1]
            line_msg.point2 = img_coord
            if self.debug:
                overlay_img = cv2.line(overlay_img, (self.pdp.vp_x, self.pdp.vp_y),
                                       (line[1][0], line[1][1]), (0, 255, 0), 4)

            line_msg.confidence = left_h_conf[i]
            line_msg.real_coord = left_h_hts[i]
            results_msg.left_hbar.append(line_msg)

        for i, line in enumerate(right_h_lines):
            line_msg = LineData()

            img_coord = Point()
            img_coord.x = line[0][0]
            img_coord.y = line[0][1]
            line_msg.point1 = img_coord

            img_coord = Point()
            img_coord.x = line[1][0]
            img_coord.y = line[1][1]
            line_msg.point2 = img_coord
            if self.debug:
                overlay_img = cv2.line(overlay_img, (line[0][0], line[0][1]),
                                       (self.pdp.vp_x, self.pdp.vp_y), (0, 0, 255), 4)

            line_msg.confidence = right_h_conf[i]
            line_msg.real_coord = right_h_hts[i]
            results_msg.right_hbar.append(line_msg)

        overlay_img, left_v_lines, left_v_conf, right_v_lines, right_v_conf = \
            self.vbar_detector.detect_lines(v_mask, overlay_img)
        # print(len(left_h_lines), len(right_h_lines), len(left_v_lines), len(right_v_lines))
        # print("=", self.hbar_detector.dist_from_right, np.rad2deg(self.hbar_detector.yaw))

        for i, line in enumerate(left_v_lines):
            line_msg = LineData()

            img_coord = Point()
            img_coord.x = line[0][0]
            img_coord.y = line[0][1]
            line_msg.point1 = img_coord

            img_coord = Point()
            img_coord.x = line[1][0]
            img_coord.y = line[1][1]
            line_msg.point2 = img_coord
            if self.debug:
                overlay_img = cv2.line(overlay_img, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (255, 0, 255),
                                       4)

            line_msg.confidence = left_v_conf[i]
            line_msg.real_coord = 0
            results_msg.left_vbar.append(line_msg)

        for i, line in enumerate(right_v_lines):
            line_msg = LineData()

            img_coord = Point()
            img_coord.x = line[0][0]
            img_coord.y = line[0][1]
            line_msg.point1 = img_coord

            img_coord = Point()
            img_coord.x = line[1][0]
            img_coord.y = line[1][1]
            line_msg.point2 = img_coord
            if self.debug:
                overlay_img = cv2.line(overlay_img, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (0, 255, 255),
                                       4)

            line_msg.confidence = right_v_conf[i]
            line_msg.real_coord = 0
            results_msg.right_vbar.append(line_msg)

        msg = self.bridge.cv2_to_imgmsg(overlay_img, "bgr8")
        msg.header = self.image_header
        self.detection_img_pub.publish(msg)

        # compressed image and hbar and vbar lines
        results_msg.header.stamp = self.image_header.stamp
        results_msg.mission_id = 'A123'
        # img_dat = CompressedImage()
        # img_dat.header.stamp = img_data.header.stamp
        # img_dat.format = "jpeg"
        # img_dat.data = np.array(cv2.imencode('.jpg', image_data)[1]).tostring()
        results_msg.frame = self.bridge.cv2_to_imgmsg(image_data)
        self.results_pub.publish(results_msg)

        # Pose with x, y, z and yaw
        self.publish_pose()

        if self.debug:
            print('x:= %2.3f \ty:= %2.3f \tz:= %2.3f \tyaw:= %2.3f' % (round(self.pdp.x, 2),
                                                                       round(self.pdp.y, 2),
                                                                       round(self.pdp.z, 2),
                                                                       round(np.rad2deg(self.pdp.yaw), 2)))
            da = Pose()
            da.position.x = self.pdp.x
            da.position.y = self.pdp.y
            da.position.z = self.pdp.z
            da.orientation.w = self.pdp.yaw
            # da.orientation.x = self.hbar_detector.dist_from_right + self.hbar_detector.dist_from_left

            self.debug_pub.publish(da)
        self.processed = True

    def publish_pose(self):
        pose_msg = TwistStamped()
        pose_msg.header = self.image_header
        pose_msg.twist.linear.x = self.pdp.x
        pose_msg.twist.linear.y = self.pdp.y
        pose_msg.twist.linear.z = - self.pdp.z

        if self.pdp.drone_facing != self.pdp.AISLE:
            vp_dir = -1 * self.pdp.drone_facing * self.pdp.gimbal_direction
        else:  # drone facing into the aisle
            vp_dir = 1
        pose_msg.twist.angular.z = (vp_dir * np.pi/2) + (self.pdp.yaw - (self.pdp.gimbal_direction * self.GIMBAL_ANGLE))

        if self.pdp.gimbal_direction == 1:  # Forward
            pose_msg.twist.linear.x += 0.05
        else:  # Backward
            pose_msg.twist.linear.x -= 0.05
        self.pose_pub.publish(pose_msg)

        self.yaw_data = TwistStamped()
        self.yaw_data.twist.linear.z = pose_msg.twist.angular.z
        self.drone_yaw_pub.publish(self.yaw_data)

    def gimbal_set(self, ckpt_y=None, switch_gimbal=False):
        rospy.loginfo("[bar_detection] setting up the gimbal")
        if self.pdp.drone_facing == self.pdp.AISLE:
            req_gimbal_direction = self.pdp.GIMBAL_FRONT
        elif switch_gimbal is True:
            if self.pdp.gimbal_direction == self.pdp.GIMBAL_RIGHT:
                req_gimbal_direction = self.pdp.GIMBAL_LEFT
            elif self.pdp.gimbal_direction == self.pdp.GIMBAL_LEFT:
                req_gimbal_direction = self.pdp.GIMBAL_RIGHT
            else:
                print("[bar_detector] Cannot switch gimbal while looking forward")
                req_gimbal_direction = self.pdp.gimbal_direction
        else:
            y = self.pdp.y if ckpt_y is None else ckpt_y
            if (y > self.FAR and self.pdp.drone_facing == self.pdp.LEFT_RACK) or \
                    (y < self.NEAR and self.pdp.drone_facing == self.pdp.RIGHT_RACK):
                req_gimbal_direction = self.pdp.GIMBAL_LEFT
            elif (y > self.FAR and self.pdp.drone_facing == self.pdp.RIGHT_RACK) or \
                    (y < self.NEAR and self.pdp.drone_facing == self.pdp.LEFT_RACK):
                req_gimbal_direction = self.pdp.GIMBAL_RIGHT
            else:
                req_gimbal_direction = self.pdp.gimbal_direction
        try:
            rospy.wait_for_service('/flytos/gimbal_control/set_angle')
            handle = rospy.ServiceProxy('/flytos/gimbal_control/set_angle', GimbalSet)
            # rospy.loginfo("Gimbal Set called: Roll: %f, Pitch: %f, Yaw: %f" % (roll, pitch, yaw))
            if self.pdp.gimbal_direction != req_gimbal_direction:
                print("[bar_detector] set angle to 0")
                req_msg = GimbalSetRequest(roll=0, pitch=0, yaw=0, mode=0, time=2.5)
                resp = handle(req_msg)
                self.pdp.delta_yaw = np.radians(180)
                self.pdp.delta_beta = np.radians(5)
            else:
                self.pdp.delta_yaw = np.radians(10)
                self.pdp.delta_beta = np.radians(5)
            self.pdp.gimbal_direction = req_gimbal_direction
            yaw = self.pdp.gimbal_direction * self.GIMBAL_ANGLE
            print("[bar_detector] setting angle", np.degrees(yaw))
            req_msg = GimbalSetRequest(roll=0, pitch=0, yaw=yaw, mode=0, time=2.5)
            resp = handle(req_msg)
            rospy.loginfo("[bar_detector] gimbal set service call response: " + str(resp))
            rospy.sleep(2.0)
            return GimbalSetResponse(success=resp.success, message=resp.message)

        except rospy.ServiceException, e:
            rospy.logerr("[bar_detector] gimbal set service call failed %s", e)
            return GimbalSetResponse(success=False, message="gimbal set service call failed")

    def get_mask(self, img, color='blue'):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if color == 'blue':
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
        else:
            # lower red  mask (0-10)
            lower_red = np.array([0, 150, 50])
            upper_red = np.array([5, 255, 255])
            mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

            # upper red mask (170-180)
            lower_red = np.array([170, 140, 60])
            upper_red = np.array([180, 255, 255])
            mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
            mask = mask0 + mask1

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img[np.where(mask == 0)] = 0
        gray_img[np.where(mask != 0)] = 255
        kernel = np.ones((5, 5), np.uint8)
        output_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
        return output_img

    def get_drone_facing(self, drone_yaw):
        if 0 <= drone_yaw <= np.pi/3:           # left rack facing
            return self.pdp.LEFT_RACK
        elif np.pi/3 < drone_yaw < 2*np.pi/3:   # aisle view facing
            return self.pdp.AISLE
        elif 2*np.pi/3 <= drone_yaw <= np.pi:   # right rack facing
            return self.pdp.RIGHT_RACK
        else:
            return None

    def get_bd_yaw(self, drone_yaw):
        gimbal_angle_wh = drone_yaw + (self.pdp.gimbal_direction * self.GIMBAL_ANGLE)
        gimbal_in_bd = gimbal_angle_wh - np.pi/2
        if gimbal_in_bd > np.pi/2:
            gimbal_in_bd -= np.pi
        elif gimbal_in_bd < -np.pi/2:
            gimbal_in_bd += np.pi
        return gimbal_in_bd

    def update_ckpt_cb(self, data):
        rospy.loginfo("[bar_detection] setting gimbal using received checkpoint")
        self.ckpt_y = data.ckpt_y
        if self.working_state == self.RUNNING:
            self.working_state = self.PAUSED
        resp = self.gimbal_set(ckpt_y=self.ckpt_y)
        if resp.success:
            self.working_state = self.RUNNING
        return UpdateCkptResponse(status=resp.success)

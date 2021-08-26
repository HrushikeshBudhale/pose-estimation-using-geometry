#!/usr/bin/env python
# Author: Hrushikesh
__author__ = "Hrushikesh"

import cv2
import numpy as np
# import math


class VBar_Detector:
    def __init__(self, pdp, rs):
        self.dp = pdp
        self.rs = rs
        self.rows = None
        self.debug = True

    def detect_lines(self, img_mask, img):
        # img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img_mask, (3, 3), 0)
        ret3, img_threshed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        _, contours, hierarchy = cv2.findContours(img_threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.rows, cols = img_mask.shape[:2]

        left_lines = []
        right_lines = []
        left_confidences = []
        right_confidences = []
        if self.dp.vp_x is None:
            print("vanishing point not given")
            return img, left_lines, left_confidences, right_lines, right_confidences

        aspect_ratios = []
        centers = []
        lengths = []
        lines = []
        valid_contours = []
        for contour in contours:
            M = cv2.moments(contour)
            xc = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
            yc = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0

            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            if vy == 0:
                vy = 1e-07
            # top  = int((-y*vx/vy) + x)
            # bottom = int(((rows-y)*vx/vy) + x)

            rect = cv2.minAreaRect(contour)
            length, aspect_ratio = self.__process_rect(rect)
            # angle = np.rad2deg(np.arctan2(bottom - top, (rows-1)))

            aspect_ratios.append(aspect_ratio)
            centers.append([xc, yc])
            lengths.append(length)
            # lines.append([[top, 0], [bottom, rows-1]])
            lines.append([[xc, 0], [xc, self.rows - 1]])
            # angles.append(angle)
            valid_contours.append(contour)

        centers = np.array(centers)
        lines = np.array(lines)
        lengths = np.array(lengths)
        aspect_ratios = np.array(aspect_ratios)
        sorted_lengths = (-lengths).argsort()[:]

        length_thresh = 70
        aspect_ratio_thresh = 3
        # angle_thresh = 6

        valid_blobs = []
        invalid_blobs = []
        for index in sorted_lengths:
            if aspect_ratios[index] > aspect_ratio_thresh and lengths[index] > length_thresh and centers[index][1] > 10:
                valid_blobs.append(index)
            elif aspect_ratios[index] > aspect_ratio_thresh:
                invalid_blobs.append(index)

        invalid_blobs = np.asarray(invalid_blobs)
        for index in invalid_blobs:
            similar_blobs = np.argwhere(abs(centers[invalid_blobs] - centers[index])[:, 0] < 20)
            if len(similar_blobs) > 1:
                invalid_blobs = np.delete(invalid_blobs, similar_blobs.T[0])
                valid_blobs.append(index)

        counted_x = [5000]
        blob_indexes = []
        dist_thresh = 15
        for index in valid_blobs:
            closest = min(counted_x, key=lambda x: abs(x - centers[index][0]))
            if abs(centers[index][0] - closest) > dist_thresh:
                counted_x.append(centers[index][0])
                blob_indexes.append(index)

        valid_lines = lines[blob_indexes]
        valid_confidences = lengths[blob_indexes] * aspect_ratios[blob_indexes]
        if valid_lines is not None and len(valid_lines) > 0:
            xs = valid_lines[:, 0, 0]

            left_sorted_indxs = xs.argsort()[:]
            sorted_lines = valid_lines[left_sorted_indxs]
            sorted_confidences = valid_confidences[left_sorted_indxs]
            left_line_indxs = np.where(sorted_lines[:, 0, 0] <= self.dp.vp_x)
            left_lines = sorted_lines[left_line_indxs]
            left_confidences = sorted_confidences[left_line_indxs]
            left_confidences = left_confidences / sum(left_confidences)

            if len(left_lines) > 0 and self.dp.Rotm is not None:
                self.measure_y_from_nearest_vbar(left_lines, rack=-1)

            if self.debug:
                left_pxs = []
                left_bar_nos = []
                if len(left_lines) > 0:
                    left_pxs, left_bar_nos = self.get_expected_vbar_px(rack=-1)  # left

                font = cv2.FONT_HERSHEY_SIMPLEX
                for i, px in enumerate(left_pxs):
                    if i > len(left_lines):
                        break
                    if px < 0:
                        continue
                    img = cv2.circle(img, (int(px), int(img.shape[0]/2)), 6, (0, 0, 255), -1)
                    cv2.putText(img, str(left_bar_nos[i]), (int(px), int(img.shape[0]/2)), font, 1.4,
                                (255, 255, 0), 2, cv2.LINE_AA)

            right_sorted_indxs = (-xs).argsort()[:]
            sorted_lines = valid_lines[right_sorted_indxs]
            sorted_confidences = valid_confidences[right_sorted_indxs]
            right_line_indxs = np.where(sorted_lines[:, 0, 0] > self.dp.vp_x)
            right_lines = sorted_lines[right_line_indxs]
            right_confidences = sorted_confidences[right_line_indxs]
            right_confidences = right_confidences / sum(right_confidences)

            if len(right_lines) > 0 and self.dp.Rotm is not None:
                self.measure_y_from_nearest_vbar(right_lines, rack=1)

            if self.debug:
                right_pxs = []
                right_bar_nos = []
                if len(right_lines) > 0:
                    right_pxs, right_bar_nos = self.get_expected_vbar_px(rack=1)  # right

                font = cv2.FONT_HERSHEY_SIMPLEX
                for i, px in enumerate(right_pxs):
                    if i > len(right_lines):
                        break
                    if px > img.shape[1]:
                        continue
                    img = cv2.circle(img, (int(px), int(img.shape[0]/2)), 6, (0, 0, 255), -1)
                    cv2.putText(img, str(right_bar_nos[i]), (int(px), int(img.shape[0]/2)), font, 1.4,
                                (255, 255, 0), 2, cv2.LINE_AA)
        else:
            print("no vertical bar Detected!")

        return img, left_lines, left_confidences, right_lines, right_confidences

    def __process_rect(self, rect):
        breadth = min(rect[1])
        length = max(rect[1])
        aspect_ratio = float(length) / breadth if breadth != 0 else 1
        return length, aspect_ratio

    def get_expected_vbar_px(self, rack):
        '''
        rack = -1    represents left rack in image
        rack = 1     represents right rack in image
        '''

        if self.dp.drone_facing == self.dp.LEFT_RACK or (self.dp.drone_facing == self.dp.AISLE and rack == -1):
            dist_from_rack = self.rs.aisle_width + self.rs.rack_depth - self.dp.x
        elif self.dp.drone_facing == self.dp.RIGHT_RACK or (self.dp.drone_facing == self.dp.AISLE and rack == 1):
            dist_from_rack = self.dp.x - self.rs.rack_depth
        else:
            dist_from_rack = None
            self.dp.print_state(rack)
            print("wrong condition! - 4")
        dist_from_bar = self.rs.vbar_ticks - self.dp.y
        # print("vv", dist_from_rack, dist_from_bar)
        # else:
        #     dist_from_bar = self.dp.y - self.rs.vbar_ticks
        # bar_no = 0
        # for bar_no, bar_y in enumerate(self.rs.vbar_ticks):
        #     if (bar_y - self.dp.y) > 0:
        #         break
        # print("==", dist_from_rack, dist_from_bar)
        angles_without_yaw = np.arctan2(dist_from_rack, dist_from_bar)    # measured a-clk-wise from vanishing point
        # print("---", np.degrees(angles_without_yaw))
        vp_dir = abs(self.dp.drone_facing + self.dp.gimbal_direction)   # value is 0 if forward else 2
        if vp_dir == 0:     # forward
            angles = (rack * angles_without_yaw) - self.dp.yaw
        else:   # vp_dir == 2 (backward)
            angles = (-1 * rack * angles_without_yaw) - (self.dp.yaw - np.pi)
        bar_nos = np.arange(len(angles))
        x_homogeneous = np.sin(angles)
        y_homogeneous = np.zeros([1, x_homogeneous.shape[0]])
        z_homogeneous = np.cos(angles)
        homogeneous_coords = np.vstack((x_homogeneous, y_homogeneous, z_homogeneous))

        # remove points behind camera
        # print(">>>", homogeneous_coords, bar_nos)
        homogeneous_coords = homogeneous_coords[:, z_homogeneous > 0]
        bar_nos = bar_nos[z_homogeneous > 0]
        pixel_coords = self.dp.camera_matrix.dot(homogeneous_coords)
        pixel_coords[0, :] = pixel_coords[0, :] / pixel_coords[2, :]
        # print(pixel_coords, bar_nos)
        return pixel_coords[0, :], bar_nos

        # rev_dir = True
        # if rev_dir:

    def measure_y_from_nearest_vbar(self, lines, rack):
        '''
        rack = -1    represents left rack in image
        rack = 1     represents right rack in image
        '''
        if self.dp.drone_facing == self.dp.LEFT_RACK or (self.dp.drone_facing == self.dp.AISLE and rack == -1):
            dist_from_rack = self.rs.aisle_width + self.rs.rack_depth - self.dp.x
        elif self.dp.drone_facing == self.dp.RIGHT_RACK or (self.dp.drone_facing == self.dp.AISLE and rack == 1):
            dist_from_rack = self.dp.x - self.rs.rack_depth
        else:
            dist_from_rack = None
            self.dp.print_state(rack)
            print("wrong condition! - 3")
        x_px = lines[:, 0, 0]
        y_px = np.ones([1, x_px.shape[0]])*int(self.rows/2)
        z_px = np.ones([1, x_px.shape[0]])
        img_pts = np.vstack((x_px, y_px, z_px))
        real_coords = self.dp.Ki.dot(img_pts)
        real_coords = np.divide(real_coords, np.linalg.norm(real_coords, axis=0))   # in camera frame

        vp = np.array([0, 0, 1])   # vp in real world (vector facing into the aisle)
        vp = self.dp.Rotm.dot(vp)    # vp in camera frame
        vp = vp.reshape(3, 1)
        line_yaws = np.arccos(np.clip(real_coords.T.dot(vp), -1.0, 1.0))    # wrt vp vector into the aisle
        dists_from_bars = dist_from_rack / np.tan(line_yaws)
        vp_dir = abs(self.dp.drone_facing + self.dp.gimbal_direction)  # value is 0 if forward else 2
        if vp_dir == 0:  # forward
            bars_ys = (dists_from_bars[:, 0] + self.dp.y)
        else:
            bars_ys = (self.dp.y - dists_from_bars[::-1, 0])    # taken in reverse order
        self.__update_y(bars_ys, rack)

    def __update_y(self, bars_ys, rack):
        bars_ys = bars_ys[bars_ys < 3.1 + self.dp.y]  # consider bars till n meters from drone
        # print(bars_ys)
        a1 = np.tile(bars_ys, (self.rs.vbar_ticks.shape[0], 1)).T
        b1 = np.tile(self.rs.vbar_ticks, (bars_ys.shape[0], 1))
        match_mat1 = a1 - b1
        np.set_printoptions(suppress=True, linewidth=np.nan)
        err_inds = np.absolute(match_mat1).argmin(axis=1)
        match_ticks = self.rs.vbar_ticks[np.absolute(match_mat1).argmin(axis=1)]

        error = np.inf
        diff = 0
        for i, err_ind in enumerate(err_inds):
            match_err = match_mat1[i, err_ind]
            adjusted_ticks = match_ticks - match_err
            a = np.tile(bars_ys, (adjusted_ticks.shape[0], 1)).T
            b = np.tile(adjusted_ticks, (bars_ys.shape[0], 1))
            match_mat = np.absolute(a - b)
            valid_ticks_inds = match_mat.argmin(axis=1)
            # valid_bar_inds = match_mat[:, valid_ticks_inds].argmin(axis=0)
            valid_errors = match_mat[:, valid_ticks_inds].min(axis=0)
            tot_err = np.sum(valid_errors)
            if tot_err < error:
                error = tot_err
                diff = match_err
        diff = np.clip(diff, -self.dp.delta_y, self.dp.delta_y)

        if (self.dp.drone_facing != self.dp.AISLE) and (self.dp.gimbal_direction * rack) == -1:
            self.dp.y -= diff
        elif (self.dp.drone_facing == self.dp.AISLE) and rack == self.dp.LEFT_RACK:
            pass
            # self.dp.y -= diff
        elif (self.dp.drone_facing == self.dp.AISLE) and rack == self.dp.RIGHT_RACK:
            # pass
            self.dp.y -= diff
        else:
            self.dp.print_state(rack)
            print("wrong condition - 5")
        # print("y:= ", self.dp.y, "\terror:= ", diff)

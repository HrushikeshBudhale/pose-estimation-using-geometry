#!/usr/bin/env python
# Author: Hrushikesh
__author__ = "Hrushikesh"

import cv2
import numpy as np
import math


class HBar_Detector:
    def __init__(self, pdp, rs):
        self.dp = pdp
        self.rs = rs
        self.debug = False

    def detect_lines(self, img_mask, overlay_img):
        # img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img_mask, (5, 5), 0)
        ret3, img_threshed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        _, contours, hierarchy = cv2.findContours(img_threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        aspect_ratios = []
        centers = []
        lengths = []
        angles = []
        lines = []
        valid_contours = []
        rows, cols = img_mask.shape[:2]

        for contour in contours:
            M = cv2.moments(contour)
            xc = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
            yc = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0

            [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            lefty = int((-x * vy / vx) + y)
            righty = int(((cols - x) * vy / vx) + y)

            rect = cv2.minAreaRect(contour)
            length, aspect_ratio = self.__process_rect(rect)
            angle = np.rad2deg(np.arctan2(righty - lefty, cols - 1))

            aspect_ratios.append(aspect_ratio)
            centers.append([xc, yc])
            lengths.append(length)
            lines.append([[cols - 1, righty], [0, lefty]])
            angles.append(angle)
            valid_contours.append(contour)

        centers = np.array(centers)
        lines = np.array(lines)
        lengths = np.array(lengths)
        aspect_ratios = np.array(aspect_ratios)
        valid_contours = np.array(valid_contours)
        sorted_lengths = (-lengths).argsort()[:]

        dist_thresh = 40
        aspect_ratio_thresh = 7
        angle_thresh = 0
        valid_lanes = []  # stores indexes
        valid_blobs = []  # stores indexes

        counted_angles = [90]  # place holder
        for index in sorted_lengths:
            closest = min(counted_angles, key=lambda x: abs(x - angles[index]))
            if abs(angles[index] - closest) > angle_thresh and aspect_ratios[index] > aspect_ratio_thresh and \
                    lengths[index] > dist_thresh:
                valid_lanes.append(index)
                counted_angles.append(angles[index])
            if aspect_ratios[index] > 1 and lengths[index] > 40:
                valid_blobs.append(index)
        counted_angles.pop(0)  # place holder removed

        left_lines = []
        left_confidences = []
        left_actual_hts = []
        right_lines = []
        right_confidences = []
        right_actual_hts = []

        if len(valid_lanes) > 1:
            valid_centers = centers[valid_lanes]
            # valid_aspect_ratios = aspect_ratios[valid_blobs]
            # valid_lengths = lengths[valid_blobs]
            # valid_confidences = valid_aspect_ratios * valid_lengths

            self.dp.vp_x, self.dp.vp_y = self.__get_average_intersection_point(lines[valid_lanes], lengths[valid_lanes])
            self.dp.Rotm, self.dp.yaw, self.dp.vp_x, self.dp.vp_y = self.__compute_rotm_yaw([self.dp.vp_x,
                                                                                             self.dp.vp_y])

            dist_from_left = 0
            left_dist_conf = 0
            if self.dp.gimbal_direction != self.dp.GIMBAL_LEFT:
                left_centers = np.argwhere(valid_centers[:, 0] < self.dp.vp_x)
                dist_from_left, left_dist_conf, left_actual_hts, left_new_ps = self.__distance_from_rack(
                    valid_centers[left_centers], [self.dp.vp_x, self.dp.vp_y], -1)
                left_lines = self.__get_end2end_lines(left_new_ps, [self.dp.vp_x, self.dp.vp_y], cols)
                left_confidences = [0] * len(left_lines)
                if left_dist_conf > 0:
                    if self.dp.drone_facing == self.dp.LEFT_RACK:
                        self.dp.x = self.rs.aisle_width + self.rs.rack_depth - dist_from_left
                    elif self.dp.drone_facing == self.dp.RIGHT_RACK:
                        self.dp.x = self.rs.rack_depth + dist_from_left

            dist_from_right = 0
            right_dist_conf = 0
            if self.dp.gimbal_direction != self.dp.GIMBAL_RIGHT:
                right_centers = np.argwhere(valid_centers[:, 0] > self.dp.vp_x)

                dist_from_right, right_dist_conf, right_actual_hts, right_new_ps = self.__distance_from_rack(
                    valid_centers[right_centers], [self.dp.vp_x, self.dp.vp_y], 1)
                right_lines = self.__get_end2end_lines(right_new_ps, [self.dp.vp_x, self.dp.vp_y], cols)
                right_confidences = [0] * len(right_lines)
                if right_dist_conf > 0:
                    if self.dp.drone_facing == self.dp.RIGHT_RACK:
                        self.dp.x = self.rs.rack_depth + dist_from_right
                    elif self.dp.drone_facing == self.dp.LEFT_RACK:
                        self.dp.x = self.rs.aisle_width + self.rs.rack_depth - dist_from_right

            if self.dp.gimbal_direction == self.dp.GIMBAL_FRONT and (left_dist_conf + right_dist_conf) > 0:
                self.dp.x = np.average([dist_from_left, dist_from_right], weights=[left_dist_conf, right_dist_conf])

        if self.debug:
            print(len(left_lines))
            overlay_img = cv2.drawContours(overlay_img, valid_contours, -1, (0, 255, 0), 3)
            for line in lines[valid_lanes]:
                overlay_img = cv2.line(overlay_img, (line[0][0], line[0][1]),
                                       (line[1][0], line[1][1]), (0, 0, 255), 4)
            overlay_img = cv2.circle(overlay_img, (int(self.dp.vp_x), int(self.dp.vp_y)), 6, (255, 0, 0), -1)

        return left_lines, left_confidences, left_actual_hts,\
               right_lines, right_confidences, right_actual_hts, overlay_img

    def __process_rect(self, rect):
        breadth = min(rect[1])
        length = max(rect[1])
        aspect_ratio = float(length) / breadth if breadth != 0 else 1
        return length, aspect_ratio

    def __get_average_intersection_point(self, lines, lengths):
        lp1 = np.vstack([lines[:, 0, :].T, np.ones(lines.shape[0])])
        lp2 = np.vstack([lines[:, 1, :].T, np.ones(lines.shape[0])])

        lp1 = self.dp.Ki.dot(lp1)
        lp2 = self.dp.Ki.dot(lp2)

        lp1 = np.divide(lp1, np.linalg.norm(lp1, axis=0))  # normalize
        lp2 = np.divide(lp2, np.linalg.norm(lp2, axis=0))

        line_vec = np.cross(lp1, lp2, axis=0)
        line_vec = np.divide(line_vec, np.linalg.norm(line_vec, axis=0))

        t1 = np.repeat(line_vec, line_vec.shape[1], axis=1)
        t2 = np.tile(line_vec, line_vec.shape[1])

        linelen1 = np.repeat(lengths, lengths.shape[0], axis=0)
        linelen2 = np.tile(lengths, lengths.shape[0])
        weights = np.multiply(linelen1, linelen2)

        vps = np.cross(t1, t2, axis=0)
        vp_norms = np.linalg.norm(vps, axis=0)
        nzis = np.where(vp_norms != 0)[0]
        vps = np.divide(vps[:, nzis], vp_norms[nzis])

        weights = weights[nzis]

        vps = self.dp.camera_matrix.dot(vps)  # convert to pixel coordinates
        vps[0, :] = vps[0, :] / vps[2, :]
        vps[1, :] = vps[1, :] / vps[2, :]
        if self.dp.vp_x is not None:
            # print(len(weights), len(vps), vps[0, :], vps[1, :])
            weights = 1.0 / np.sqrt((vps[0, :] - self.dp.vp_x)**2 + (vps[1, :] - self.dp.vp_y)**2)
            # print("**", weights)
        normalized_weights = weights / np.sum(weights)
        average_x = np.average(vps[0, :], weights=normalized_weights)
        average_y = np.average(vps[1, :], weights=normalized_weights)
        return int(average_x), int(average_y)

    def __compute_rotm_yaw(self, p1):
        p1 = np.array([p1[0], p1[1], 1])
        x = self.dp.Ki.dot(p1.T)
        x = x / np.linalg.norm(x)   # x = [sin(alpha)*sin(beta), cos(beta), cos(alpha)*sin(beta)]
        beta = math.acos(x[1])      # beta measured from downward direction (pitch)
        # beta = min(np.pi/2, beta)
        # beta = max(min(np.pi/1.6, beta), np.pi/2.4)
        self.dp.beta = np.clip(beta, self.dp.beta-self.dp.delta_beta, self.dp.beta+self.dp.delta_beta)
        yaw = -math.asin(x[0] / math.sin(self.dp.beta))
        pitch = np.pi / 2 - self.dp.beta  # removes gimbal lock
        # print(np.rad2deg(pitch), np.rad2deg(yaw))
        Rx_mat = np.array([[1, 0, 0],
                           [0, math.cos(pitch), -math.sin(pitch)],
                           [0, math.sin(pitch), math.cos(pitch)]])
        # Rz_mat = np.array([[math.cos(roll), -math.sin(roll), 0],
        #                    [math.sin(roll),  math.cos(roll), 0],
        #                    [             0,               0, 1]])

        clipped_yaw = np.clip(yaw, self.dp.yaw-self.dp.delta_yaw, self.dp.yaw+self.dp.delta_yaw)
        if self.dp.delta_yaw > np.radians(90):
            self.dp.delta_yaw = np.radians(0.1)
            self.dp.delta_beta = np.radians(0.1)
        # print("----", np.degrees(yaw), "\t", np.degrees(clipped_yaw), "\t", np.degrees(self.dp.yaw))

        Ry_mat = np.array([[math.cos(clipped_yaw),  0, math.sin(clipped_yaw)],
                            [0,              1,             0],
                            [-math.sin(clipped_yaw), 0, math.cos(clipped_yaw)]])
        rotm = Rx_mat.T.dot(Ry_mat.T)

        # compute new vp
        x = np.array([math.sin(-clipped_yaw)*math.sin(self.dp.beta), math.cos(self.dp.beta),
                      math.cos(-clipped_yaw)*math.sin(self.dp.beta)])
        p2 = self.dp.camera_matrix.dot(x.T)
        p2 = np.asarray(p2/p2[2], dtype=int)
        # print("p1:= ", p1)
        # print("p2:= ", p2)
        return rotm, clipped_yaw, p2[0], p2[1]

    def __solve_ASA(self, angle_pairs, dist_c):
        A = np.pi - np.max(angle_pairs, axis=1)  # lower angle
        B = np.min(angle_pairs, axis=1)  # higher angle
        # Ad = np.rad2deg(angle_pairs)
        # Bd = np.rad2deg(B)
        # print(Ad, dist_c)

        C = np.pi - (A + B)
        K = dist_c / np.sin(C)
        dist_a = np.sin(A) * K
        dist_b = np.sin(B) * K
        s = (dist_a + dist_b + dist_c) / 2
        area = np.sqrt(s * (s - dist_a) * (s - dist_b) * (s - dist_c))
        # area = 1/2 * base * height
        height = 2 * (area / dist_c)
        confidence = abs(np.sin(C))
        return height, confidence

    def __distance_from_rack(self, ps, vp, rack):
        '''
        rack = -1    represents left rack in image
        rack = 1     represents right rack in image
        '''
        # print('---', ps)
        if len(ps) == 0:
            return -1, 0, [], []
        vp = self.dp.Ki.dot(np.array([vp[0], vp[1], 1]))
        vp = vp / np.linalg.norm(vp)

        vp2 = np.array([0, 1, 0])  # vp at bottom
        vp2 = self.dp.Rotm.dot(vp2)  # vp in camera frame
        yz_plane_vec = np.cross(vp2, vp)
        yz_plane_vec = yz_plane_vec / np.linalg.norm(yz_plane_vec)

        lps = np.vstack([ps[:, 0].T, np.ones(ps.shape[0])])
        lps = self.dp.Ki.dot(lps)
        lps = np.divide(lps, np.linalg.norm(lps, axis=0))

        vp = vp.reshape(3, 1)
        vps = np.tile(vp, lps.shape[1])
        line_vec = np.cross(lps, vps, axis=0)
        line_vec = np.divide(line_vec, np.linalg.norm(line_vec, axis=0))

        line_roll = np.arccos(line_vec.T.dot(yz_plane_vec))

        rolls = [0]
        new_ps = []
        for i, roll in enumerate(line_roll):
            closest = min(rolls, key=lambda x: abs(x-roll))
            if abs(roll-closest) > np.deg2rad(10):
                rolls.append(roll)
                new_ps.append(ps[i, 0])
        rolls.pop(0)
        rolls = np.asarray(rolls)
        new_ps = np.asarray(new_ps)
        line_roll = np.sort(rolls)
        new_ps = new_ps[np.argsort(rolls)]
        lvls, line_roll, new_ps = self.update_z(line_roll, new_ps, rack)
        if len(lvls) == 0:
            return -1, 0, [], []
        sri = line_roll.argsort()[:]
        t1, t2 = np.meshgrid(sri, sri)
        angles_i = np.asarray([t2.flatten()[:], t1.flatten()[:]]).T.reshape(sri.shape[0], sri.shape[0], 2)
        angles_i = angles_i[np.triu_indices(sri.shape[0], 1)]

        angle_pairs = line_roll[angles_i]

        i_distances = lvls[angles_i]
        level_heights = self.rs.hbar_ticks[i_distances[:, 1]] - self.rs.hbar_ticks[i_distances[:, 0]]

        # # angle_pairs = np.take(line_roll, angles_i)
        # level_heights = i_distances * self.rs.level_height
        # print(np.rad2deg(angle_pairs), level_heights)
        if len(level_heights) != 0:
            distances, confidences = self.__solve_ASA(angle_pairs, level_heights)
            distance = np.average(distances, weights=confidences)
            confidence = 0
            if 0 < distance < self.rs.aisle_width:
                confidence = np.mean(confidences)
            return distance, confidence, self.rs.hbar_ticks[lvls], new_ps
        return -1, 0, [], []

    def __get_end2end_lines(self, ps, vp, cols):
        lines = []
        if len(ps) == 0:
            return []

        for p in ps:
            vx = vp[0] - p[0]
            vy = vp[1] - p[1]
            x = vp[0]
            y = vp[1]
            lefty = int((-x*vy/vx) + y)
            righty = int(((cols-x)*vy/vx)+y)
            lines.append([[cols-1, righty], [0, lefty]])
        return lines

    # def __get_end2end_lines(self, ps, indexes, vp, cols, rack):
    #     '''
    #     rack = -1    represents left rack in image
    #     rack = 1     represents right rack in image
    #     '''
    #     angle_thresh = 10
    #     ps = ps[indexes]
    #     ps = ps[:, 0]
    #
    #     lines = []
    #     angles = []
    #     for p in ps:
    #         vx = vp[0] - p[0]
    #         vy = vp[1] - p[1]
    #         x = vp[0]
    #         y = vp[1]
    #         lefty = int((-x * vy / vx) + y)
    #         righty = int(((cols - x) * vy / vx) + y)
    #         lines.append([[cols - 1, righty], [0, lefty]])
    #         angles.append(np.rad2deg(np.arctan2(righty - lefty, cols - 1)))
    #
    #     valid_indexes = []
    #     valid_lines = []
    #     valid_angles = [1000]
    #     for i, angle in enumerate(angles):
    #         closest = min(valid_angles, key=lambda x: abs(x - angle))
    #         if abs(angle - closest) > angle_thresh:
    #             valid_angles.append(angle)
    #             valid_indexes.append(indexes[i][0])
    #             valid_lines.append(lines[i])
    #     valid_angles.pop(0)
    #
    #     valid_angles = np.asarray(valid_angles)
    #     valid_indexes = np.asarray(valid_indexes)
    #     valid_lines = np.asarray(valid_lines)
    #
    #     sorted_angles = (rack * valid_angles).argsort()[:]
    #
    #     return valid_lines[sorted_angles], valid_indexes[sorted_angles]

    def update_z(self, line_roll, centers, rack):
        '''
        rack = -1    represents left rack in image
        rack = 1     represents right rack in image
        '''
        actual_bar_zs = self.get_actual_bars_heights(line_roll, rack)
        # print(actual_bar_zs)
        # print(self.rs.hbar_ticks)
        a = np.tile(actual_bar_zs, (self.rs.hbar_ticks.shape[0], 1)).T
        b = np.tile(self.rs.hbar_ticks, (actual_bar_zs.shape[0], 1))
        match_mat = np.absolute(a - b)
        np.set_printoptions(suppress=True, linewidth=np.nan)
        diff_index = match_mat.argmin(axis=1)
        least_err = np.inf
        final_bars_info = []
        diff = 0
        for i, bar_z in enumerate(actual_bar_zs):
            # print("matching ", bar_z, " with ", self.rs.hbar_ticks[diff_index[i]])
            error = bar_z - self.rs.hbar_ticks[diff_index[i]]
            bars_info = np.empty((0, 3), float)
            if abs(error) < (self.rs.level_height/2):
                new_bar_zs = self.rs.hbar_ticks[diff_index] + error
                # print("\t", self.rs.hbar_ticks + error)
                for j, new_bar_z in enumerate(new_bar_zs):
                    (nearest_index, nearest_value) = min(enumerate(actual_bar_zs), key=lambda x: abs(x[1]-new_bar_z))
                    # print(nearest_index, nearest_value)
                    err = abs(nearest_value - new_bar_z)
                    if err < self.rs.level_height/2:
                        if len(bars_info) == 0 or diff_index[j] not in bars_info[:, 0]:
                            bars_info = np.append(bars_info, np.array([[diff_index[j], nearest_index, err]]), axis=0)
                        else:
                            bar_index = np.argwhere(bars_info[:, 0] == diff_index[j])[0, 0]
                            if err <= bars_info[bar_index, 2]:
                                bars_info[bar_index, :] = np.array([diff_index[j], nearest_index, err])
                # print("totalerr:= ", total_err)
            else:
                bars_info = np.append(bars_info, [[0, 0, 100]], axis=0)
            # print("total_err:= ",sum(bars_info[:][2]))
            if sum(bars_info[:, 2]) < least_err:
                least_err = sum(bars_info[:, 2])
                # print("\terror==", least_err)
                diff = error
                final_bars_info = bars_info

        diff = np.clip(diff, -self.dp.delta_z, self.dp.delta_z)

        if (self.dp.drone_facing != self.dp.AISLE) and (self.dp.gimbal_direction * rack) == -1:
            self.dp.z -= diff
        elif (self.dp.drone_facing == self.dp.AISLE) and rack == self.dp.RIGHT_RACK:
            self.dp.z -= diff
        else:
            pass
            # todo: add dominant_rack attribute
            # self.dp.z -= diff
            # self.dp.print_state(rack)
            # print("wrong condition - 1")

        # print("z:= ", self.dp.z, "\terror:= ", diff)
        inds, nearest_indexes = [], []
        if len(final_bars_info) > 0:
            inds, nearest_indexes, _ = map(list, zip(*final_bars_info))
        inds = np.array(inds, dtype=int)
        nearest_indexes = np.array(nearest_indexes, dtype=int)
        line_roll = np.array(line_roll)
        line_roll = line_roll[nearest_indexes]
        centers = centers[nearest_indexes]
        # print(inds, np.rad2deg(line_roll))
        return inds, line_roll, centers

    def get_actual_bars_heights(self, line_roll, rack):
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
            print("wrong condition! - 2")

        dists_from_drone = dist_from_rack / np.tan(line_roll)
        bar_heights = self.dp.z - dists_from_drone
        return np.array(bar_heights)

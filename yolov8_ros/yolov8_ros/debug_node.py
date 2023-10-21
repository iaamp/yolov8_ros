# Copyright (C) 2023  Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import cv2
import random
import numpy as np
from typing import Tuple

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data

import message_filters
from cv_bridge import CvBridge
from ultralytics.utils.plotting import Annotator, colors

from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from yolov8_msgs.msg import BoundingBox2D
from yolov8_msgs.msg import KeyPoint2D
from yolov8_msgs.msg import KeyPoint3D
from yolov8_msgs.msg import Detection
from yolov8_msgs.msg import DetectionArray


class DebugNode(Node):

    def __init__(self) -> None:
        super().__init__("debug_node")

        self._class_to_color = {}
        self.cv_bridge = CvBridge()

        # pubs
        self._dbg_pub = self.create_publisher(Image, "dbg_image", 10)
        self._bb_markers_pub = self.create_publisher(
            MarkerArray, "dgb_bb_markers", 10)
        self._kp_markers_pub = self.create_publisher(
            MarkerArray, "dgb_kp_markers", 10)

        # subs
        image_sub = message_filters.Subscriber(
            self, Image, "image_raw", qos_profile=qos_profile_sensor_data)
        detections_sub = message_filters.Subscriber(
            # self, DetectionArray, "detections", qos_profile=10)
            self, DetectionArray, "tracking", qos_profile=10)

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (image_sub, detections_sub), 10, 0.5)
        self._synchronizer.registerCallback(self.detections_cb)

        self.tracking_blacklist= [6, 10, 14, 18, 25, 41, 44, 45, 48, 49, 50, 46, 47, 68, 73, 75, 81, 85, 89, 92, 95, 100, 110, 109, 115, 116, 130, 131, 135, 159, 238, 242, 243, 250, 251, 252, 255, 258, 273, 280, 285, 284, 334, 335, 354, 373, 374, 375, 377, 381, 394, 402, 405, 411, 442, 449, 452, 444, 455, 456, 467, 475, 469, 468, 465, 476, 477, 475, 488, 486, 489, 490, 493, 495, 500, 503, 506, 508, 509, 510, 514, 516, 515, 520, 521, 528, 526, 528, 535, 533, 544, 548, 551, 559, 571, 572, 580, 581, 574, 584, 588, 592, 595, 599, 605, 607, 609, 610, 614, 625, 629, 634, 636, 631, 645, 655, 658, 660, 663, 664, 668, 670, 676, 682, 680, 683, 685, 690, 691, 710, 721, 723, 728, 732, 740, 741, 743, 745, 748, 751, 753, 754, 756, 761, 764, 766, 769, 775, 779, 780, 781, 788, 787, 789, 796, 798, 815, 817, 837, 838, 828, 839, 841, 844, 846, 858, 868, 864, 870, 876, 887, 888, 892, 894, 907, 910, 913, 910, 920, 916, 922, 923, 938, 942, 943, 945, 946, 947, 950, 952, 953, 965, 971, 979, 977, 983, 985, 1004, 1022, 1040, 1041, 1039, 1043, 1059, 1061, 1063, 1064, 1065, 1078, 1087, 1072, 1127, 1154, 1159, 1161, 1152, 1165, 1169, 1170, 1171, 1175, 1180, 1182, 1178, 1303, 1309, 1311, 1312, 1317, 1320, 1322, 1323, 1327, 1332, 1335, 1338, 1339, 1343, 1347, 1348, 1353, 1358, 1360, 1362, 1364, 1365, 1370, 1371, 1417, 1422, 1424, 1438, 1440, 1454, 1456, 1459, 1460, 1463, 1473, 1472, 1474, 1480, 1482, 1491, 1501, 1513, 1537, 1578, 1577, 1583, 1593, 1588, 1592, 1602, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1624, 1623, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640, 1641, 1642, 1643, 1644, 1645, 1646, 1647, 1648, 1649, 1650, 1651, 1652, 1653, 1654, 1655, 1656, 1657, 1658, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1673, 1674, 1675, 1676, 1677, 1678, 1679, 1680, 1691, 1697, 1698, 1701, 1706, 1710, 1717, 1722, 1724, 1725, 1729, 1730, 1736, 1741, 1742, 1748, 1752, 1796, 1797, 1806, 1817, 1820, 1842, 1865, 1875, 1897, 1896, 1898, 1926, 1929, 1934, 1936, 1940, 1942, 1944, 1947, 1949, 1950, 1952, 1960, 1964, 1965, 1967, 1973, 1975, 1977, 1979, 1980, 1981, 1985, 1992, 1996, 1998, 2002, 2013, 2036, 2051, 2050, 2090, 2085, 2086, 2100, 2103, 2105, 2119, 2121, 2127, 2135, 2138, 2141, 2144, 2145, 2146, 2150, 2152, 2151, 2155, 2157, 2161, 2162, 2166, 2175, 2176, 2182, 2184, 2185, 2188, 2194, 2197, 2196, 2199, 2202, 2207, 2209, 2210, 2211, 2214, 2215, 2218, 2221, 2222, 2223, 2224, 2225, 2228, 2229, 2231, 2232, 2236, 2237, 2238, 2244, 2254, 2255, 2258, 2266, 2269, 2274, 2281, 2290, 2294, 2296, 2300, 2299, 2305, 2307, 2310, 2311, 2312, 2313, 2314, 2315, 2316, 2317, 2318, 2319, 2320, 2321, 2322, 2323, 2324, 2325, 2326, 2327, 2328, 2329, 2330, 2332, 2335, 2336, 2337, 2342, 2347, 2359, 2358, 2357, 2356, 2359, 1357, 1369, 1208, 1211, 1218, 1257, 1258, 1235, 1271, 1280, 1287, 1282, 1436, 1438, 1440, 1442, 1462, 1479, 1496, 1597, 1740, 1799, 1852, 1850, 1862, 1886, 1889, 1892, 1912, 2078, 42, 53, 54, 58, 60, 83, 88, 99, 117, 122, 254, 410, 421, 435, 436, 447, 448, 446, 450, 451, 492, 497, 499, 498, 502, 518, 524, 547, 565, 578, 594, 601, 635, 638, 647, 648, 654, 659, 719, 747, 783, 835, 845, 908, 905, 926, 931, 951, 958, 970, 976, 1053, 1083, 1146, 1153, 1199, 1183, 1224, 1229, 1313, 1325, 1329, 1466, 1494, 1484, 1487, 1486, 1899, 2003, 2046, 2077, 2089, 1848, 2186, 2233, 2295, 2361, 2367, 2345, 2352, 2353, 2361, 2371, 2372, 2375, 1003, 1009, 1012, 1011, 1015, 1016, 1017, 1117, 1262, 1263, 1264, 1265, 1267, 1269, 1272, 1366, 1367, 1368, 1591, 1600, 1683]

        self.tracking_blacklist.sort()


    def draw_box(self, cv_image: np.array, detection: Detection, color: Tuple[int]) -> np.array:

        # get detection info
        label = detection.class_name
        score = detection.score
        box_msg: BoundingBox2D = detection.bbox
        track_id = detection.id

        min_pt = (round(box_msg.center.position.x - box_msg.size.x / 2.0),
                  round(box_msg.center.position.y - box_msg.size.y / 2.0))
        max_pt = (round(box_msg.center.position.x + box_msg.size.x / 2.0),
                  round(box_msg.center.position.y + box_msg.size.y / 2.0))

        # draw box
        cv2.rectangle(cv_image, min_pt, max_pt, color, 2)

        # write text
        # label = "{} ({}) ({:.3f})".format(label, str(track_id), score)
        label = "{} ({:.3f})".format(str(track_id), score)
        pos = (min_pt[0] + 5, min_pt[1] + 25)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (155, 250, 157)
        cv2.putText(cv_image, label, pos, font,
                    # 1, color, 1, cv2.LINE_AA)
                    1, text_color, 1, cv2.LINE_AA)

        return cv_image

    def draw_mask(self, cv_image: np.array, detection: Detection, color: Tuple[int]) -> np.array:

        mask_msg = detection.mask
        mask_array = np.array([[int(ele.x), int(ele.y)]
                              for ele in mask_msg.data])

        if mask_msg.data:
            layer = cv_image.copy()
            layer = cv2.fillPoly(layer, pts=[mask_array], color=color)
            cv2.addWeighted(cv_image, 0.4, layer, 0.6, 0, cv_image)
            cv_image = cv2.polylines(cv_image, [mask_array], isClosed=True,
                                     color=color, thickness=2, lineType=cv2.LINE_AA)

            # write text
            label = detection.class_name
            score = detection.score
            label = "{} ({:.2f})".format(label, score)
            x_min, y_min = np.min(mask_array, axis=0)
            pos = (x_min + 5, y_min - 10)  # Adjust the position as needed
            font = cv2.FONT_HERSHEY_SIMPLEX
            # text_color = (155, 250, 157)
            text_color = (220, 220, 220) # offwhite
            # text_color = (167, 201, 52) # light blue-green
            # text_color = (245, 164, 2) # light blue
            cv2.putText(cv_image, label, pos, font,
                        1, text_color, 1, cv2.LINE_AA)
                        # 1, color, 1, cv2.LINE_AA)

        return cv_image

    def draw_keypoints(self, cv_image: np.array, detection: Detection) -> np.array:

        keypoints_msg = detection.keypoints

        ann = Annotator(cv_image)

        kp: KeyPoint2D
        for kp in keypoints_msg.data:
            color_k = [int(x) for x in ann.kpt_color[kp.id - 1]
                       ] if len(keypoints_msg.data) == 17 else colors(kp.id - 1)

            cv2.circle(cv_image, (int(kp.point.x), int(kp.point.y)),
                       5, color_k, -1, lineType=cv2.LINE_AA)

        def get_pk_pose(kp_id: int) -> Tuple[int]:
            for kp in keypoints_msg.data:
                if kp.id == kp_id:
                    return (int(kp.point.x), int(kp.point.y))
            return None

        for i, sk in enumerate(ann.skeleton):
            kp1_pos = get_pk_pose(sk[0])
            kp2_pos = get_pk_pose(sk[1])

            if kp1_pos is not None and kp2_pos is not None:
                cv2.line(cv_image, kp1_pos, kp2_pos, [
                    int(x) for x in ann.limb_color[i]], thickness=2, lineType=cv2.LINE_AA)

        return cv_image

    def create_bb_marker(self, detection: Detection) -> Marker:

        bbox3d = detection.bbox3d

        marker = Marker()
        marker.header.frame_id = bbox3d.frame_id

        marker.ns = "yolov8_3d"
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.frame_locked = False

        marker.pose.position.x = bbox3d.center.position.x
        marker.pose.position.y = bbox3d.center.position.y
        marker.pose.position.z = bbox3d.center.position.z

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = bbox3d.size.x
        marker.scale.y = bbox3d.size.y
        marker.scale.z = bbox3d.size.z

        marker.color.b = 0.0
        marker.color.g = detection.score * 255.0
        marker.color.r = (1.0 - detection.score) * 255.0
        marker.color.a = 0.4

        marker.lifetime = Duration(seconds=0.5).to_msg()
        marker.text = detection.class_name

        return marker

    def create_kp_marker(self, keypoint: KeyPoint3D) -> Marker:

        marker = Marker()

        marker.ns = "yolov8_3d"
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.frame_locked = False

        marker.pose.position.x = keypoint.point.x
        marker.pose.position.y = keypoint.point.y
        marker.pose.position.z = keypoint.point.z

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        marker.color.b = keypoint.score * 255.0
        marker.color.g = 0.0
        marker.color.r = (1.0 - keypoint.score) * 255.0
        marker.color.a = 0.4

        marker.lifetime = Duration(seconds=0.5).to_msg()
        marker.text = str(keypoint.id)

        return marker

    def detections_cb(self, img_msg: Image, detection_msg: DetectionArray) -> None:

        self.get_logger().info("debug node: detections_cb")

        cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg)
        # cv_image = self.cv_bridge.imgmsg_to_cv2(img_msg, "bgr8")

        bb_marker_array = MarkerArray()
        kp_marker_array = MarkerArray()

        detection: Detection
        for detection in detection_msg.detections:

            # self.get_logger().info(f"detection.id: {detection.id}")
            if int(detection.id) in self.tracking_blacklist:
                # self.get_logger().info(f"skipping id: {detection.id}")
                continue

            # random color
            label = detection.class_name

            if label not in self._class_to_color:
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                self._class_to_color[label] = (r, g, b)

            color = self._class_to_color[label]
            # color = (115, 13, 91)
            color = (25, 17, 184)
            color_text = (252, 140, 20)

            cv_image = self.draw_mask(cv_image, detection, color)
            # cv_image = self.draw_box(cv_image, detection, color)
            # cv_image = self.draw_keypoints(cv_image, detection)

            if detection.bbox3d.frame_id:
                marker = self.create_bb_marker(detection)
                marker.header.stamp = img_msg.header.stamp
                marker.id = len(bb_marker_array.markers)
                bb_marker_array.markers.append(marker)

            if detection.keypoints3d.frame_id:
                for kp in detection.keypoints3d.data:
                    marker = self.create_kp_marker(kp)
                    marker.header.frame_id = detection.keypoints3d.frame_id
                    marker.header.stamp = img_msg.header.stamp
                    marker.id = len(kp_marker_array.markers)
                    kp_marker_array.markers.append(marker)

        # publish dbg image
        self._dbg_pub.publish(self.cv_bridge.cv2_to_imgmsg(cv_image,
                                                           encoding=img_msg.encoding))
        # self.get_logger().info("debug img published")

        self._bb_markers_pub.publish(bb_marker_array)
        self._kp_markers_pub.publish(kp_marker_array)

def main():
    rclpy.init()
    node = DebugNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

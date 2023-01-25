from typing import List, Tuple, TypeVar, Optional
from shapely.geometry import Polygon, LineString, Point

import utils


class PolygonInfo:
    TYPE_FREE = "free"
    TYPE_EDGE_ON_BOUNDARY = "edge"
    TYPE_VERTEX_ON_BOUNDARY = "vertex"

    def __init__(self, poly_id: int, polygon: Polygon, poly_type: str = TYPE_FREE):
        self.poly_id = poly_id
        self.polygon = polygon
        self.poly_type = poly_type


class ArcPointInfo:
    TYPE_CRITICAL = "critical"
    TYPE_BOUNDARY = "boundary"

    def __init__(self, point_type: str, point: Point, polygon_id_connect: int):
        self.point_type = point_type
        self.point = point
        self.polygon_id_connect = polygon_id_connect


class ArcInfo:
    def __init__(self, radius: float, center: Point, left_border: ArcPointInfo, critical: ArcPointInfo,
                 right_border: ArcPointInfo):
        self.radius = radius
        self.center = center
        self.left_border = left_border
        self.critical = critical
        self.right_border = right_border
        self.arc_id = -1

    def set_id(self, new_id: int):
        self.arc_id = new_id


class ArcPartInfo:
    ARC_PART_LEFT = "left"
    ARC_PART_RIGHT = "right"
    ARC_PART_FULL = "full"
    ARC_PART_POINT = "point"

    def __init__(self, arc: ArcInfo, part_type: str):
        self.arc: ArcInfo = arc
        self.part_type = part_type

    def is_part_a_point(self):
        if self.part_type == ArcPartInfo.ARC_PART_LEFT:
            return self.arc.left_border.point == self.arc.critical.point
        if self.part_type == ArcPartInfo.ARC_PART_RIGHT:
            return self.arc.critical.point == self.arc.right_border.point

    # Для FULL и POINT вернет critical
    # Для LEFT и RIGHT вернет середину дуги
    def get_middle_of_part(self) -> Point:
        # if self.part_type in (ArcPartInfo.ARC_PART_POINT, ArcPartInfo.ARC_PART_FULL):
        #     return self.arc.critical.point

        if self.part_type == ArcPartInfo.ARC_PART_POINT:
            return self.arc.critical.point

        left_phi = 0.0
        right_phi = 0.0

        if self.part_type == ArcPartInfo.ARC_PART_FULL:
            _, left_phi = utils.cartesian_to_polar(self.arc.left_border.point, self.arc.center)
            _, right_phi = utils.cartesian_to_polar(self.arc.right_border.point, self.arc.center)

        if self.part_type == ArcPartInfo.ARC_PART_LEFT:
            _, left_phi = utils.cartesian_to_polar(self.arc.left_border.point, self.arc.center)
            _, right_phi = utils.cartesian_to_polar(self.arc.critical.point, self.arc.center)

        if self.part_type == ArcPartInfo.ARC_PART_RIGHT:
            _, left_phi = utils.cartesian_to_polar(self.arc.critical.point, self.arc.center)
            _, right_phi = utils.cartesian_to_polar(self.arc.right_border.point, self.arc.center)

        middle_phi = left_phi + (right_phi - left_phi) / 2
        middle_point = utils.polar_to_cartesian((self.arc.radius, middle_phi), self.arc.center)

        return Point(middle_point)


class AreaInfo:
    def __init__(self, area_id: int, input_arc: ArcPartInfo, output_arc: ArcPartInfo):
        self.area_id = area_id
        self.input_arc: ArcPartInfo = input_arc
        self.output_arc: ArcPartInfo = output_arc

    def get_center_point(self):
        input_center_point = self.input_arc.get_middle_of_part()
        output_center_point = self.output_arc.get_middle_of_part()

        return utils.get_middle_between_points(input_center_point, output_center_point)

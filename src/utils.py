import numpy as np
from typing import List, Tuple, TypeVar, Optional
import math

import shapely.ops
from shapely.geometry import Polygon, LineString, Point


def dist(p1, p2):
    dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
    return dist


def get_points_from_polygon(polygon: Polygon) -> List[Point]:
    return [Point(raw_point) for raw_point in get_raw_points_from_polygon(polygon)]


def get_raw_points_from_polygon(polygon: Polygon) -> List[Tuple[float, float]]:
    return list(zip(*polygon.exterior.xy))[:-1]


def get_polygon_lines(polygon: Polygon) -> List[LineString]:
    bound_coords = polygon.boundary.coords
    return [LineString(bound_coords[k:k+2]) for k in range(len(bound_coords) - 1)]


def get_circle_radius(center_point: Point, circle_point: Point):
    return math.sqrt( math.pow(circle_point.x - center_point.x, 2) + math.pow(circle_point.y - center_point.y, 2))


def get_line_circle_intersections(line: LineString, circle_center: Point, circle_radius: float) -> List[Point]:
    point1, point2 = line.coords[0], line.coords[1]

    res_point1 = None
    res_point2 = None

    # отдельная обработка вертикальной прямой, для которой нельзя посчитать тангенс
    if point2[0] - point1[0] == 0:
        line_x_0 = point1[0]

        b = -2 * circle_center.y

        discriminant = 4 * (circle_radius ** 2 - line_x_0 ** 2 + 2 * line_x_0 * circle_center.x - circle_center.x ** 2)
        if discriminant >= 0:
            if discriminant < 1e-10:
                y1 = -b / 2
                res_point1 = Point([line_x_0, y1])
            else:
                y1 = (-b - math.sqrt(discriminant)) / 2
                y2 = (-b + math.sqrt(discriminant)) / 2
                res_point1 = Point([line_x_0, y1])
                res_point2 = Point([line_x_0, y2])
    else:

        k_line = (point2[1] - point1[1]) / (point2[0] - point1[0])
        b_line = -k_line * point1[0] + point1[1]

        # Решение квадратного уравнения
        a = 1 + k_line ** 2
        b = -2 * circle_center.x + 2 * k_line * b_line - 2 * k_line * circle_center.y
        c = -circle_radius ** 2 + (b_line - circle_center.y) ** 2 + circle_center.x ** 2
        discriminant = b ** 2 - 4 * a * c
        if discriminant >= 0:
            if discriminant < 1e-10:
                x1 = -b / (2 * a)
                res_point1 = Point([x1, k_line * x1 + b_line])
            else:
                x1 = (-b - math.sqrt(discriminant)) / (2 * a)
                x2 = (-b + math.sqrt(discriminant)) / (2 * a)
                res_point1 = Point([x1, k_line * x1 + b_line])
                res_point2 = Point([x2, k_line * x2 + b_line])

    result = []

    if res_point1 is not None and line.distance(res_point1) < 1e-8:
        result.append(res_point1)
    if res_point2 is not None and line.distance(res_point2) < 1e-8:
        result.append(res_point2)

    return result


def cartesian_to_polar(point: Point, center: Point=Point([0, 0])) -> Tuple[float, float]:
    radius = math.sqrt((point.x - center.x) ** 2 + (point.y - center.y) ** 2)
    phi = math.atan2(point.y - center.y, point.x - center.x)

    return radius, phi


def polar_to_cartesian(point: Tuple[float, float], center: Point) -> Point:
    radius = point[0]
    phi = point[1]

    return Point([radius * math.cos(phi) + center.x, radius * math.sin(phi) + center.y])


def get_phi_for_arc_length(radius: float, length: float) -> float:
    return length / (math.pi * radius)


def get_arc_linestring(center: Point, radius: float, phi_bounds: Tuple[float, float]) -> LineString:
    SEGMENTS = 100

    theta = np.linspace(phi_bounds[0], phi_bounds[-1], SEGMENTS)
    x = center.x + radius * np.cos(theta)
    y = center.y + radius * np.sin(theta)

    return LineString(np.column_stack([x, y]))


def get_middle_between_points(first: Point, second: Point) -> Point:
    half_x = (first.x + second.x) / 2
    half_y = (first.y + second.y) / 2
    return Point([half_x, half_y])

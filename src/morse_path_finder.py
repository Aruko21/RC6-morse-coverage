import json

import numpy as np
from typing import List, Tuple, TypeVar, Optional, Dict
import math
import copy
import sys

import shapely.ops
from shapely.geometry import Polygon, LineString, Point

import networkx as nx

import utils
from models import *

DATA_TYPE_INFO = "info"
DATA_TYPE_START_POINT = "startPoint"
DATA_TYPE_END_POINT = "endPoint"
DATA_TYPE_POLYGON = "polygon"


# Motion field obstacles parser
def parse_field_from_file(file_name):
    obstacles = []
    start_point = []
    end_point = []

    with open(file_name, "r") as read_file:
        data = json.load(read_file)
        if type(data) is not list:
            raise AttributeError("Incorrect data format. Should be array")

        for data_entry in data:
            if data_entry["type"] == DATA_TYPE_START_POINT:
                start_point = [data_entry['x'], data_entry['y']]
            elif data_entry["type"] == DATA_TYPE_END_POINT:
                end_point = [data_entry['x'], data_entry['y']]
            elif data_entry["type"] == DATA_TYPE_POLYGON:
                if len(data_entry['points']) != 0:
                    obstacles.append([(point['x'], point['y']) for point in data_entry['points']])
            else:
                continue

    return start_point, end_point, obstacles


class MorseCoverage:
    ARC_RADIUS_EPS = 1e-1
    GAP_ARC_LENGTH_TOLERANCE = 10
    GAP_ARC_CRITICAL_OFFSET = 1e-2
    COMPARISON_EPS = 1
    BORDER_POLY_VERTEX_EPS = 2
    BORDER_POLYGON_ID = 0

    def __init__(self, obstacles: List[List[Tuple[float, float]]], start_point: Tuple[float, float],
                 end_point: Tuple[float, float], x_boundary: Tuple[float, float], y_boundary: Tuple[float, float]):
        polygons = MorseCoverage.get_polygons_from_coords(obstacles, unite=True)
        # Сопоставляем каждому препятствию (полигону) свой id. У границ поля id = BORDER_POLYGON_ID = 0
        self.polygons: List[PolygonInfo] = [PolygonInfo(poly_id=item[0], polygon=item[1]) for item in zip(range(1, len(polygons) + 1), polygons)]

        self.start_point = Point(start_point)
        self.end_point = Point(end_point)
        self.x_boundary = sorted(x_boundary)
        self.y_boundary = sorted(y_boundary)
        # границы поля
        self.field_boundaries = [
            LineString([(self.x_boundary[0], self.y_boundary[0]), (self.x_boundary[0], self.y_boundary[1])]),
            LineString([(self.x_boundary[0], self.y_boundary[1]), (self.x_boundary[1], self.y_boundary[1])]),
            LineString([(self.x_boundary[1], self.y_boundary[1]), (self.x_boundary[1], self.y_boundary[0])]),
            LineString([(self.x_boundary[1], self.y_boundary[0]), (self.x_boundary[0], self.y_boundary[0])])
        ]

        self.morse_arcs: List[ArcInfo] = []
        self.areas: List[AreaInfo] = []
        self.coverage_graph: Optional[nx.Graph] = None

        self.__mark_polygons_on_boundary()
        # self.__delete_polygons_on_boundary()

    @staticmethod
    def get_polygons_from_coords(obstacles: List[List[Tuple[float, float]]], unite: bool = True) -> List[Polygon]:
        polygons = [Polygon(obstacle) for obstacle in obstacles]

        if unite:
            # NOTE: Loop is not optimized
            no_intersections = False
            while not no_intersections:
                # Pair-wise comparison between all polygons
                intersection_found = False
                for i in range(0, len(polygons)):
                    for j in range(i + 1, len(polygons)):
                        # If there is intersection — unite these polygons and go through the loop again
                        if polygons[i].overlaps(polygons[j]):
                            union = polygons[i].union(polygons[j])
                            polygons[i] = union
                            del polygons[j]
                            intersection_found = True
                            break
                    if intersection_found:
                        break

                if not intersection_found:
                    no_intersections = True

        return polygons

    # Пометить все препятствия на границе как границу (идентефикатор = 0)
    def __mark_polygons_on_boundary(self):
        for polygon_info in self.polygons:
            poly_points = utils.get_points_from_polygon(polygon_info.polygon)

            boundary_points = 0
            for point in poly_points:
                if self.is_point_on_boundary(point):
                    boundary_points += 1

            if boundary_points == 1:
                polygon_info.poly_type = PolygonInfo.TYPE_VERTEX_ON_BOUNDARY
            elif boundary_points >= 2:
                polygon_info.poly_type = PolygonInfo.TYPE_EDGE_ON_BOUNDARY

    # Если все совсем плохо — для упрощения можно удалить препятствия, соприкасающиеся с границей поля
    def __delete_polygons_on_boundary(self):
        self.__mark_polygons_on_boundary()

        for i in range(len(self.polygons) - 1, -1, -1):
            if self.polygons[i].poly_id == MorseCoverage.BORDER_POLYGON_ID:
                del self.polygons[i]

    # Проверка принадлежит ли точка границе с определенной погрешностью (или выходит за нее)
    def is_point_on_boundary(self, point: Point) -> bool:
        return point.x - MorseCoverage.COMPARISON_EPS <= self.x_boundary[0] or \
               point.x + MorseCoverage.COMPARISON_EPS >= self.x_boundary[1] or \
               point.y - MorseCoverage.COMPARISON_EPS <= self.y_boundary[0] or \
               point.y + MorseCoverage.COMPARISON_EPS >= self.y_boundary[1]

    @staticmethod
    def are_points_equal(point1: Point, point2: Point, eps: Optional[float] = COMPARISON_EPS) -> bool:
        return abs(point1.x - point2.x) < eps and abs(point1.y - point2.y) < eps

    @property
    def polygons_points_raw(self) -> List[Tuple[int, List[Tuple[float, float]]]]:
        return list(map(lambda poly_info: (poly_info.poly_id, utils.get_raw_points_from_polygon(poly_info.polygon)), self.polygons))

    @property
    def polygons_points(self) -> List[Tuple[int, List[Point]]]:
        return list(map(lambda poly_info: (poly_info.poly_id, utils.get_points_from_polygon(poly_info.polygon)), self.polygons))

    def get_arcs(self, clear: Optional[bool] = False) -> List[ArcInfo]:
        if len(self.morse_arcs) > 0:
            if not clear:
                return self.morse_arcs
            else:
                self.morse_arcs.clear()

        points_per_polygon = self.polygons_points

        # Проход по всем полигонам с целью определения критических точек и построения дуг через них
        for i in range(len(self.polygons)):
            polygon_id = self.polygons[i].poly_id
            polygon = self.polygons[i].polygon
            polygon_type = self.polygons[i].poly_type
            polygon_points = points_per_polygon[i][1]

            bounds = utils.get_polygon_lines(polygon)
            # точки препятствия, отсортированные по их удалению от начальной точки
            # (или по радиусу в полярной системе координат, где центр — начальная точка)
            polygon_points = sorted(polygon_points, key=lambda point: utils.get_circle_radius(self.start_point, point))
            # сами радиусы точек, также отсортированные
            polygon_radii = sorted(list(map(lambda point: utils.get_circle_radius(self.start_point, point), polygon_points)))

            # Первая критическая точка — максимально удаленная от центра
            # (здесь и далее центр — это стартовая точка и центр окружности, которая выделяет область)
            max_radius = polygon_radii[-1]
            max_point = polygon_points[-1]
            max_point_arc = ArcPointInfo(point_type=ArcPointInfo.TYPE_CRITICAL, point=max_point, polygon_id_connect=polygon_id)

            # Вторая критическая точка — минимально удаленная от центра — тут может возникнуть развилка разбиения
            min_radius = polygon_radii[0]
            min_point = polygon_points[0]

            # При этом так как фронт в виде окружности — может быть такое, что окружность упрется в грань, а не в вершину
            # Надо это проверить и в случае чего уточнить минимальную критическую точку как точку касания к грани
            for bound in bounds:
                intersection = utils.get_line_circle_intersections(line=bound, circle_center=self.start_point, circle_radius=min_radius)
                if len(intersection) == 2:
                    half_x = (intersection[0].x + intersection[1].x) / 2
                    half_y = (intersection[0].y + intersection[1].y) / 2
                    min_point = Point([half_x, half_y])
                    min_radius = utils.get_circle_radius(self.start_point, min_point)

            min_point_arc = ArcPointInfo(point_type=ArcPointInfo.TYPE_CRITICAL, point=min_point, polygon_id_connect=polygon_id)

            max_arc_boundaries = self.get_closest_poly_intersections(max_point)
            min_arc_boundaries = self.get_closest_poly_intersections(min_point)

            # Если минимальная и/или максимальная критическая точка совпадает с границей (полигон лежит на ней)
            # то эту точку можно не рассматривать по той причине, что связность прямой в декомпозиции Морса не нарушается.
            border_crit_point = None
            if not polygon_type == PolygonInfo.TYPE_EDGE_ON_BOUNDARY:
                # Однако, если полигон касается границы только в одной точке, то это гарантированно критическая точка
                if polygon_type == PolygonInfo.TYPE_VERTEX_ON_BOUNDARY:
                    crit_radius = 0

                    if self.is_point_on_boundary(min_point):
                        crit_radius = min_radius
                        border_crit_point = min_point_arc
                    elif self.is_point_on_boundary(max_point):
                        crit_radius = max_radius
                        border_crit_point = max_point_arc

                    if border_crit_point is not None:
                        if self.is_border_on_left(border_crit_point.point):
                            left_border = ArcPointInfo(point_type=ArcPointInfo.TYPE_BOUNDARY, point=border_crit_point.point,
                                                       polygon_id_connect=MorseCoverage.BORDER_POLYGON_ID)
                            right_border = border_crit_point
                        else:
                            right_border = ArcPointInfo(point_type=ArcPointInfo.TYPE_BOUNDARY, point=border_crit_point.point,
                                                        polygon_id_connect=MorseCoverage.BORDER_POLYGON_ID)
                            left_border = border_crit_point

                        self.morse_arcs.append(ArcInfo(radius=crit_radius, center=self.start_point,
                                                       left_border=left_border, critical=border_crit_point,
                                                       right_border=right_border))

            if polygon_id == 9:
                print("ss")

            if min_point_arc != border_crit_point and not self.is_point_on_boundary(min_point):
                min_point_redundant = False

                if min_arc_boundaries[0].polygon_id_connect == polygon_id \
                        or min_arc_boundaries[1].polygon_id_connect == polygon_id:
                    min_point_redundant = True

                if not min_point_redundant:
                    for point in polygon_points:
                        if self.are_points_equal(point, min_arc_boundaries[0].point, eps=MorseCoverage.BORDER_POLY_VERTEX_EPS) \
                                or self.are_points_equal(point, min_arc_boundaries[1].point, eps=MorseCoverage.BORDER_POLY_VERTEX_EPS):
                            min_point_redundant = True
                            break

                if not min_point_redundant:
                    self.morse_arcs.append(ArcInfo(radius=min_radius, center=self.start_point,
                                                   left_border=min_arc_boundaries[0], critical=min_point_arc,
                                                   right_border=min_arc_boundaries[1]))

            if max_point_arc != border_crit_point and not self.is_point_on_boundary(max_point):
                max_point_redundant = False

                if max_arc_boundaries[0].polygon_id_connect == polygon_id \
                        or max_arc_boundaries[1].polygon_id_connect == polygon_id:
                    max_point_redundant = True

                if not max_point_redundant:
                    for point in polygon_points:
                        if self.are_points_equal(point, max_arc_boundaries[0].point, eps=MorseCoverage.BORDER_POLY_VERTEX_EPS) \
                                or self.are_points_equal(point, max_arc_boundaries[1].point, eps=MorseCoverage.BORDER_POLY_VERTEX_EPS):
                            max_point_redundant = True
                            break

                if not max_point_redundant:
                    self.morse_arcs.append(ArcInfo(radius=max_radius, center=self.start_point,
                                                   left_border=max_arc_boundaries[0], critical=max_point_arc,
                                                   right_border=max_arc_boundaries[1]))

            # Далее необходимо рассмотреть случаи выпуклых внутрь препятствий (или случаев, когда они лежат на границах)
            # и также образуют выпуклость внутрь. Такие выпуклости (впуклости с: ) образуют развилки
            for k in range(1, len(polygon_radii) - 1):
                mid_point = polygon_points[k]
                mid_point_arc = ArcPointInfo(point_type=ArcPointInfo.TYPE_CRITICAL, point=mid_point,
                                             polygon_id_connect=polygon_id)
                mid_point_radius = polygon_radii[k]

                if polygon_type == PolygonInfo.TYPE_VERTEX_ON_BOUNDARY and self.is_point_on_boundary(mid_point):
                    if self.is_border_on_left(mid_point):
                        left_border = ArcPointInfo(point_type=ArcPointInfo.TYPE_BOUNDARY, point=mid_point,
                                                   polygon_id_connect=MorseCoverage.BORDER_POLYGON_ID)
                        right_border = mid_point_arc
                    else:
                        right_border = ArcPointInfo(point_type=ArcPointInfo.TYPE_BOUNDARY, point=mid_point,
                                                    polygon_id_connect=MorseCoverage.BORDER_POLYGON_ID)
                        left_border = mid_point_arc

                    # Нужно добавить 2 точки на вход и на выход
                    self.morse_arcs.append(ArcInfo(radius=mid_point_radius, center=self.start_point,
                                                   left_border=left_border, critical=mid_point_arc,
                                                   right_border=right_border))
                    self.morse_arcs.append(ArcInfo(radius=mid_point_radius, center=self.start_point,
                                                   left_border=left_border, critical=mid_point_arc,
                                                   right_border=right_border))
                    continue

                # if self.is_point_on_boundary(mid_point):
                #     mid_point_arc = ArcPointInfo(point_type=ArcPointInfo.TYPE_ON_EDGE, point=mid_point,
                #                                  polygon_id_connect=polygon_id)
                #
                #     mid_arc_boundaries = self.get_closest_poly_intersections(mid_point)
                #     if MorseCoverage.are_points_equal(mid_arc_boundaries[0].point, mid_point):
                #         left_border = mid_arc_boundaries[0]
                #         right_border = mid_point_arc
                #     else:
                #         left_border = mid_point_arc
                #         right_border = mid_arc_boundaries[1]
                #     arcs.append(ArcInfo(radius=mid_point_radius, center=self.start_point,
                #                         left_border=left_border, critical=mid_point_arc,
                #                         right_border=right_border))
                #     continue

                # Строим маленькие дуги влево и вправо и смотрим, пересекают ли они полигон
                tmp_critical_polar = utils.cartesian_to_polar(mid_point, self.start_point)
                # Для длины дуги рассчитываем ее угол, чтобы ее построить
                tmp_arc_phi_tolerance = utils.get_phi_for_arc_length(mid_point_radius, MorseCoverage.GAP_ARC_LENGTH_TOLERANCE) / 2
                tmp_arc_left = utils.get_arc_linestring(
                    self.start_point, mid_point_radius,
                    (tmp_critical_polar[1] - tmp_arc_phi_tolerance, tmp_critical_polar[1] - MorseCoverage.GAP_ARC_CRITICAL_OFFSET)
                )
                tmp_arc_right = utils.get_arc_linestring(
                    self.start_point, mid_point_radius,
                    (tmp_critical_polar[1] + MorseCoverage.GAP_ARC_CRITICAL_OFFSET, tmp_critical_polar[1] + tmp_arc_phi_tolerance)
                )

                left_intersects = polygon.intersects(tmp_arc_left)
                right_intersects = polygon.intersects(tmp_arc_right)

                # TODO: анализ граничных полигонов
                # если слева и справа свободно — значит возникает развилка, т.е. это критическая точка
                if not left_intersects and not right_intersects:
                    mid_point_arc = ArcPointInfo(point_type=ArcPointInfo.TYPE_CRITICAL, point=mid_point,
                                                 polygon_id_connect=polygon_id)

                    mid_arc_boundaries = self.get_closest_poly_intersections(mid_point)

                    mid_point_redundant = False

                    if self.are_points_equal(mid_arc_boundaries[0].point, max_point) \
                            or self.are_points_equal(mid_arc_boundaries[1].point, max_point) \
                            or self.are_points_equal(mid_arc_boundaries[0].point, min_point) \
                            or self.are_points_equal(mid_arc_boundaries[1].point, min_point):
                        mid_point_redundant = True

                    if not mid_point_redundant:
                        self.morse_arcs.append(ArcInfo(radius=mid_point_radius, center=self.start_point,
                                               left_border=mid_arc_boundaries[0], critical=mid_point_arc,
                                               right_border=mid_arc_boundaries[1]))

                # если и слева и справа от точки полигон, значит это угол выпуклой внутрь области, который ограничивает
                # ее в паре с критической точкой, вызывающей развилку в эту область
                if left_intersects and right_intersects or \
                        left_intersects and self.is_point_on_boundary(mid_point) or \
                        right_intersects and self.is_point_on_boundary(mid_point):
                    mid_point_arc = ArcPointInfo(point_type=ArcPointInfo.TYPE_CRITICAL, point=mid_point,
                                                 polygon_id_connect=polygon_id)

                    self.morse_arcs.append(ArcInfo(radius=mid_point_radius, center=self.start_point,
                                        left_border=mid_point_arc, critical=mid_point_arc,
                                        right_border=mid_point_arc))

        sorted_arcs = sorted(self.morse_arcs, key=lambda arc: arc.radius)

        # TODO: проверить случай, когда дугу образуют несколько полигонов
        # Могут быть ситуации, когда две дуги от разных критических точек накладываются друг на друга.
        # В рамках алгоритма полагаем, что все дуги содержат максимум 3 точки, чтобы было проще считать
        # Если такая ситуация возникает — оставляем только 1 дугу из накладывающихся
        indices_to_remove = []
        for i in range(len(sorted_arcs) - 1):
            if abs(sorted_arcs[i].radius - sorted_arcs[i+1].radius) > MorseCoverage.ARC_RADIUS_EPS:
                continue
            if sorted_arcs[i].critical.polygon_id_connect != sorted_arcs[i+1].critical.polygon_id_connect:
                continue

            if not (sorted_arcs[i].left_border.point == sorted_arcs[i].right_border.point
                    and sorted_arcs[i+1].left_border.point == sorted_arcs[i+1].right_border.point):
                indices_to_remove.append(i)

        for index in reversed(indices_to_remove):
            del sorted_arcs[index]

        # Проставляем id для всех дух
        for i in range(len(sorted_arcs)):
            sorted_arcs[i].set_id(i)

        self.morse_arcs = sorted_arcs

        return self.morse_arcs

    def is_border_on_left(self, point: Point):
        point_radius, point_phi = utils.cartesian_to_polar(point, self.start_point)
        point_phi -= MorseCoverage.GAP_ARC_CRITICAL_OFFSET
        check_point = utils.polar_to_cartesian((point_radius, point_phi), self.start_point)

        return self.is_point_on_boundary(check_point)

    def is_border_on_right(self, point: Point):
        point_radius, point_phi = utils.cartesian_to_polar(point, self.start_point)
        point_phi += MorseCoverage.GAP_ARC_CRITICAL_OFFSET
        check_point = utils.polar_to_cartesian((point_radius, point_phi), self.start_point)

        return self.is_point_on_boundary(check_point)

    def get_closest_poly_intersections(self, critical_point: Point) -> Tuple[ArcPointInfo, ArcPointInfo]:
        # TODO: можно ввести оптимизацию, ограничив количество полигонов на основе того, что
        # дистанцией до них не должна превышать дистанцию до границ поля
        closest_polygons = sorted(self.polygons, key=lambda poly_info: poly_info.polygon.distance(critical_point))
        polygons_distances = sorted([poly_info.polygon.distance(critical_point) for poly_info in self.polygons])
        circle_radius, critical_phi = utils.cartesian_to_polar(critical_point, self.start_point)

        min_point_left = None
        min_point_right = None
        min_distance_left = 1e5
        min_distance_right = 1e5

        # Проходим по всем полигонам, начиная с ближайшего
        for i in range(len(closest_polygons)):
            closest_poly_info = closest_polygons[i]
            closest_poly = closest_poly_info.polygon
            closest_poly_id = closest_poly_info.poly_id

            closest_bounds = utils.get_polygon_lines(closest_poly)
            # Анализируем каждую границу полигона
            # Можно использовать и intersection от shapely, но библиотека возвращает пересечения не только с гранями,
            # но и с внутренней областью, а нужно анализировать только грани
            for closest_bound in closest_bounds:
                intersection = utils.get_line_circle_intersections(line=closest_bound, circle_center=self.start_point,
                                                                   circle_radius=circle_radius)
                # Проверяем, что пересечение с гранью есть и что это пересечение отлично от самОй критической точки
                if len(intersection) > 0 and not MorseCoverage.are_points_equal(intersection[0], critical_point):
                    # Определеяем левая точка или правая от критической точки
                    tmp_distance = intersection[0].distance(critical_point)
                    _, intersection_phi = utils.cartesian_to_polar(intersection[0], self.start_point)

                    # левая точка
                    if intersection_phi > critical_phi:
                        if tmp_distance < min_distance_left:
                            min_distance_left = tmp_distance
                            min_point_left = ArcPointInfo(point_type=ArcPointInfo.TYPE_BOUNDARY, point=intersection[0], polygon_id_connect=closest_poly_id)
                    # правая
                    else:
                        if tmp_distance < min_distance_right:
                            min_distance_right = tmp_distance
                            min_point_right = ArcPointInfo(point_type=ArcPointInfo.TYPE_BOUNDARY, point=intersection[0], polygon_id_connect=closest_poly_id)

            # Если мы нашли две близкие точки пересечения — найти еще более близкие мы можем только в том случае
            # Если в массиве отсортированных полигонов еще есть полигоны, которые ближе к рассматриваемой точке,
            # чем расстояние до найденных пересечений
            if min_point_left is not None and min_point_right is not None:
                if min_distance_left < polygons_distances[i] and min_distance_right < polygons_distances[i]:
                    break

        left_border_point = None
        right_border_point = None
        # Если слева или справа не нашли пересечения с полигоном, то надо найти пересечение с полем, чтобы ограничить дугу
        if min_point_left is None or min_point_right is None:
            for border in self.field_boundaries:
                border_intersection = utils.get_line_circle_intersections(
                    line=border, circle_center=self.start_point, circle_radius=circle_radius)
                if len(border_intersection) > 0:
                    critical_on_border = False
                    # TODO: привести в порядок и разобраться
                    if MorseCoverage.are_points_equal(border_intersection[0], critical_point):
                        left_check_point = utils.polar_to_cartesian((circle_radius, critical_phi + 0.1), self.start_point)
                        right_check_point = utils.polar_to_cartesian((circle_radius, critical_phi - 0.1), self.start_point)

                        if left_check_point.x < self.x_boundary[0] or left_check_point.x > self.x_boundary[1] or left_check_point.y < self.y_boundary[0] or left_check_point.y > self.x_boundary[1]:
                            left_border_point = ArcPointInfo(point_type=ArcPointInfo.TYPE_BOUNDARY,
                                                             point=critical_point,
                                                             polygon_id_connect=MorseCoverage.BORDER_POLYGON_ID)
                            critical_on_border = True
                        if right_check_point.x < self.x_boundary[0] or right_check_point.x > self.x_boundary[1] or right_check_point.y < self.y_boundary[0] or right_check_point.y > self.x_boundary[1]:
                            right_border_point = ArcPointInfo(point_type=ArcPointInfo.TYPE_BOUNDARY,
                                                              point=critical_point,
                                                              polygon_id_connect=MorseCoverage.BORDER_POLYGON_ID)
                            critical_on_border = True

                    if critical_on_border:
                        continue

                    _, intersection_phi = utils.cartesian_to_polar(border_intersection[0], self.start_point)

                    # граница слева
                    if intersection_phi > critical_phi:
                        left_border_point = ArcPointInfo(point_type=ArcPointInfo.TYPE_BOUNDARY,
                                                         point=border_intersection[0],
                                                         polygon_id_connect=MorseCoverage.BORDER_POLYGON_ID)
                    # граница справа
                    else:
                        right_border_point = ArcPointInfo(point_type=ArcPointInfo.TYPE_BOUNDARY,
                                                          point=border_intersection[0],
                                                          polygon_id_connect=MorseCoverage.BORDER_POLYGON_ID)

                    if len(border_intersection) == 2:
                        # Может быть такое, что пересечения с одной и той же границей 2, тогда надо обработать и вторую
                        # точку

                        _, intersection_phi = utils.cartesian_to_polar(border_intersection[1], self.start_point)
                        # граница слева
                        if intersection_phi > critical_phi:
                            left_border_point = ArcPointInfo(point_type=ArcPointInfo.TYPE_BOUNDARY,
                                                             point=border_intersection[1],
                                                             polygon_id_connect=MorseCoverage.BORDER_POLYGON_ID)
                        # граница справа
                        else:
                            right_border_point = ArcPointInfo(point_type=ArcPointInfo.TYPE_BOUNDARY,
                                                              point=border_intersection[1],
                                                              polygon_id_connect=MorseCoverage.BORDER_POLYGON_ID)

                if left_border_point is not None and right_border_point is not None:
                    break

        if min_point_left is None:
            min_point_left = left_border_point
        if min_point_right is None:
            min_point_right = right_border_point

        return min_point_left, min_point_right

    def get_field_areas_and_graph(self, clear: bool = False, verbose: Optional[bool] = True):
        if self.coverage_graph is not None:
            if not clear:
                return self.areas, self.coverage_graph
            else:
                self.areas.clear()
                self.coverage_graph = None

        arc_parts_dict: Dict[Tuple[int, int], List[ArcPartInfo]] = {}
        # Отсортированные дуги по радиусу
        sorted_arcs: List[ArcInfo] = self.get_arcs()

        start_arc = sorted_arcs[0]


        # arc_parts_dict[(start_arc.left_border.polygon_id_connect, start_arc.critical.polygon_id_connect)] = [ArcPartInfo(start_arc, ArcPartInfo.ARC_PART_LEFT)]
        # right_init_part = (start_arc.critical.polygon_id_connect, start_arc.right_border.polygon_id_connect)
        # if right_init_part in arc_parts_dict:
        #     arc_parts_dict[right_init_part].append(ArcPartInfo(start_arc, ArcPartInfo.ARC_PART_RIGHT))
        # else:
        #     arc_parts_dict[(start_arc.critical.polygon_id_connect, start_arc.right_border.polygon_id_connect)] = [ArcPartInfo(start_arc, ArcPartInfo.ARC_PART_RIGHT)]

        last_not_point_arc_index = 0

        for i in range(len(sorted_arcs) - 1, 0, -1):
            arc = sorted_arcs[i]
            if not MorseCoverage.are_points_equal(arc.left_border.point, arc.right_border.point):
                last_not_point_arc_index = i
                break

        finish_arc = sorted_arcs[last_not_point_arc_index]

        # for arc in sorted_arcs[1:-1]:
        for i in range(len(sorted_arcs)):
            arc = sorted_arcs[i]

            full_arc_key = (arc.left_border.polygon_id_connect, arc.right_border.polygon_id_connect)
            left_half_arc_key = (arc.left_border.polygon_id_connect, arc.critical.polygon_id_connect)
            right_half_arc_key = (arc.critical.polygon_id_connect, arc.right_border.polygon_id_connect)

            # частный случай точки
            if MorseCoverage.are_points_equal(arc.left_border.point, arc.right_border.point):
                if self.polygons[arc.right_border.polygon_id_connect - 1].poly_type != PolygonInfo.TYPE_FREE:
                    full_arc_key = (MorseCoverage.BORDER_POLYGON_ID, MorseCoverage.BORDER_POLYGON_ID)

                if full_arc_key in arc_parts_dict:
                    arc_parts_dict[full_arc_key].append(ArcPartInfo(arc, ArcPartInfo.ARC_PART_POINT))
                else:
                    arc_parts_dict[full_arc_key] = [ArcPartInfo(arc, ArcPartInfo.ARC_PART_POINT)]
                continue

            # частный случай когда критическая точка является левой границей
            if MorseCoverage.are_points_equal(arc.left_border.point, arc.critical.point):
                selected_part = right_half_arc_key

                if selected_part in arc_parts_dict:
                    arc_parts_dict[selected_part].append(ArcPartInfo(arc, ArcPartInfo.ARC_PART_FULL))
                else:
                    arc_parts_dict[selected_part] = [ArcPartInfo(arc, ArcPartInfo.ARC_PART_FULL)]
                continue

            # частный случай когда критическая точка является правой границей
            if MorseCoverage.are_points_equal(arc.right_border.point, arc.critical.point):
                selected_part = left_half_arc_key

                if selected_part in arc_parts_dict:
                    arc_parts_dict[selected_part].append(ArcPartInfo(arc, ArcPartInfo.ARC_PART_FULL))
                else:
                    arc_parts_dict[selected_part] = [ArcPartInfo(arc, ArcPartInfo.ARC_PART_FULL)]
                continue

            # частный случай, когда граница дуги находится на граничном полигоне
            if self.polygons[arc.left_border.polygon_id_connect - 1].poly_type != PolygonInfo.TYPE_FREE:
                full_arc_key = (MorseCoverage.BORDER_POLYGON_ID, full_arc_key[1])
                left_half_arc_key = (MorseCoverage.BORDER_POLYGON_ID, left_half_arc_key[1])

            if self.polygons[arc.critical.polygon_id_connect - 1].poly_type != PolygonInfo.TYPE_FREE:
                left_half_arc_key = (left_half_arc_key[0], MorseCoverage.BORDER_POLYGON_ID)
                right_half_arc_key = (MorseCoverage.BORDER_POLYGON_ID, right_half_arc_key[1])

            if self.polygons[arc.right_border.polygon_id_connect - 1].poly_type != PolygonInfo.TYPE_FREE:
                full_arc_key = (full_arc_key[0], MorseCoverage.BORDER_POLYGON_ID)
                right_half_arc_key = (right_half_arc_key[0], MorseCoverage.BORDER_POLYGON_ID)

            if i != 0 and i != last_not_point_arc_index:
                if full_arc_key in arc_parts_dict:
                    arc_parts_dict[full_arc_key].append(ArcPartInfo(arc, ArcPartInfo.ARC_PART_FULL))
                else:
                    arc_parts_dict[full_arc_key] = [ArcPartInfo(arc, ArcPartInfo.ARC_PART_FULL)]

            if left_half_arc_key in arc_parts_dict:
                arc_parts_dict[left_half_arc_key].append(ArcPartInfo(arc, ArcPartInfo.ARC_PART_LEFT))
            else:
                arc_parts_dict[left_half_arc_key] = [ArcPartInfo(arc, ArcPartInfo.ARC_PART_LEFT)]

            if right_half_arc_key in arc_parts_dict:
                arc_parts_dict[right_half_arc_key].append(ArcPartInfo(arc, ArcPartInfo.ARC_PART_RIGHT))
            else:
                arc_parts_dict[right_half_arc_key] = [ArcPartInfo(arc, ArcPartInfo.ARC_PART_RIGHT)]

        # Проверять на наличие таких пар по идее не нужно из-за того, как формируются дуги
        # arc_parts_dict[(finish_arc.left_border.polygon_id_connect, finish_arc.critical.polygon_id_connect)] += [ArcPartInfo(finish_arc, ArcPartInfo.ARC_PART_LEFT)]
        # arc_parts_dict[(finish_arc.critical.polygon_id_connect, finish_arc.right_border.polygon_id_connect)] += [ArcPartInfo(finish_arc, ArcPartInfo.ARC_PART_RIGHT)]

        start_arc_point = ArcPointInfo(point_type=ArcPointInfo.TYPE_CRITICAL, point=self.start_point,
                                       polygon_id_connect=0)
        init_point_arc = ArcInfo(radius=0, center=self.start_point, left_border=start_arc_point,
                                 critical=start_arc_point, right_border=start_arc_point)

        self.areas.append(AreaInfo(area_id=0, input_arc=ArcPartInfo(arc=init_point_arc, part_type=ArcPartInfo.ARC_PART_POINT),
                                   output_arc=ArcPartInfo(arc=start_arc, part_type=ArcPartInfo.ARC_PART_FULL)))

        area_number = 1
        for key in arc_parts_dict.keys():
            list_of_arcs = arc_parts_dict[key]
            for i in range(0, len(list_of_arcs), 2):
                if i + 1 >= len(list_of_arcs):
                    if verbose:
                        print("Ignored pair for '{}'".format(key))
                    continue

                # candidate_index = i + 1
                #
                # if list_of_arcs[i].arc.right_border.polygon_id_connect == list_of_arcs[candidate_index].arc.left_border.polygon_id_connect \
                #     or list_of_arcs[i].arc.left_border.polygon_id_connect == list_of_arcs[candidate_index].arc.right_border.polygon_id_connect

                output_arc = list_of_arcs[i+1]
                self.areas.append(AreaInfo(area_id=area_number, input_arc=list_of_arcs[i], output_arc=output_arc))

                area_number += 1

        end_arc_point = ArcPointInfo(point_type=ArcPointInfo.TYPE_CRITICAL, point=self.end_point,
                                     polygon_id_connect=0)

        end_arc = ArcInfo(radius=self.start_point.distance(self.end_point), center=self.start_point,
                          left_border=end_arc_point, critical=end_arc_point, right_border=end_arc_point)

        self.areas.append(AreaInfo(area_id=area_number, input_arc=ArcPartInfo(arc=finish_arc, part_type=ArcPartInfo.ARC_PART_FULL),
                                   output_arc=ArcPartInfo(arc=end_arc, part_type=ArcPartInfo.ARC_PART_FULL)))

        # Создание графа
        self.coverage_graph = nx.Graph()
        self.coverage_graph.add_nodes_from(list(range(0, len(self.areas))))

        for i in range(len(self.areas)):
            for j in range(i, len(self.areas)):
                output_arc_i = self.areas[i].output_arc
                output_arc_j = self.areas[j].output_arc
                input_arc_i = self.areas[i].input_arc
                input_arc_j = self.areas[j].input_arc

                if output_arc_i is not None and input_arc_j is not None \
                        and output_arc_i.arc.arc_id != -1 \
                        and output_arc_i.arc.arc_id == input_arc_j.arc.arc_id:
                    if not (output_arc_i.is_part_a_point() or input_arc_j.is_part_a_point()):
                        self.coverage_graph.add_edge(i, j)
                        self.coverage_graph.edges[i, j]["weight"] = 1


                if output_arc_j is not None and input_arc_i is not None \
                        and output_arc_j.arc.arc_id != -1 \
                        and output_arc_j.arc.arc_id == input_arc_i.arc.arc_id:
                    if not (output_arc_j.is_part_a_point() or input_arc_i.is_part_a_point()):
                        self.coverage_graph.add_edge(i, j)
                        self.coverage_graph.edges[i, j]["weight"] = 1

        return self.areas, self.coverage_graph

    def get_path(self):
        self.get_field_areas_and_graph()

        return nx.approximation.traveling_salesman_problem(self.coverage_graph)

        # open_tsp_graph = self.coverage_graph.copy().to_directed()

        # for vertex in range(1, len(self.areas)):
        #     open_tsp_graph.add_edge(vertex, 0)
        #     open_tsp_graph.edges[vertex, 0]["weight"] = 0

        # DEPRECATED_WEIGHT = 1e5
        #
        # for vertex_first in range(0, len(self.areas) - 1):
        #     for vertex_second in range(vertex_first + 1, len(self.areas)):
        #         if not open_tsp_graph.has_edge(vertex_first, vertex_second):
        #             open_tsp_graph.add_edge(vertex_first, vertex_second)
        #             open_tsp_graph.edges[vertex_first, vertex_second]["weight"] = DEPRECATED_WEIGHT
        #
        #         if not open_tsp_graph.has_edge(vertex_second, vertex_first):
        #             open_tsp_graph.add_edge(vertex_second, vertex_first)
        #             open_tsp_graph.edges[vertex_second, vertex_first]["weight"] = DEPRECATED_WEIGHT


        # new_cost = sum(open_tsp_graph[n][nbr]["weight"] for n, nbr in nx.utils.pairwise(init_cycle))

        # tsp = nx.approximation.traveling_salesman_problem
        # # https://or.stackexchange.com/questions/6174/travelling-salesman-problem-variant-without-returning-to-the-starting-point
        # return tsp(open_tsp_graph, method=nx.approximation.threshold_accepting_tsp)
        # return tsp(self.coverage_graph)

    # def print_solution(self, manager, routing, solution):
    #     """Prints solution on console."""
    #     print('Objective: {}'.format(solution.ObjectiveValue()))
    #     index = routing.Start(0)
    #     plan_output = 'Route:\n'
    #     route_distance = 0
    #     while not routing.IsEnd(index):
    #         plan_output += ' {} ->'.format(manager.IndexToNode(index))
    #         previous_index = index
    #         index = solution.Value(routing.NextVar(index))
    #         route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    #     plan_output += ' {}\n'.format(manager.IndexToNode(index))
    #     print(plan_output)
    #     plan_output += 'Objective: {}m\n'.format(route_distance)

    def get_mst(self, adj_matrix):
        MST = np.zeros(adj_matrix.shape)
        areas_number = len(adj_matrix)
        visited = [False] * areas_number
        edge_num = 0
        visited[0] = True

        while edge_num < areas_number - 1:
            min_weight = sys.maxsize
            from_vertex = 0
            to_vertex = 0

            for i in range(areas_number):
                if not visited[i]:
                    continue
                for j in range(areas_number):
                    if not visited[j] and adj_matrix[i, j]:
                        if min_weight > adj_matrix[i, j]:
                            min_weight = adj_matrix[i, j]
                            from_vertex= i
                            to_vertex = j
            MST[from_vertex, to_vertex] = 1
            visited[to_vertex] = True
            edge_num += 1

        return MST


class MorseCoverageInduction(MorseCoverage):
    RADIUS_STEP = 1

    def __init__(self, obstacles: List[List[Tuple[float, float]]], start_point: Tuple[float, float],
                 end_point: Tuple[float, float], x_boundary: Tuple[float, float], y_boundary: Tuple[float, float]):
        super().__init__(obstacles, start_point, end_point, x_boundary, y_boundary)

    def get_induction_arcs(self):
        corners = [
            Point(self.x_boundary[0], self.y_boundary[0]),
            Point(self.x_boundary[0], self.y_boundary[1]),
            Point(self.x_boundary[1], self.y_boundary[1]),
            Point(self.x_boundary[1], self.y_boundary[0])
        ]

        corners = sorted(corners, key=lambda corner: self.start_point.distance(corner))
        most_far_corner = corners[-1]
        max_radius = self.start_point.distance(most_far_corner)

        radii = np.arange(0, max_radius, MorseCoverageInduction.RADIUS_STEP)
        radii = np.append(radii, max_radius)

        critical_radii = []

        for radius in radii:
            circle_shapely = self.start_point.buffer(radius)
            radius_intersections = []
            for polygon_info in self.polygons:
                poly_bounds = utils.get_polygon_lines(polygon_info.polygon)
                # Анализируем каждую границу полигона
                # Можно использовать и intersection от shapely, но библиотека возвращает пересечения не только с гранями,
                # но и с внутренней областью, а нужно анализировать только грани
                for poly_bound in poly_bounds:
                    intersection = utils.get_line_circle_intersections(line=poly_bound,
                                                                       circle_center=self.start_point,
                                                                       circle_radius=radius)

                    # if len(intersection) > 0 and not MorseCoverage.are_points_equal(intersection[0], critical_point):
                    #     # Определеяем левая точка или правая от критической точки
                    #     tmp_distance = intersection[0].distance(critical_point)
                    #     _, intersection_phi = utils.cartesian_to_polar(intersection[0], self.start_point)
                    #
                    #     # левая точка
                    #     if intersection_phi > critical_phi:
                    #         if tmp_distance < min_distance_left:
                    #             min_distance_left = tmp_distance
                    #             min_point_left = ArcPointInfo(point_type=ArcPointInfo.TYPE_BOUNDARY,
                    #                                           point=intersection[0], polygon_id_connect=closest_poly_id)
                    #     # правая
                    #     else:
                    #         if tmp_distance < min_distance_right:
                    #             min_distance_right = tmp_distance
                    #             min_point_right = ArcPointInfo(point_type=ArcPointInfo.TYPE_BOUNDARY,
                    #                                            point=intersection[0],
                    #                                            polygon_id_connect=closest_poly_id)

                    if len(intersection) > 0:
                        radius_intersections += intersection

            if len(radius_intersections) > 0:
                critical_radii.append(len(radius_intersections))


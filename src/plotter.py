import matplotlib.pyplot as plt
from typing import List, Tuple, TypeVar, Optional
from shapely.geometry import LineString, Point
import numpy as np
import networkx as nx

from models import *
import utils


def show_performance(times, obstacles):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    # plt.grid()
    # plt.axis([0, 100, 0, 100])

    ax.plot(obstacles, times["arcs"], color="green", label="Arcs decomposing")
    ax.plot(obstacles, times["graph"], color="blue", label="Graph assembling")
    ax.plot(obstacles, times["path"], color="orange", label="Path finding")
    ax.plot(obstacles, times["all"], "--", color="violet", label="Overall")

    ax.set_xlabel("Obstacles Count")
    ax.set_ylabel("Time (secs)")
    ax.legend()
    ax.grid(True)
    plt.show()


def show_field(start_point: Tuple[float, float], finish_point: Tuple[float, float],
               obstacles: List[PolygonInfo], arcs: Optional[List[ArcInfo]] = None,
               areas: Optional[List[AreaInfo]] = None, graph: Optional[nx.Graph] = None):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    plt.grid()
    # TODO: убрать хардкод границ
    plt.axis([0, 100, 0, 100])
    ax.plot(start_point[0], start_point[1], 'bo')
    ax.plot(finish_point[0], finish_point[1], 'bo')


    for poly_info in obstacles:
        obstacle_poly = poly_info.polygon
        obstacle_id = poly_info.poly_id
        obstacle_points = utils.get_points_from_polygon(obstacle_poly)
        x = []
        y = []
        for j in range(len(obstacle_points)):
            x.append(obstacle_points[j].x)
            y.append(obstacle_points[j].y)
        ax.fill(x, y, "b")
        ax.annotate(obstacle_id, xy=(obstacle_poly.centroid.x, obstacle_poly.centroid.y), fontsize=10, va="center", ha="center",
                    zorder=11)

    if arcs is not None:
        for arc in arcs:
            arc_xy = get_arc_xy(arc, center_point=start_point)

            ax.plot(arc_xy[0], arc_xy[1])

            ax.plot(arc.left_border.point.x, arc.left_border.point.y, marker="o", linewidth=50, color="lime")
            ax.plot(arc.right_border.point.x, arc.right_border.point.y, marker="o", linewidth=50, color="lime")
            ax.plot(arc.critical.point.x, arc.critical.point.y, marker="o", linewidth=50, color="red")
            ax.annotate(arc.arc_id, xy=(arc.critical.point.x + 1, arc.critical.point.y + 1), color="violet", fontsize=10, va="center",
                        ha="center", zorder=11)

    if graph is not None:
        for edge in graph.edges:
            first_point = areas[edge[0]].get_center_point()
            second_point = areas[edge[1]].get_center_point()
            ax.plot([first_point.x, second_point.x], [first_point.y, second_point.y], color="black")

        for area in areas:
            area_id = area.area_id
            area_center_point = area.get_center_point()

            marker_circle = plt.Circle((area_center_point.x, area_center_point.y), radius=2, color="gray", zorder=10)
            ax.add_patch(marker_circle)

            label = ax.annotate(area_id, xy=(area_center_point.x, area_center_point.y), fontsize=10, va="center", ha="center", zorder=11)

    plt.show()


def show_coverage_path(areas: List[AreaInfo], graph: nx.Graph, path: List[int]):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    plt.grid()
    plt.axis([0, 100, 0, 100])

    for edge in graph.edges:
        first_point = areas[edge[0]].get_center_point()
        second_point = areas[edge[1]].get_center_point()
        ax.plot([first_point.x, second_point.x], [first_point.y, second_point.y], color="lightgray")

    for i in range(0, len(path) - 1):
        first_point = areas[path[i]].get_center_point()
        second_point = areas[path[i+1]].get_center_point()
        ax.plot([first_point.x, second_point.x], [first_point.y, second_point.y], color="blue")

    for area in areas:
        area_id = area.area_id
        area_center_point = area.get_center_point()

        marker_circle = plt.Circle((area_center_point.x, area_center_point.y), radius=2, color="gray", zorder=10)
        ax.add_patch(marker_circle)

        label = ax.annotate(area_id, xy=(area_center_point.x, area_center_point.y), fontsize=10, va="center", ha="center", zorder=11)

    plt.show()


def get_arc_xy(arc: ArcInfo, center_point: Tuple[float, float]) -> Tuple[List[float], List[float]]:
    center_point_obj = Point(center_point)

    _, left_phi = utils.cartesian_to_polar(arc.left_border.point, center_point_obj)
    _, right_phi = utils.cartesian_to_polar(arc.right_border.point, center_point_obj)
    arc_linestring = utils.get_arc_linestring(center_point_obj, arc.radius, (left_phi, right_phi))

    return arc_linestring.xy

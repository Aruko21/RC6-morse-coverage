from pathlib import Path
import morse_path_finder as coverage
import plotter as plotter
import time


DATA_DIR = "data"
# FIELD_JSON_FILE = "test_auto.json"
# FIELD_JSON_FILE = "test_auto9_broken.json"
FIELD_JSON_FILE = "field_edges.json"


def main():
    field_file_path = Path("..") / DATA_DIR / FIELD_JSON_FILE

    if not field_file_path.exists():
        errmsg = "There is no {} file for field. Aborting".format(field_file_path)
        print(errmsg)
        raise ValueError(errmsg)

    sp, fp, obst = coverage.parse_field_from_file(field_file_path)
    print("Number of obstacles: ", len(obst))

    print('Start point ->', sp)
    print('Finish point ->', fp)
    print('Obstacles:')
    for i in range(len(obst)):
        print(i, ':', obst[i])

    sum_time = 0

    test_induction = coverage.MorseCoverageInduction(obstacles=obst, start_point=sp, end_point=fp, x_boundary=(0, 100),
                                                     y_boundary=(0, 100))

    plotter.show_field(sp, fp, test_induction.polygons)
    test_induction.get_induction_arcs()

    return


    print("Initialising solver")
    cur_time = time.time()
    morse_coverage = coverage.MorseCoverage(obstacles=obst, start_point=sp, end_point=fp, x_boundary=(0, 100),
                                            y_boundary=(0, 100))

    init_time = time.time() - cur_time
    sum_time += init_time
    print("Solver initialised in: ", init_time)

    plotter.show_field(sp, fp, morse_coverage.polygons)

    print("Decomposing by arcs")
    cur_time = time.time()
    arcs = morse_coverage.get_arcs()
    arcs_time = time.time() - cur_time
    sum_time += arcs_time
    print("Decomposed in: ", arcs_time)

    plotter.show_field(sp, fp, morse_coverage.polygons, arcs)

    cur_time = time.time()
    areas, graph = morse_coverage.get_field_areas_and_graph(verbose=True)
    graph_time = time.time() - cur_time
    sum_time += graph_time

    print("Graph created in: ", graph_time)

    plotter.show_field(sp, fp, morse_coverage.polygons, arcs, areas, graph)

    print("Solving TSP")
    cur_time = time.time()
    tsp_path = morse_coverage.get_path()
    tsp_time = time.time() - cur_time
    sum_time += tsp_time

    print(tsp_path)
    plotter.show_coverage_path(areas, graph, tsp_path)

    print("TSP solved in: ", tsp_time)

    print("Total time: ", sum_time)

    # plotter.show_field(sp, fp, morse_coverage.polygons, arcs, areas, mst, True)

if __name__ == "__main__":
    main()

#include <brandes.cuh>
#include <fstream>
#include <graph.cuh>
#include <iostream>
#include <limits>
#include <map>
#include <utility>
#include <vector>

std::vector<std::pair<int, int> > read_graph(std::ifstream& stream_) {
    std::vector<std::pair<int, int> > edges;
    int v1, v2;

    while (stream_ >> v1 >> v2) {
        edges.push_back(std::make_pair(v1, v2));
    }

    return edges;
}

#define EXIT_ERROR(str)                                                                           \
    {                                                                                             \
        std::cerr << "Usage: " << str << " <in_file> <out_file> [--show-results] [--mdeg number]" \
                  << std::endl;                                                                   \
        exit(1);                                                                                  \
    }

int main(int argc, char** argv) {
    if (argc < 3) {
        EXIT_ERROR(argv[0])
    }
    bool verbose = false;
    int mdeg = 6;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        if (s.compare("-v") == 0 || s.compare("--verbose") == 0) {
            verbose = true;
            break;
        }

        if (s.compare("--mdeg") == 0) {
            if (i + 1 < argc) {
                mdeg = std::stoi(argv[i + 1]);
            } else {
                EXIT_ERROR(argv[0])
            }
            i++;
        }
    }

    std::ifstream infile(argv[1]);
    std::ofstream outfile(argv[2]);

    float time = 0;
    CPUStrideCSRGraph graph = CPUStrideCSRGraph(read_graph(infile), mdeg);
    double* bc = run_stride_csr(graph, time, verbose);

    outfile.precision(std::numeric_limits<double>::max_digits10);
    for (int i = 0; i < graph.num_vertices; ++i) {
        outfile << bc[i] << std::endl;
    }

    delete[] bc;
    return 0;
}

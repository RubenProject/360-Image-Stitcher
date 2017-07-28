#ifndef astarHVar  
#define astarHVar

// represents a single pixel
class Node {
  public:
    int idx;     // index in the flattened grid
    float cost;  // cost of traversing this pixel

    Node(int i, float c) : idx(i),cost(c) {}
};

bool astar(const float* weights, const int height, const int width,
          const int start, const int goal, int* paths);

bool operator<(const Node &n1, const Node &n2);

bool operator==(const Node &n1, const Node &n2);

float heuristic(int i0, int j0, int i1, int j1);

#endif

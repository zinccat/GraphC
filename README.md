# GraphC

Graph classification using graph neural networks.

## Data format
Let

n = total number of nodes

m = total number of edges

N = number of graphs

DS_A.txt (m lines): sparse (block diagonal) adjacency matrix for all graphs, each line corresponds to (row, col) resp. (node_id, node_id). All graphs are undirected. Hence, DS_A.txt contains two entries for each edge.

DS_graph_indicator.txt (n lines): column vector of graph identifiers for all nodes of all graphs, the value in the i-th line is the graph_id of the node with node_id i

DS_graph_labels.txt (N lines): class labels for all graphs in the data set, the value in the i-th line is the class label of the graph with graph_id i

DS_node_labels.txt (n lines): column vector of node labels, the value in the i-th line corresponds to the node with node_id i

DS_edge_attributes.txt (m lines): contains the attributes of the edges in the sparse (block diagonal) adjacency matrix. Each line contains the attributes of the edge (row, col) resp. (node_id, node_id)
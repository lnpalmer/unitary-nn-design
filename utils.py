import random
from pygraphviz import AGraph

def draw_net(net, path):
    N, I, O = net.N, net.I, net.O

    net.sync_graph()
    AG = AGraph(directed=True, dpi=300)
    AG.graph_attr["outputorder"] = "edgesfirst"
    AG.graph_attr["bgcolor"] = "#000000"

    for node in net._graph.nodes(data=True):
        i, props = node
        random.seed(i)
        if i < I:
            AG.add_node(i, shape="point", pos=("%f,%f!" % (0 - .4, 4. * i / (I - 1))))
        elif i >= N - O:
            i_ = i - (N - O)
            AG.add_node(i, shape="point", pos=("%f,%f!" % (6 + .4, 4. * i_ / (O - 1))))
        else:
            i_ = i - I
            AG.add_node(i,
                        shape="point",
                        pos=("%f,%f!" % (6. * i_ / (N - (I + O) + 1), .25 + 3.5 * random.random())),
                        label="")

    for edge in net._graph.edges(data=True):
        j, i, props = edge
        w_ij = props["weight"]
        color = "#FFAF7F" if w_ij > 0. else "#CF7FCF"
        AG.add_edge(j, i, penwidth=abs(w_ij) * .3, arrowsize=abs(w_ij) * .05 + .3, color=color)

    AG.draw(path, prog="neato")

def gae(self):
    raise NotImplementedError

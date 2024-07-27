import math

from graphviz import Digraph


class Value:
    def __init__(self, data, _children=(), label="", _op="") -> None:
        self.data = data
        self._prev = set(_children)
        self.label = label
        self.grad = 0
        self._op = _op

    def backward(self):
        # ???
        return None

    def __repr__(self) -> str:
        return f"Value(data={self.data}) | children={self._prev} | Label:{self.label} "

    def __add__(self, other):
        x = self.data + other.data
        out = Value(x, _children=(self, other))
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data)
        return out

    def __pow__(self, other):
        out = Value(self.data**other)
        return out

    def tang(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t)
        return out


def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
            shape="record",
        )
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name=uid + n._op, label=n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

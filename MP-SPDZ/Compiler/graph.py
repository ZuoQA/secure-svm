import heapq
from Compiler.exceptions import *

class GraphError(CompilerError):
    pass

class SparseDiGraph(object):
    """ Directed graph suitable when each node only has a small number of edges.

    Edges are stored as a list instead of a dictionary to save memory, leading
    to slower searching for dense graphs.

    Node attributes must be specified in advance, as these are stored in the
    same list as edges.
    """
    def __init__(self, max_nodes, default_attributes=None):
        """ max_nodes: maximum no of nodes
        default_attributes: dict of node attributes and default values """
        if default_attributes is None:
            default_attributes = { 'merges': None }
        self.default_attributes = default_attributes
        self.attribute_pos = dict(list(zip(list(default_attributes.keys()), list(range(len(default_attributes))))))
        self.n = max_nodes
        # each node contains list of default attributes, followed by outoing edges
        self.nodes = [list(self.default_attributes.values()) for i in range(self.n)]
        self.succ = [set() for i in range(self.n)]
        self.pred = [set() for i in range(self.n)]
        self.weights = {}

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        """ Get list of the neighbours of node i """
        return self.succ[i]

    def __iter__(self):
        pass #return iter(self.nodes)

    def __contains__(self, i):
        return i >= 0 and i < self.n

    def add_node(self, i, **attr):
        if i >= self.n:
            raise CompilerError('Cannot add node %d to graph of size %d' % (i, self.n))
        node = self.nodes[i]

        for a,value in list(attr.items()):
            if a in self.default_attributes:
                node[self.attribute_pos[a]] = value
            else:
                raise CompilerError('Invalid attribute %s for graph node' % a)

    def set_attr(self, i, attr, value):
        if attr in self.default_attributes:
            self.nodes[i][self.attribute_pos[attr]] = value
        else:
            raise CompilerError('Invalid attribute %s for graph node' % attr)

    def get_attr(self, i, attr):
        return self.nodes[i][self.attribute_pos[attr]]

    def remove_node(self, i):
        """ Remove node i and all its edges """
        succ = self[i]
        pred = self.pred[i]
        for v in succ:
            self.pred[v].remove(i)
            #del self.weights[(i,v)]
        for v in pred:
            self.succ[v].remove(i)
            #del self.weights[(v,i)]
            #self.nodes[v].remove(i)
        self.pred[i] = []
        self.nodes[i] = list(self.default_attributes.values())

    def add_edge(self, i, j, weight=1):
        if j not in self[i]:
            self.pred[j].add(i)
            self.succ[i].add(j)
        self.weights[(i,j)] = weight

    def add_edges_from(self, tuples):
        for edge in tuples:
            if len(edge) == 3:
                # use weight
                self.add_edge(edge[0], edge[1], edge[2])
            else:
                self.add_edge(edge[0], edge[1])

    def remove_edge(self, i, j):
        self.succ[i].remove(j)
        self.pred[j].remove(i)
        del self.weights[(i,j)]

    def remove_edges_from(self, pairs):
        for i,j in pairs:
            self.remove_edge(i, j)

    def degree(self, i):
        return len(self.succ[i])


def topological_sort(G, nbunch=None, pref=None):
    seen={}
    order_explored=[] # provide order and 
    explored={}       # fast search without more general priorityDictionary
    
    if pref is None:
        def get_children(node):
            return G[node]
    else:
        def get_children(node):
            if node in pref:
                pref_set = set(pref[node])
                for i in G[node]:
                    if i not in pref_set:
                        yield i
                for i in reversed(pref[node]):
                    yield i
            else:
                for i in G[node]:
                    yield i

    if nbunch is None:
        nbunch = reversed(list(range(len(G))))
    for v in nbunch:     # process all vertices in G
        if v in explored: 
            continue
        fringe=[v]   # nodes yet to look at
        while fringe:
            w=fringe[-1]  # depth first search
            if w in explored: # already looked down this branch
                fringe.pop()
                continue
            seen[w]=1     # mark as seen
            # Check successors for cycles and for new nodes
            new_nodes=[]
            for n in get_children(w):
                if n not in explored:
                    if n in seen: #CYCLE !!
                        raise GraphError("Graph contains a cycle at %d (%s,%s)." % \
                                                        (n, G[n], G.pred[n]))
                    new_nodes.append(n)
            if new_nodes:   # Add new_nodes to fringe
                fringe.extend(new_nodes)
            else:           # No new nodes so w is fully explored
                explored[w]=1
                order_explored.append(w)
                fringe.pop()    # done considering this node
    
    order_explored.reverse() # reverse order explored
    return order_explored

def dag_shortest_paths(G, source):
    top_order = topological_sort(G)
    dist = [None] * len(G)
    dist[source] = 0
    for u in top_order:
        if dist[u] is None:
            continue
        for v in G[u]:
            if dist[v] is None or dist[v] > dist[u] + G.weights[(u,v)]:
                dist[v] = dist[u] + G.weights[(u,v)]
    return dist

def reverse_dag_shortest_paths(G, source):
    top_order = reversed(topological_sort(G))
    dist = [None] * len(G)
    dist[source] = 0
    for u in top_order:
        if u ==68273:
            print('dist[68273]', dist[u])
            print('pred[u]', G.pred[u])
        if dist[u] is None:
            continue
        for v in G.pred[u]:
            if dist[v] is None or dist[v] > dist[u] + G.weights[(v,u)]:
                dist[v] = dist[u] + G.weights[(v,u)]
    return dist

def single_source_longest_paths(G, source, reverse=False):
    # make weights negative, then do shortest paths
    for edge in G.weights:
        G.weights[edge] = -G.weights[edge]
    if reverse:
        dist = reverse_dag_shortest_paths(G, source)
    else:
        dist = dag_shortest_paths(G, source)
    #dist = johnson(G, sources)
    # reset weights
    for edge in G.weights:
        G.weights[edge] = -G.weights[edge]
    for i,n in enumerate(dist):
        if n is None:
            dist[i] = 0
        else:
            dist[i] = -dist[i]
    #for k, v in dist.iteritems():
    #    dist[k] = -v
    return dist


def longest_paths(G, sources=None):
    # make weights negative, then do shortest paths
    for edge in G.weights:
        G.weights[edge] = -G.weights[edge]
    dist = {}
    for source in sources:
        print(('%s, ' % source), end=' ')
        dist[source] = dag_shortest_paths(G, source)
    #dist = johnson(G, sources)
    # reset weights
    for edge in G.weights:
        G.weights[edge] = -G.weights[edge]
    return dist

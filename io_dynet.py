import dynet as dy
from itertools import product

# lengths of words of the form a^n
corpus = [1, 3, 5, 2]

# rules
# 0: S -> a
# 1: S -> S S

# function that builds the hypergraph from length
def build_hypergraph_rec(i, graph={}):
  if i not in graph: 
    if i == 1:
      graph[i] = [(0, [])]
    else: 
      inputs = []
      for j in range(1, i):
        build_hypergraph_rec(j, graph)
        build_hypergraph_rec(i-j, graph)
        inputs.append((1, [j, i-j]))
      graph[i] = inputs
  return graph

# converts rule probabilies and hypergraph into dynet computation network
def build_network(p_, graph):
  dy.renew_cg()
  
  p = {i: dy.parameter(p_[i]) for i in p_}
  nodes = sorted([n for n in graph])
  
  network_nodes = {}
  for n in nodes:
    s = dy.zeros(dim=1)
    for inp in graph[n]:
      prod = p[inp[0]]
      for child in inp[1]:
        prod = prod * network_nodes[child]
      s = s + prod
    network_nodes[n] = s
  return network_nodes, nodes[-1] 

m = dy.ParameterCollection()
initial_values = [0.2, 0.8]
p = {}
for idx, val in enumerate(initial_values):
  p[idx] = m.add_parameters((1), init=dy.ConstInitializer(val))
trainer = dy.AdamTrainer(m, alpha=0.01)

for i in range(1,4):
  print("\nSTART of Epoch", i, "\n")
  counts = {}
  for idx in p:
    print("rule", idx, "prob:", dy.parameter(p[idx]).value())
    counts[idx] = 0.0
  print()
  
  for elem in corpus:
    hypergraph = build_hypergraph_rec(elem, {})
    print("hypergraph for", elem, ":", hypergraph)
    network, output = build_network(p, hypergraph)
    loss = 1 * network[output]
    loss.backward()
    for n in range(1, output + 1):
      print("node", n, "value", network[n].value(), "gradient", network[n].gradient())
    for r in p:
      rv = dy.parameter(p[r])
      count = (rv.gradient() * rv.value() / loss.value())[0]
      print("rule", r, "prob", rv.value(), "gradient", rv.gradient(), "count", count)
      counts[r] += count
    print()
  
  print("Total counts of epoch", i, ":", counts)
  c_sum = sum([counts[r] for r in counts])
  for r in counts:
    p[r] = m.add_parameters(1, init=dy.ConstInitializer(counts[r] / c_sum))

print("Final rule weights")
for idx in p:
  print(idx, dy.parameter(p[idx]).value())

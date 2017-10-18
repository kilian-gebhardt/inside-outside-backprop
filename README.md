# inside-outside-backprop

A small script that facilitates [dynet](https://github.com/clab/dynet) to implement
the inside-outside algorithm as backpropagition as outlined by [Jason Eisner](https://www.cs.jhu.edu/~jason/papers/eisner.spnlp16.pdf).

A very simple context-free grammar with the following rules is used:
```
S -> a   (idx: 0)
S -> SS  (idx: 1)
```

- All words are of the form `a^n`. For each word the hypergraph that encodes the intersection grammar 
(also known as reduct, parse forest, ...) can easily be constructed, without doing actual parsing.
- Each hypergraph is transformed into a dynet computation graph which computes the inside weight of each vertex.
- Rule counts are computed by backpropagation and accumulated over a corpus. 
- The rule weights are normalized to update the rule probabilites.

The algorithm already converges after the first iteration.

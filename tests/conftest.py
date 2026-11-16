import pytest

# A small, valid neuron: 3-point soma, one basal dendrite that bifurcates,
# one apical dendrite and one axon.
SIMPLE_SWC = """\
# id type x y z r parent
1 1 0 0 0 8 -1
2 1 0 -8 0 8 1
3 1 0 8 0 8 1
4 3 -8 0 0 1 1
5 3 -60 0 0 1 4
6 3 -100 40 0 0.6 5
7 3 -100 -40 0 0.6 5
8 4 0 20 0 1.5 1
9 4 0 300 0 1.2 8
10 2 0 -10 0 0.5 1
11 2 0 -400 0 0.5 10
"""

# Same neuron but with a branched multi-point soma, which MorphIO rejects.
BRANCHED_SOMA_SWC = """\
1 1 0 0 0 4 -1
2 1 0 3 0 4 1
3 1 -2 6 0 2 2
4 1 2 6 0 2 2
5 3 -2 9 0 1 3
6 3 -2 90 0 1 5
7 4 2 9 0 1.5 4
8 4 2 300 0 1.2 7
"""


@pytest.fixture
def simple_swc():
    return SIMPLE_SWC


@pytest.fixture
def branched_soma_swc():
    return BRANCHED_SOMA_SWC

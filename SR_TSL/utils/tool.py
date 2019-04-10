import torch as tor
def expendRank( t, m):
    [k, b] = m.size()
    tm = tor.zeros(k, t, b)
    for i in range(k):
        for j in range(t):
            for x in range(b):
                tm[i][j][x] = m[i][x]
    return tm.contiguous()
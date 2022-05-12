

def make_embedding(in_firsts, out_firsts, sizes):
    assert len(in_firsts) == len(out_firsts) == len(sizes)

    length = len(in_firsts)
    embedding = {}
    for i in range(length):
        in_first = in_firsts[i]
        out_first = out_firsts[i]
        for j in range(sizes[i]):
            embedding[(in_first[0]-j, in_first[1] + j)
                      ] = out_first[0], out_first[1]+j
    return embedding


in_firsts = [
    (0, 0), (2, 0), (4, 0), (6, 0), (7, 1), (7, 3), (7, 5), (7, 7)]
out_firsts = [(0, 3), (1, 2), (2, 1), (3, 0), (4, 0), (5, 1), (6, 2), (7, 3)]
sizes = [1, 3, 5, 7, 7, 5, 3, 1]
embedding = make_embedding(in_firsts, out_firsts, sizes)
print(embedding)

in_firsts = [
    (1, 0), (3, 0), (5, 0), (7, 0), (7, 2), (7, 4), (7, 6)]

out_firsts = [(1, 3), (2, 2), (3, 1), (4, 0), (5, 1), (6, 2), (7, 3)]
sizes = [2, 4, 6, 8, 6, 4, 2]

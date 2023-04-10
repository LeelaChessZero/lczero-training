import numpy as np


move = np.arange(1, 8)

diag = np.array([
    move    + move*8,
    move    - move*8,
    move*-1 - move*8,
    move*-1 + move*8
])

orthog = np.array([
    move,
    move*-8,
    move*-1,
    move*8
])

knight = np.array([
    [2 + 1*8],
    [2 - 1*8],
    [1 - 2*8],
    [-1 - 2*8],
    [-2 - 1*8],
    [-2 + 1*8],
    [-1 + 2*8],
    [1 + 2*8]
])

promos = np.array([2*8, 3*8, 4*8])
pawn_promotion = np.array([
    -1 + promos,
    0 + promos,
    1 + promos
])


def make_map():
    """theoretically possible put-down squares (numpy array) for each pick-up square (list element).
    squares are [0, 1, ..., 63] for [a1, b1, ..., h8]. squares after 63 are promotion squares.
    each successive "row" beyond 63 (ie. 64:72, 72:80, 80:88) are for over-promotions to queen, rook, and bishop;
    respectively. a pawn traverse to row 56:64 signifies a "default" promotion to a knight."""
    traversable = []
    for i in range(8):
        for j in range(8):
            sq = (8*i + j)
            traversable.append(
                sq +
                np.sort(
                    np.int32(
                        np.concatenate((
                            orthog[0][:7-j], orthog[2][:j], orthog[1][:i], orthog[3][:7-i],
                            diag[0][:np.min((7-i, 7-j))], diag[3][:np.min((7-i, j))],
                            diag[1][:np.min((i, 7-j))], diag[2][:np.min((i, j))],
                            knight[0] if i < 7 and j < 6 else [], knight[1] if i > 0 and j < 6 else [],
                            knight[2] if i > 1 and j < 7 else [], knight[3] if i > 1 and j > 0 else [],
                            knight[4] if i > 0 and j > 1 else [], knight[5] if i < 7 and j > 1 else [],
                            knight[6] if i < 6 and j > 0 else [], knight[7] if i < 6 and j < 7 else [],
                            pawn_promotion[0] if i == 6 and j > 0 else [],
                            pawn_promotion[1] if i == 6           else [],
                            pawn_promotion[2] if i == 6 and j < 7 else [],
                        ))
                    )
                )
            )
    z = np.zeros((64*64+8*24, 1858), dtype=np.int32)
    # first loop for standard moves (for i in 0:1858, stride by 1)
    i = 0
    for pickup_index, putdown_indices in enumerate(traversable):
        for putdown_index in putdown_indices:
            if putdown_index < 64:
                z[putdown_index + (64*pickup_index), i] = 1
                i += 1
    # second loop for promotions (for i in 1792:1858, stride by ls[j])
    j = 0
    j1 = np.array([3, -2, 3, -2, 3])
    j2 = np.array([3, 3, -5, 3, 3, -5, 3, 3, 1])
    ls = np.append(j1, 1)
    for k in range(6):
        ls = np.append(ls, j2)
    ls = np.append(ls, j1)
    ls = np.append(ls, 0)
    for pickup_index, putdown_indices in enumerate(traversable):
        for putdown_index in putdown_indices:
            if putdown_index >= 64:
                pickup_file = pickup_index % 8
                promotion_file = putdown_index % 8
                promotion_rank = (putdown_index // 8) - 8
                z[4096 + pickup_file*24 + (promotion_file*3+promotion_rank), i] = 1
                i += ls[j]
                j += 1

    return z

def make_pos_enc():
    traversable = []
    for i in range(8):
        for j in range(8):
            sq = (8*i + j)
            traversable.append(
                sq +
                np.sort(
                    np.int32(
                        np.concatenate((
                            orthog[0][:7-j], orthog[2][:j], orthog[1][:i], orthog[3][:7-i],
                            diag[0][:np.min((7-i, 7-j))], diag[3][:np.min((7-i, j))],
                            diag[1][:np.min((i, 7-j))], diag[2][:np.min((i, j))],
                            knight[0] if i < 7 and j < 6 else [], knight[1] if i > 0 and j < 6 else [],
                            knight[2] if i > 1 and j < 7 else [], knight[3] if i > 1 and j > 0 else [],
                            knight[4] if i > 0 and j > 1 else [], knight[5] if i < 7 and j > 1 else [],
                            knight[6] if i < 6 and j > 0 else [], knight[7] if i < 6 and j < 7 else [],
                            # pawn_promotion[0] if i == 6 and j > 0 else [],
                            # pawn_promotion[1] if i == 6           else [],
                            # pawn_promotion[2] if i == 6 and j < 7 else [],
                        ))
                    )
                )
            )

    # pos_enc = np.zeros((1, 64, 88), dtype=np.float32)
    pos_enc = np.zeros((1, 64, 64), dtype=np.float32)
    for i, k in enumerate(traversable):
        pos_enc[0][i][i] = -1.
        for j in k:
            pos_enc[0][i][j] = 1.

    return pos_enc

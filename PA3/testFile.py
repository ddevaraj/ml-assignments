rook_pos = 0
res = 0
for i in board:
    if 'R' in i:
        row = "".join(x for x in i if x != '.')
        if 'Rp' in row:
            res += 1
        if 'pR' in row:
            res += 1
        rook_pos = i.index('R')
col = ''.join(
    board[r][rook_pos] for r in range(8) if board[r][rook_pos] != '.')
#     print(col)
if 'Rp' in col:
    res += 1
if 'pR' in col:
    res += 1
return res

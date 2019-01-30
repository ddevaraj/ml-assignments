import numpy as np
import math
import utils as Util

features = [['a', 'b'], ['b', 'a'], ['b', 'c'], ['c', 'b']]
labels = [0,0,1,1]
num_cls = np.max(labels)+1

def get_entropy(labels):
    n=len(labels)
    unique_labels = np.unique(labels)
    sums = 0
    for label in unique_labels:
        no_instances=labels.count(label)
        p=float(no_instances)/float(n)
        sums+=-(p*math.log(p,2))
    return sums

children = []
# for idx in range(len(features[0])):
#     if not "info_gain" in locals():
#         info_gain = float('-inf')
#     values_at_dimensions = np.array(features)[:, idx]
#     if None in values_at_dimensions:
#         continue
#     value_at_branches = np.unique(values_at_dimensions)
#     branches = np.zeros((num_cls, len(value_at_branches)))
#     # for branch_ind in range(0, len(value_at_branches)):
#     #     pred_vals = np.array(labels)[
#     #         np.where(values_at_dimensions == value_at_branches[branch_ind])]
#     #     for eachPred in pred_vals:
#     #         branches[eachPred, branch_ind] += 1
#     for i, val in enumerate(value_at_branches):
#         y = np.array(labels)[np.where(values_at_dimensions == val)]
#         for yi in y:
#             branches[yi, i] += 1
#     branches = branches.T
#     S = get_entropy(labels)
#     print(branches)
#     IG = Util.Information_Gain(S, branches)
#     # IG = 0
#     # IG = IG +num_cls
#     if IG > info_gain:
#         info_gain = IG
#         dim_split = idx
#         feature_uniq_split = value_at_branches.tolist()
#
# xi = np.array(features)[:, dim_split]
# x = np.array(features, dtype=object)
# x[:, dim_split] = None
# # x = np.delete(self.features, self.dim_split, axis=1)
# for val in feature_uniq_split:
#     indexes = np.where(xi == val)
#     x_new = x[indexes].tolist()
#     y_new = np.array(labels)[indexes].tolist()
#     # child = TreeNode(x_new, y_new, num_cls)
#     if np.array(x_new).size == 0 or all(
#                     v is None for v in x_new[0]):
#         splittable = False
#         print('hi')
#     children.append(1)
#
# # split the child nodes
# for child in children:
#     if child.splittable:
#         child.split()


def parent_entropy(labels):
    n = len(labels)
    unique_labels = np.unique(labels)
    sums = 0
    for label in unique_labels:
        no_instances = labels.count(label)
        p = float(no_instances) / float(n)
        sums += -(p * math.log(p, 2))
    return sums


S = parent_entropy(labels)
th = []
print('features, label, entropy', features, labels, S)
for idx_dim in range(len(features[0])):

    fe = np.array(features).transpose()
    fa = fe[idx_dim]
    fg = set(fa)
    ft = list(fg)
    n = len(ft)
    lb = set(labels)
    lt = list(lb)
    m = len(lt)
    le = len(features)
    rt = np.zeros((m, n))

    dt = dict()
    j = 0
    for af in ft:
        dt[af] = j
        j = j + 1

    i = 0
    dc = dict()
    for fc in lt:
        dc[fc] = i
        i = i + 1

    for index in range(le):
        re = dt[features[index][idx_dim]]
        rf = dc[labels[index]]

        rt[rf][re] += 1
    th.append(Util.Information_Gain(S, rt.T))

if th == []:
    dim_split = None
    feature_uniq_split = None
    splittable = False

yu = max(th)
hj = th.index(yu)
ou = fe[hj]
ot = set(ou)
feature_uniq_split = list(ot)

dim_split = hj
for w in ot:
    kl = []
    kj = []
    for j, dh in enumerate(features):
        if dh[hj] == w:
            kj.append(labels[j])
            kl.append(dh)

    kl = np.delete(kl, hj, 1)
    km = kl.tolist()
    sd = set(kj)
    ln = len(sd)

    children.append(1)


    # split the child nodes
for child in children:
    if child.splittable:
        child.split()


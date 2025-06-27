import sys
import numpy as np

np.set_printoptions(precision=6, suppress=True, floatmode='fixed')


def parse_vertex(line):
    tokens = line.strip().split()
    vertex_type = tokens[0]
    if vertex_type == 'var':
        return {'type': 'var', 'rows': int(tokens[1]), 'cols': int(tokens[2])}
    if vertex_type == 'tnh':
        return {'type': 'tnh', 'input': int(tokens[1]) - 1}
    if vertex_type == 'rlu':
        return {'type': 'rlu', 'alpha_inv': int(tokens[1]), 'input': int(tokens[2]) - 1}
    if vertex_type == 'mul':
        return {'type': 'mul', 'a': int(tokens[1]) - 1, 'b': int(tokens[2]) - 1}
    if vertex_type in ('sum', 'had'):
        count = int(tokens[1])
        inputs = [int(x) - 1 for x in tokens[2:2 + count]]
        return {'type': vertex_type, 'inputs': inputs}


def read_matrix(rows):
    return np.array([list(map(float, sys.stdin.readline().split())) for _ in range(rows)], dtype=float)


def compute_shape(index, vertices):
    v = vertices[index]
    if 'rows' in v and 'cols' in v:
        return v['rows'], v['cols']
    t = v['type']
    if t in ('tnh', 'rlu'):
        r, c = compute_shape(v['input'], vertices)
    elif t == 'mul':
        r, _ = compute_shape(v['a'], vertices)
        _, c = compute_shape(v['b'], vertices)
    elif t in ('sum', 'had'):
        r, c = compute_shape(v['inputs'][0], vertices)
    v['rows'], v['cols'] = r, c
    return r, c


def forw_pass(vertices, inputs):
    res = [None] * len(vertices)
    for i, v in enumerate(vertices):
        t = v['type']
        if t == 'var':
            res[i] = inputs.pop(0)
        elif t == 'tnh':
            res[i] = np.tanh(res[v['input']])
        elif t == 'rlu':
            alpha = 1.0 / v['alpha_inv']
            x = res[v['input']]
            res[i] = np.where(x > 0, x, alpha * x)
        elif t == 'mul':
            res[i] = res[v['a']] @ res[v['b']]
        elif t == 'sum':
            res[i] = sum(res[j] for j in v['inputs'])
        elif t == 'had':
            res = res[v['inputs'][0]].copy()
            for j in v['inputs'][1:]:
                res *= res[j]
            res[i] = res
    return res


def backw_pass(vertices, res, output_grads):
    grads = [np.zeros_like(r) for r in res]
    for i, g in enumerate(output_grads):
        grads[len(vertices) - len(output_grads) + i] = g
    for i in reversed(range(len(vertices))):
        v = vertices[i]
        g = grads[i]
        t = v['type']
        if t == 'var':
            continue
        if t == 'tnh':
            x = res[v['input']]
            grads[v['input']] += g * (1 - np.tanh(x) ** 2)
        elif t == 'rlu':
            alpha = 1.0 / v['alpha_inv']
            x = res[v['input']]
            gi = g.copy()
            gi[x < 0] *= alpha
            grads[v['input']] += gi
        elif t == 'mul':
            a, b = res[v['a']], res[v['b']]
            grads[v['a']] += g @ b.T
            grads[v['b']] += a.T @ g
        elif t == 'sum':
            for j in v['inputs']:
                grads[j] += g
        elif t == 'had':
            vals = [res[j] for j in v['inputs']]
            for j, idx in enumerate(v['inputs']):
                p = np.ones_like(g)
                for k, val in enumerate(vals):
                    if j != k:
                        p *= val
                grads[idx] += g * p
    return grads


def print_matrix(mat):
    for row in mat:
        print(" ".join(f"{val:.6f}" for val in row))


N, M, K = map(int, sys.stdin.readline().split())
vertices = [parse_vertex(sys.stdin.readline()) for _ in range(N)]
for i in range(N):
    compute_shape(i, vertices)
input_matrices = [read_matrix(vertices[i]['rows']) for i in range(M)]
output_gradients = [read_matrix(vertices[N - K + i]['rows']) for i in range(K)]
res = forw_pass(vertices, input_matrices.copy())
grads = backw_pass(vertices, res, output_gradients)
for i in range(N - K, N):
    print_matrix(res[i])
for i in range(M):
    print_matrix(grads[i])


import sys
import numpy as np

np.set_printoptions(precision=15, suppress=True)

def activation(x, kind):
    if kind == 'sigmoid':
        y = 1 / (1 + np.exp(-x))
        dy = y * (1 - y)
    elif kind == 'tanh':
        y = np.tanh(x)
        dy = 1 - y * y
    else:
        raise ValueError("Unknown activation kind")
    return y, dy

def forward_step(xt, ht_prev, ct_prev, Wf, Uf, Bf,
                 Wi, Ui, Bi, Wo, Uo, Bo, Wc, Uc, Bc):
    zf = Wf @ xt + Uf @ ht_prev + Bf
    ft, dft = activation(zf, 'sigmoid')

    zi = Wi @ xt + Ui @ ht_prev + Bi
    it, dit = activation(zi, 'sigmoid')

    zo = Wo @ xt + Uo @ ht_prev + Bo
    ot, dot = activation(zo, 'sigmoid')

    zc = Wc @ xt + Uc @ ht_prev + Bc
    gt, dgt = activation(zc, 'tanh')

    ct = ft * ct_prev + it * gt
    ht = ot * ct

    return ht, ct, ft, it, ot, gt, dft, dit, dot, dgt

def backward_step(t, dht_next, dct_next, dO_t, f, i_, o, g, c, h, X,
                  dft, dit, dot, dgt,
                  Wf, Uf, Wi, Ui, Wo, Uo, Wc, Uc):
    ft = f[t]
    it = i_[t]
    ot = o[t]
    gt = g[t]
    ct = c[t+1]
    ct_prev = c[t]
    ht_prev = h[t]
    xt = X[t]

    dot_combined = dO_t + dht_next * ct
    dct = dct_next + dht_next * ot

    dot_final = dot_combined * dot[t]
    dft_final = dct * ct_prev * dft[t]
    dit_final = dct * gt * dit[t]
    dgt_final = dct * it * dgt[t]

    dWf = np.outer(dft_final, xt)
    dUf = np.outer(dft_final, ht_prev)
    dBf = dft_final

    dWi = np.outer(dit_final, xt)
    dUi = np.outer(dit_final, ht_prev)
    dBi = dit_final

    dWo = np.outer(dot_final, xt)
    dUo = np.outer(dot_final, ht_prev)
    dBo = dot_final

    dWc = np.outer(dgt_final, xt)
    dUc = np.outer(dgt_final, ht_prev)
    dBc = dgt_final

    dxt = (Wf.T @ dft_final) + (Wi.T @ dit_final) + (Wo.T @ dot_final) + (Wc.T @ dgt_final)
    dht_prev = (Uf.T @ dft_final) + (Ui.T @ dit_final) + (Uo.T @ dot_final) + (Uc.T @ dgt_final)
    dct_prev = dct * ft

    return dxt, dht_prev, dct_prev, dWf, dUf, dBf, dWi, dUi, dBi, dWo, dUo, dBo, dWc, dUc, dBc

def forw_pass(X, h0, c0, Wf, Uf, Bf,
              Wi, Ui, Bi, Wo, Uo, Bo, Wc, Uc, Bc):
    M = len(X)
    h = [h0]
    c = [c0]
    f = []
    i_ = []
    o = []
    g = []
    dft_list = []
    dit_list = []
    dot_list = []
    dgt_list = []

    for t in range(M):
        ht, ct, ft, it, ot, gt, dft, dit, dot, dgt = forward_step(
            X[t], h[t], c[t], Wf, Uf, Bf,
            Wi, Ui, Bi, Wo, Uo, Bo, Wc, Uc, Bc
        )
        h.append(ht)
        c.append(ct)
        f.append(ft)
        i_.append(it)
        o.append(ot)
        g.append(gt)
        dft_list.append(dft)
        dit_list.append(dit)
        dot_list.append(dot)
        dgt_list.append(dgt)
    return h, c, f, i_, o, g, dft_list, dit_list, dot_list, dgt_list

def backw_pass(dhM, dcM, dO, f, i_, o, g, c, h, X,
               dft_list, dit_list, dot_list, dgt_list,
               Wf, Uf, Wi, Ui, Wo, Uo, Wc, Uc):
    N = dhM.shape[0]
    M = len(X)

    dWf = np.zeros((N, N))
    dUf = np.zeros((N, N))
    dBf = np.zeros(N)

    dWi = np.zeros((N, N))
    dUi = np.zeros((N, N))
    dBi = np.zeros(N)

    dWo = np.zeros((N, N))
    dUo = np.zeros((N, N))
    dBo = np.zeros(N)

    dWc = np.zeros((N, N))
    dUc = np.zeros((N, N))
    dBc = np.zeros(N)

    dht_next = dhM.copy()
    dct_next = dcM.copy()

    dX = [np.zeros(N) for _ in range(M)]

    for t in reversed(range(M)):
        dxt, dht_prev, dct_prev, dWf_t, dUf_t, dBf_t, dWi_t, dUi_t, dBi_t, dWo_t, dUo_t, dBo_t, dWc_t, dUc_t, dBc_t = backward_step(
            t, dht_next, dct_next, dO[t], f, i_, o, g, c, h, X,
            dft_list, dit_list, dot_list, dgt_list,
            Wf, Uf, Wi, Ui, Wo, Uo, Wc, Uc
        )

        dWf += dWf_t
        dUf += dUf_t
        dBf += dBf_t

        dWi += dWi_t
        dUi += dUi_t
        dBi += dBi_t

        dWo += dWo_t
        dUo += dUo_t
        dBo += dBo_t

        dWc += dWc_t
        dUc += dUc_t
        dBc += dBc_t

        dX[t] = dxt
        dht_next = dht_prev
        dct_next = dct_prev

    dh0 = dht_next
    dc0 = dct_next

    return dX, dh0, dc0, dWf, dUf, dBf, dWi, dUi, dBi, dWo, dUo, dBo, dWc, dUc, dBc

def read_vector(n):
    return np.array(list(map(float, sys.stdin.readline().split())), dtype=np.float64)

def read_matrix(n):
    mat = []
    for _ in range(n):
        mat.append(read_vector(n))
    return np.array(mat, dtype=np.float64)

def print_vector(v):
    print(' '.join(f'{x:.15g}' for x in v))

def print_matrix(m):
    for row in m:
        print_vector(row)

N = int(sys.stdin.readline())

Wf = read_matrix(N)
Uf = read_matrix(N)
Bf = read_vector(N)

Wi = read_matrix(N)
Ui = read_matrix(N)
Bi = read_vector(N)

Wo = read_matrix(N)
Uo = read_matrix(N)
Bo = read_vector(N)

Wc = read_matrix(N)
Uc = read_matrix(N)
Bc = read_vector(N)

M = int(sys.stdin.readline())

h0 = read_vector(N)
c0 = read_vector(N)

X = [read_vector(N) for _ in range(M)]

dhM = read_vector(N)
dcM = read_vector(N)

dO_rev = [read_vector(N) for _ in range(M)]
dO = dO_rev[::-1]

h, c, f, i_, o, g, dft_list, dit_list, dot_list, dgt_list = forw_pass(
    X, h0, c0, Wf, Uf, Bf,
    Wi, Ui, Bi, Wo, Uo, Bo, Wc, Uc, Bc
)

dX, dh0, dc0, dWf, dUf, dBf, dWi, dUi, dBi, dWo, dUo, dBo, dWc, dUc, dBc = backw_pass(
    dhM, dcM, dO, f, i_, o, g, c, h, X,
    dft_list, dit_list, dot_list, dgt_list,
    Wf, Uf, Wi, Ui, Wo, Uo, Wc, Uc
)


for v in o:
    print_vector(v)

print_vector(h[-1])
print_vector(c[-1])

for v in dX[::-1]:
    print_vector(v)

print_vector(dh0)
print_vector(dc0)

print_matrix(dWf)
print_matrix(dUf)
print_vector(dBf)

print_matrix(dWi)
print_matrix(dUi)
print_vector(dBi)

print_matrix(dWo)
print_matrix(dUo)
print_vector(dBo)

print_matrix(dWc)
print_matrix(dUc)
print_vector(dBc)

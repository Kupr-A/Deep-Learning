import itertools

def get_imp():
    m = int(input())
    out = [int(input()) for _ in range(2 ** m)]
    return m, out

def tabeling(m):
    return list(itertools.product([0, 1], repeat=m))

def neuron(pat):
    w = [1.0 if bit else -1.0 for bit in pat]
    b = -sum(wi for wi in w if wi > 0) + 0.5
    return w, b

def first_layer(table, out):
    l = []
    for pat, val in zip(table, out):
        if val == 1:
            l.append(neuron(pat))
    return l

def sec_layer(n):
    w = [1.0] * n
    b = -0.5
    return [(w, b)]

def print_ans(l1, l2, m):
    if not l1:
        print(1)
        print(1)
        print(" ".join(["0.0"] * m + ["-1.0"]))
        return
    print(2)
    print(len(l1), 1)
    for w, b in l1:
        print(" ".join(f"{x:.1f}" for x in w + [b]))
    for w, b in l2:
        print(" ".join(f"{x:.1f}" for x in w + [b]))

m, outputs = get_imp()
table = tabeling(m)
l1 = first_layer(table, outputs)
l2 = sec_layer(len(l1)) if l1 else []
print_ans(l1, l2, m)

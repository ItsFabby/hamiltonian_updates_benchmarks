import numpy as np
from scipy import linalg, stats


def randomized_rounding(C_til, rho, shots, gauss_list=None, number_batches=1):
    n = C_til.shape[0]

    if gauss_list is None:
        gauss_list = [gen_gauss(n) for _ in range(shots)]

    rho_clean = rho + 10 ** (-10) * np.eye(n)
    rho_sqrt = linalg.sqrtm(rho_clean * n)

    x_list = np.array([np.array([np.sign(np.inner(gauss, rho_sqrt[i, :])) for i in range(n)]) for gauss in gauss_list])
    value_list = np.array([np.inner(x, C_til @ x) / n for x in x_list])

    max_value = np.max(value_list).real
    max_x = x_list[np.argmax(value_list)].astype(int)
    avg_value = np.average(value_list).real
    avg_value_sem = stats.sem(value_list)

    if max_x[0] == -1:
        max_x *= -1

    value_batches = np.array_split(value_list, number_batches)
    max_list = [np.max(batch) for batch in value_batches]

    avg_max_value = np.average(max_list)
    sem_max_value = stats.sem(max_list)

    return {'max_value': max_value, 'max_x': max_x,
            'avg_value': avg_value, 'avg_value_sem': avg_value_sem,
            'avg_max_value': avg_max_value, 'sem_max_value': sem_max_value}


def truncate_matrix(rho, epsilon, n):
    d = np.ones(n) / n - np.diag(rho)
    B = np.argwhere(np.abs(d) > epsilon / n)

    trunc_rho = rho.copy()
    trunc_rho[:, B] = 0
    trunc_rho[B, :] = 0
    np.fill_diagonal(trunc_rho, 1 / n)

    psd_trunc_rho = (trunc_rho + epsilon / n * np.eye(n)) / (1 + epsilon)

    return psd_trunc_rho


def gen_gauss(n):
    gauss = np.random.normal(size=n)
    return gauss / np.linalg.norm(gauss)

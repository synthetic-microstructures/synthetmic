import time
from dataclasses import asdict
from enum import StrEnum, auto

import matplotlib.pyplot as plt
import numpy as np

from examples.utils import (
    get_rcparams,
)
from synthetmic import LaguerreDiagramGenerator
from synthetmic.data.paper import create_example5p1_data

plt.rcParams.update(get_rcparams())


class Phase(StrEnum):
    SINGLE = auto()
    DUAL = auto()


def recreate_figure6(save_path: str, is_periodic: bool) -> None:
    N_GRAINS_SEQ = (1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000)
    R_SEQ = (1, 5)
    N_ITER = 5
    TOL = 1.0

    runtimes = dict(zip((R_SEQ), ([], [])))

    for r in R_SEQ:
        for n_grains in N_GRAINS_SEQ:
            print(
                f"Calculating run times for n_grains={n_grains}, phase={Phase.SINGLE if r == 1 else Phase.DUAL}"
            )
            data = create_example5p1_data(
                n_grains=n_grains, r=r, is_periodic=is_periodic
            )

            ldg = LaguerreDiagramGenerator(
                tol=TOL, n_iter=N_ITER, damp_param=1.0, verbose=False
            )
            start = time.time()
            ldg.fit(**asdict(data))
            end = time.time()
            duration = end - start

            print(
                f"\t* completed in {duration:.4f} seconds \n"
                f"\t* max percentage error = {ldg.max_percentage_error_:.4f}% \n"
                f"\t* mean percentage error = {ldg.mean_percentage_error_:.4f}%\n\n"
            )

            runtimes[r].append(duration)

    # Add O(n) curve to the runtimes
    c_on = np.exp(np.log(runtimes[R_SEQ[0]][0]) - np.log(N_GRAINS_SEQ[0]))
    runtimes["O(n)"] = c_on * np.array(N_GRAINS_SEQ)

    # Add O(n^2) curve to the runtimes
    c_on2 = np.exp(np.log(runtimes[R_SEQ[1]][0]) - 2 * np.log(N_GRAINS_SEQ[0]))
    runtimes["O(n^2)"] = c_on2 * np.array(N_GRAINS_SEQ) ** 2

    styles = ("bs-", "ro-", "k--", "k:")

    _, ax = plt.subplots()

    for i, (k, v) in enumerate(runtimes.items()):
        if k in R_SEQ:
            label = f"${k}$ ({Phase.SINGLE if v == 1 else Phase.DUAL} phase)"
        else:
            label = f"${k}$"
        ax.loglog(N_GRAINS_SEQ, v, styles[i], label=label)

    ax.set_xlabel("$n$")
    ax.set_ylabel("Run time / s")
    ax.legend(frameon=False)

    plt.savefig(save_path, bbox_inches="tight")

    return None


def damp_param_effect(is_periodic: bool, save_path: str) -> None:
    N_GRAINS_SEQ = (1000, 2000, 3000)
    R = 1
    N_ITER = 5
    TOL = 1.0

    damp_params = np.linspace(0.0, 1.0, num=20, endpoint=True)
    linestyles = ("--", "-.", ":")

    _, ax = plt.subplots()

    for i, n_grains in enumerate(N_GRAINS_SEQ):
        print(f"\nRunning analysis with n_grains={n_grains}")

        data = create_example5p1_data(n_grains=n_grains, r=R, is_periodic=is_periodic)
        max_percentage_errors = np.zeros_like(damp_params)

        for j, dp in enumerate(damp_params):
            ldg = LaguerreDiagramGenerator(
                tol=TOL, n_iter=N_ITER, damp_param=dp, verbose=False
            )
            ldg.fit(**asdict(data))

            print(
                f"\t* damp param = {dp:.4f} seconds, max percentage error = {ldg.max_percentage_error_:.4f}%, mean percentage error = {ldg.mean_percentage_error_:.4f}%"
            )

            max_percentage_errors[j] = ldg.max_percentage_error_

        ax.plot(
            damp_params,
            max_percentage_errors,
            label=f"$n = {n_grains}$",
            marker="o",
            markerfacecolor="white",
            linestyle=linestyles[i],
        )

    ax.set_xlabel(r"Damping parameter, $\lambda$")
    ax.set_ylabel(r"Max percentage error (\%)")
    ax.legend(frameon=False)

    plt.savefig(save_path, bbox_inches="tight")

    return None

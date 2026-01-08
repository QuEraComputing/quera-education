import cirq
import numpy as np
import matplotlib.pyplot as plt


class BitFlipOn2QGateNoiseModel(cirq.NoiseModel):
    """
    After every 2-qubit gate, apply an X error (bit flip) on the first qubit
    with probability p, after the gate is applied. All other gates are left noiseless.
    """

    def __init__(self, p: float = 0.02):
        self.p = p

    def noisy_operation(self, operation: cirq.Operation):
        # Check if this is a 2-qubit gate
        if len(operation.qubits) == 2:
            first, second = operation.qubits
            # cirq.bit_flip(p) is an X channel with probability p
            return [
                operation,  # ideal gate
                cirq.bit_flip(self.p).on(first),  # X noise on first qubit
            ]
        return operation


def plot_qec_by_run(
    simple_11: np.ndarray,
    steane_11: np.ndarray,
    y_min: float = 0,
    y_max: float = 1.0,
):
    """
    Line plot of per-run success probability for |11⟩.

    x-axis: run index
    y-axis: success probability
    Two lines: simple vs Steane+QEC

    Parameters
    ----------
    simple_11 : list or np.ndarray
        Probabilities for the simple experiment
    steane_11 : list or np.ndarray
        Probabilities for the Steane+QEC experiment
    y_min, y_max : float
        Y-axis zoom window
    """
    labels = ("2-qubit CX", "Steane CX + QEC")
    runs = np.arange(1, len(simple_11) + 1, dtype=int)

    plt.figure(figsize=(10, 5))

    plt.plot(runs, simple_11, "-o", label=labels[0], color="tab:blue")
    plt.plot(runs, steane_11, "-s", label=labels[1], color="tab:orange")

    plt.xlabel("Run number")
    plt.ylabel("Success probability for |11⟩")
    plt.title("Run-by-run performance: Simple vs Error-Corrected")
    plt.ylim(y_min, y_max)
    plt.xlim(1, len(runs))

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_qec_by_noise(
    simple_11: np.ndarray,
    steane_11: np.ndarray,
    p_range: np.ndarray,
    y_min: float = 0,
    y_max: float = 1.0,
):
    """
    Line plot of success probability for measuring |11⟩ vs
    bit-flip probability in noise model.

    x-axis: bit-flip probability
    y-axis: success probability
    Two lines: simple vs Steane+QEC

    Parameters
    ----------
    simple_11 : np.ndarray
        Probabilities for the simple experiment
    steane_11 : np.ndarray
        Probabilities for the Steane+QEC experiment
    p_range: np.ndarray
        Array of bit-flip probabilities used for the different runs
    y_min, y_max : float
        Y-axis zoom window
    """
    labels = ("2-qubit CX", "Steane CX + QEC")

    plt.figure(figsize=(10, 5))

    plt.plot(p_range, simple_11, "-o", label=labels[0], color="tab:blue")
    plt.plot(p_range, steane_11, "-s", label=labels[1], color="tab:orange")

    plt.xlabel("bit-flip probability")
    plt.ylabel("Success probability for |11⟩")
    plt.title("Bit-flip against success probability: Simple vs Error-Corrected")
    plt.ylim(y_min, y_max)
    plt.xlim(p_range[0], p_range[-1])

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

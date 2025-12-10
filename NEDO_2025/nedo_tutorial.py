# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv (3.13.2)
#     language: python
#     name: python3
# ---

#%% [markdown]
# # Programming Gemini with Bloqade
#
# In this tutorial we will demonstrate how to write circuits and quantum executions with Bloqade, as well as analyize and improve performance of these circuits using noise simulations. There are two main parts to the tutorial. The first part introduces how to write bloqade kernels, as well as simulate them using both a noiseless PyQRack and noisy Cirq backend. This part is a modified version of the "[Circuits with Bloqade](https://bloqade.quera.com/latest/digital/tutorials/circuits_with_bloqade/)" tutorial and the "[GHZ state preparation](https://bloqade.quera.com/latest/digital/examples/interop/noisy_ghz/)" tutorials.
# The second part focuses on circuit parallelism, and how to improve circuit performance by maximizing parallel gate execution. It is based on the "[Parallelism of Static Circuits](https://github.com/QuEraComputing/bloqade/pull/288)" tutorial.

# %%
from typing import Any
import cirq
import matplotlib.pyplot as plt
import numpy as np
from bloqade import squin, cirq_utils
import bloqade.cirq_utils as utils
from cirq.contrib.svg import SVGCircuit

from bloqade.types import MeasurementResult, Qubit

# Some types we will use, useful for type hints
from kirin.dialects.ilist import IList


import warnings
warnings.filterwarnings("ignore")

#%% [markdown]
#Bloqade uses the `squin` dialect set from the compiler toolchain `kirin`. `SQUIN` stands for `S`tructural `Qu`antum `IN`struction set and is the circuit-level representation of quantum executions. It is built on top of the `kirin` framework, an [open-source compiler toolchain](https://queracomputing.github.io/kirin/latest/) for embedded domain-specific languages (eDSLs) that target scientific computing kernels. A key feature of squin is the _kernel_, which can roughly be seen as the object which will be executed on the target hardware. Naturally, this hardware could be a quantum computer, but it also extends to classical execution as well, such as mid-circuit feedforward or even non-quantum execution such as robotics. For more details, please check out the [squin documentation here](https://bloqade.quera.com/latest/digital/).
#
# Kernels can be built using decorators of python functions. We will use the `@squin.kernel` decorator in this notebook but keep in mind that other eDSLs have different decorators inherited from base Kirin decorators. The decorator lowers Python's abstract syntax tree (AST) into a kirin SSA (single static assignment) form, which is a useful intermediate representation for compiler analysis. You don't have to worry too much about SSA or compilers here, but if you want to learn more check out the [kirin documentation](https://queracomputing.github.io/kirin/latest/).
#%%

@squin.kernel
def hello_world(theta: float) -> IList[MeasurementResult, Any]:
    """
    Prepare a Bell state and measure in a basis that might have a Bell violation
    """
    qubits = squin.qalloc(2)
    squin.h(qubits[0])
    squin.cx(qubits[0], qubits[1])
    squin.rx(theta, qubits[0])
    bits = squin.broadcast.measure(qubits)
    return bits

#%% [markdown]
# SQUIN kernels can also let you do complex control flow with mid-circuit measurements and feedforward. Here is an example of a constant-depth GHZ state preparation that uses feedforward. For more details, check out [Efficient Long-Range Entanglement Using Dynamic Circuits](https://doi.org/10.1103/PRXQuantum.5.030339)

# %%
def ghz_constant_depth(n_qubits: int):

    @squin.kernel
    def main()->IList[Qubit, Any]:
        qreg = squin.qalloc(n_qubits)
        ancilla = squin.qalloc(n_qubits - 1)

        for i in range(n_qubits):
            squin.h(qreg[i])

        for i in range(n_qubits - 1):
            squin.cx(qreg[i], ancilla[i])
        for i in range(n_qubits - 1):
            squin.cx(qreg[i + 1], ancilla[i])

        parity: int = 0
        bits = squin.broadcast.measure(ancilla)
        for i in range(n_qubits - 1):
            parity = parity ^ bits[i]
            if parity == 1:
                squin.x(qreg[i + 1])
        return qreg

    return main


# %% [markdown]

# We choose to interop with cirq, so we can convert to and from cirq circuits easily. However, note that due to the flexibility of kernels representing classical control flow, not every squin kernel can be converted to a cirq circuit. In particular, kernels with mid-circuit measurements and feedforward cannot be represented as cirq circuits.

#%%
def bell_prep() -> cirq.Circuit:
    """
    Builder function that returns a simple N-qubit
    GHZ state preparation circuit
    """
    qubits = cirq.LineQubit.range(2)
    output = cirq.Circuit()
    output.append(cirq.H(qubits[0]))
    output.append(cirq.CX(qubits[0], qubits[1]))
    return output

circuit_cirq_bell_prep = bell_prep()
# Load a cirq circuit into squin
kernel = cirq_utils.load_circuit(
    circuit_cirq_bell_prep,
    kernel_name="bell_prep_cirq",  # Define the name of the kernel as if one were using @squin.kernel on a function
    register_as_argument=False,  # If the resulting kernel should take in a qubit register (True) or make a new one (False)
    return_register=True,  # If the resulting kernel should return the register of the qubits it acts on.
)

# Then, we can convert the circuit back to cirq as a roundtrip.
# Note that this is **not possible** in a general case because
# cirq cannot represent complex control flow.
circuit2_cirq_bell_prep: cirq.Circuit = cirq_utils.emit_circuit(kernel, ignore_returns=True)

#%% [markdown]
# # Noiseless simulation of arbitrary kernels with PyQRack
# The first option for simulation is to use the [PyQrack quantum simulator](https://pyqrack.readthedocs.io/en/latest/) backend, which can handle arbitrary feed forward operations. Furthermore, the simulator can switch between different backends, and so is capable of simulating large Clifford circuits with a stabilizer simulator, or small general circuits with a state vector simulator. Lets check out using the PyQrack backend to simulate the `hello_world` kernel defined above.

#%%
from bloqade.pyqrack import StackMemorySimulator
emulator = StackMemorySimulator(min_qubits=8)

#%%
task = emulator.task(hello_world, args=(0.0,))
results = task.run() # Run once
results_batch = task.batch_run(1000) # Run 1000 times
print(results_batch)

#%% And lets do the same for the constant-depth GHZ state
emulator = StackMemorySimulator(min_qubits=7)
task = emulator.task(ghz_constant_depth(3))

state = task.batch_state(shots=1000, qubit_map=lambda x: x)
# Even though there is measurement and feedforward, the final state is still pure. Neat!
print(state.eigenvalues)
print(state.eigenvectors)
# %% [markdown]
# As a final note, consider how difficult it would be to represent this circuit in Cirq. In particular, there is a for loop, where inside the for loop there is an algebraic operation (XOR) that feeds forward onto a variable (parity). This circuit is very hard to express in Cirq without some serious hacking of ancilla registers.


#%% [markdown]
# # Noisy simulation of circuits with Cirq
# The second option for simulation is to use Cirq and noise-annotated circuits. As commented above, not every squin kernel can be converted to a cirq circuit, so this method is limited to kernels without midcircuit measurement and feed forward.

#%%
from bloqade.cirq_utils.noise import (
    transform_circuit,
    GeminiOneZoneNoiseModel,
    GeminiTwoZoneNoiseModel,
)

import qsimcirq
simulator_sv = qsimcirq.QSimSimulator() # Much faster state vector simulator
simulator_dm = cirq.DensityMatrixSimulator()
#%%
circuit_cirq_bell_prep = bell_prep()
model = GeminiOneZoneNoiseModel()

noisy_circuit_cirq_bell_prep = transform_circuit(circuit_cirq_bell_prep, model=model)

# Simulate noisy circuits
clean = simulator_dm.simulate(circuit_cirq_bell_prep).final_density_matrix
noisy = simulator_dm.simulate(noisy_circuit_cirq_bell_prep).final_density_matrix

# Calculate fidelities
fidelity = np.trace(clean @ noisy).real
print("Fidelity of noisy Bell prep circuit:", fidelity)

#%% [markdown]
# # Exploration: Global vs Local gates
# Lets reproduce the insight of comparing global and local gates, as shown in the slides. Lets prepare K states in the |+> state and N-K states in the |0> state, and compare applying global $\sqrt{Y}$ gates and then a local correction, vs. just local $\sqrt{Y}$ gates.

#%%

sqrt_Y = cirq.PhasedXZGate(axis_phase_exponent=0.5,x_exponent=0.5,z_exponent=0)
sqrt_Y_dag = cirq.PhasedXZGate(axis_phase_exponent=0.5,x_exponent=-0.5,z_exponent=0)
identity = cirq.I#cirq.PhasedXZGate(axis_phase_exponent=0.0,x_exponent=0.0,z_exponent=0)

# Care must be taken to line up global and local gates, by putting all gates in a single moment.
Nqubits = 10
qubits = cirq.LineQubit.range(Nqubits)
fidelity_local = []
fidelity_global = []
for k in range(1,Nqubits):
    
    # A small hack is required to bypass gate merging (https://github.com/QuEraComputing/bloqade-circuit/issues/659).
    # Here we split the circuit into two parts to enforce that
    # no gates are accidentally merged.
    circuit_global1 = cirq.Circuit.from_moments(
        cirq.Moment([identity.on(q) for q in qubits]),
        cirq.Moment([sqrt_Y.on(q) for q in qubits]))
    circuit_global2 = cirq.Circuit.from_moments(
        cirq.Moment([identity.on(q) for q in qubits]),
        cirq.Moment([sqrt_Y_dag.on(q) for q in qubits[k::]]))
    noisy_circuit_global1 = transform_circuit(circuit_global1)
    noisy_circuit_global2 = transform_circuit(circuit_global2)
    noisy_circuit_global = noisy_circuit_global1 + noisy_circuit_global2
    circuit_global = circuit_global1 + circuit_global2
    
    # Convert the local circuit to a noisy version.
    circuit_local = cirq.Circuit.from_moments(
        cirq.Moment([identity.on(q) for q in qubits]),
        cirq.Moment([sqrt_Y.on(q) for q in qubits[0:k]]))
    noisy_circuit_local = transform_circuit(circuit_local)
    
    if k==4:
        todraw_1 = circuit_local
        todraw_2 = circuit_global1 + circuit_global2[1::]
    
    # Simulate noisy circuits and compute fidelities
    # Local:
    clean = simulator_dm.simulate(circuit_local).final_density_matrix
    noisy = simulator_dm.simulate(noisy_circuit_local).final_density_matrix
    fidelity_local.append(np.trace(clean @ noisy).real)
    # Global:
    clean = simulator_dm.simulate(circuit_global).final_density_matrix
    noisy = simulator_dm.simulate(noisy_circuit_global).final_density_matrix
    fidelity_global.append(np.trace(clean @ noisy).real)
plt.plot(range(1,Nqubits),fidelity_local,"o-",label='Local')
plt.plot(range(1,Nqubits),fidelity_global,"o-",label='Global')
plt.legend()
plt.xlabel("Number of gates to apply")
plt.ylabel("Fidelity")
plt.axis([1,Nqubits-1,plt.axis()[2],1.0])
plt.show()

#%%
SVGCircuit(todraw_1)
#%%
SVGCircuit(todraw_2)

# %% [markdown]
# # Parallelism of Static Circuits
#
# Now, lets explore more in depth the notion of maximizing parallelism. The following is an edit of a bloqade tutorial "[Parallelism of Static Circuits](https://github.com/QuEraComputing/bloqade/pull/288)".
#
# Parallelism lets gates that act on disjoint qubits execute at the same time, reducing circuit depth and overall runtime. On neutral-atom quantum computers, many transversal operations (same gate type and parameters) can often be executed together in a single layer (moment).
#
# Reducing depth typically improves fidelity and increases the number of operations that can complete within the hardware's coherence time.
#
# Bloqade supports both automatic and manual parallelization. The examples below show both methods and compare fidelity using representative noise models.
#

# %% [markdown]
# ## Example 1: GHZ Circuit
#
# We take the GHZ state preparation as an example. It prepares the state
#
# $\sqrt{2}|\psi\rangle = |000\cdots000\rangle + |111\cdots111\rangle$
#
# The GHZ state can be prepared using a sequence of Hadamard and CNOT gates. In a linear (sequential) implementation, the CNOT gates are applied one after another, resulting in a circuit depth that grows linearly with the number of qubits. In contrast, a log-depth (parallel) implementation arranges the CNOT gates so that multiple gates acting on disjoint qubits can execute simultaneously, reducing the overall depth to logarithmic in the number of qubits. This comes at the cost of requiring arbitrary connectivity, which is not native to all architectures. However, it is perfect for reconfigurable neutral atom systems, which have a native "all to all" connectivity through mid-circuit atom shuttling.

# %%
def build_linear_ghz(n_qubits: int) -> cirq.Circuit:
    """
    Build a linear GHZ circuit using squin and convert to Cirq.
    Inputs:
    n_qubits: Number of qubits in the GHZ state.
    Returns:
    cirq.Circuit: The constructed linear GHZ circuit.
    """

    @squin.kernel
    def linear_ghz_kernel():
        q = squin.qalloc(n_qubits)
        squin.h(q[0])
        for i in range(n_qubits - 1):
            squin.cx(q[i], q[i + 1])

    # Create LineQubits for compatibility with existing code
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq_utils.emit_circuit(linear_ghz_kernel, circuit_qubits=qubits)
    return circuit


def build_log_ghz(n_qubits: int) -> cirq.Circuit:
    """
    Build logarithmic-depth GHZ circuit using squin and convert to Cirq.
    Inputs:
    n_qubits: Number of qubits in the GHZ state.
    Returns:
    cirq.Circuit: The constructed log-depth GHZ circuit.
    """

    max_iterations = int(np.ceil(np.log2(n_qubits))) if n_qubits > 1 else 1

    @squin.kernel
    def log_ghz_kernel():
        q = squin.qalloc(n_qubits)
        squin.h(q[0])

        for level in range(max_iterations):
            width = 2**level
            for i in range(n_qubits):
                if i < width:
                    target = i + width
                    if target < n_qubits:
                        squin.cx(q[i], q[target])

    # Create LineQubits for compatibility with existing code
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq_utils.emit_circuit(log_ghz_kernel, circuit_qubits=qubits)
    return circuit


linear_ghz = build_linear_ghz(12)
log_ghz = build_log_ghz(12)

# %%
SVGCircuit(linear_ghz)

# %%
SVGCircuit(log_ghz)

# %% [markdown]
# ### The benefits of parallelism
# We'll run noise simulations for both circuits and compare their fidelities as we scale the number of qubits.
#
# See our blog post [Simulating noisy circuits for near-term quantum hardware](https://bloqade.quera.com/latest/blog/2025/07/30/simulating-noisy-circuits-for-near-term-quantum-hardware/) for detailed information about the noise model used here. The analysis workflow is:
#
# 1. Build a noiseless (ideal) circuit.
# 2. Choose a noise model (we use the Gemini noise model).
# 3. Apply the noise model to the circuit to produce a noisy circuit.
# 4. Simulate the noisy circuit to obtain the final density matrix.
# 5. Simulate the ideal circuit and compare its state to the noisy density matrix to compute fidelity.
#

# %%

# Initialize noise model (using Gemini one-zone architecture)
noise_model = utils.noise.GeminiOneZoneNoiseModel()
simulator = cirq.DensityMatrixSimulator()


# %% [markdown]
# We run noise-model simulations for circuit sizes from 3 to 9 qubits and compute the fidelity (the higher is better). The ideal noiseless circuit has fidelity 1 by construction.

# %%
# Scan a range of qubit numbers and compute fidelities
fidelities_linear = []
fidelities_log = []
num_qubits = list(range(2, 11))
# Test both linear and log GHZ circuits with noise model
for n in num_qubits:
    # Linear GHZ circuit
    linear_circuit = build_linear_ghz(n)

    # Log GHZ circuit
    log_circuit = build_log_ghz(n)

    # Apply noise model
    linear_noisy_circuit = utils.noise.transform_circuit(
        linear_circuit, model=noise_model
    )
    log_noisy_circuit = utils.noise.transform_circuit(log_circuit, model=noise_model)

    # Simulate noiseless circuits
    rho_linear = simulator.simulate(linear_circuit).final_density_matrix
    rho_log = simulator.simulate(log_circuit).final_density_matrix

    # Simulate noisy circuits
    rho_linear_noisy = simulator.simulate(linear_noisy_circuit).final_density_matrix
    rho_log_noisy = simulator.simulate(log_noisy_circuit).final_density_matrix

    # Calculate fidelities
    fidelity_linear = np.trace(rho_linear @ rho_linear_noisy).real
    fidelity_log = np.trace(rho_log @ rho_log_noisy).real

    # Store results
    fidelities_linear.append(fidelity_linear)
    fidelities_log.append(fidelity_log)

# %%
# Create comparison plot
plt.figure(figsize=(10, 6))

plt.plot(
    num_qubits,
    fidelities_linear,
    "ro-",
    label="Linear GHZ",
    linewidth=2,
    markersize=8,
)
plt.plot(
    num_qubits,
    fidelities_log,
    "bo-",
    label="Log-depth GHZ",
    linewidth=2,
    markersize=8,
)

plt.xlabel("Number of Qubits", fontsize=14)
plt.ylabel("Fidelity", fontsize=14)
plt.title(
    "GHZ State Fidelity Comparison: Linear vs Log-Depth Circuits",
    fontsize=16,
)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(num_qubits)
plt.axis([1.5,10.5,0.6,1.0])
# Add annotations for better understanding
plt.text(
    0.15,
    0.98,
    "Higher fidelity = Better performance",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n=== Summary Statistics ===")
print(
    f"Linear GHZ: Mean = {np.mean(fidelities_linear):.4f}, Std = {np.std(fidelities_linear):.4f}"
)
print(
    f"Log-depth GHZ: Mean = {np.mean(fidelities_log):.4f}, Std = {np.std(fidelities_log):.4f}"
)


# %% [markdown]
# The GHZ results show that parallelizing gates increases fidelity compared with the sequential implementation. The log-depth circuit consistently outperforms the linear-depth circuit, with the advantage growing as we increase the number of qubits. Observe that there is a jump in the fidelity at every power of two, corresponding to the addition of a new level in the log-depth circuit.

# %% [markdown]
# ## Automatic toolkits for circuit parallelization
#
# Bloqade provides automatic tools to compress circuits into more parallel forms:
#
# ```python
# import bloqade.cirq_utils as utils
#
# # Parallelize a circuit
# parallel_circuit = utils.parallelize(circuit)
#
# # Remove internal tags (for cleaner visualization)
# parallel_circuit = utils.remove_tags(parallel_circuit)
# ```
#
# The algorithm builds a DAG of gate dependencies (only commuting gates can be reordered), then solves an integer linear program (ILP) to assign gates to moments while minimizing circuit depth. Similar gates are attracted to the same moment via weighted objectives.
#

# %% [markdown]
# ## Example 2: [7,1,3] Steane code circuit
# Lets explore manual and automatic parallelism optimization on the Steane code, which is a prototypical quantum error correcting code that encodes one logical qubit into seven physical qubits, and can correct a single qubit error.

# We construct several versions of the [7,1,3] Steane code encoder circuit, based on three different initial circuits. The `seq` circuit is designed to be the "worst" possible version of the Steane code, with as much sequential operation as possible. The `11-CNOT` circuit is the textbook version of the Steane code, which uses 11 CNOT gates to perform the encoding. The `9-CZ` circuit is an optimized version that reduces the number of entangling gates to 9 CZ gates by using √Y and √Y† gates instead of Hadamards.
#
# | Version | Description | Parallelization |
# |---------|-------------|-----------------|
# | seq | Sequential circuit using CZ gates (native to neutral atoms) | Manual |
# | seq-auto | Auto-parallelized sequential circuit | Auto |
# | 11-CNOT | Textbook encoder using 11 CNOT gates | Manual |
# | 11-CNOT-auto | Auto-parallelized 11-CNOT circuit | Auto |
# | 9-CZ | Optimized encoder using only 9 CZ gates with √Y gates | Manual |
# | 9-CZ-auto | Auto-parallelized 9-CZ circuit | Auto |
#


# %%
def build_steane_code_circuit():
    """Build the Steane code circuit (version a) using CZ gates - native to neutral atoms, but designed to be as sequential as possible."""

    @squin.kernel
    def steane_kernel():
        q = squin.qalloc(7)

        # H gate on qubits 1, 2, 3
        squin.h(q[1])
        squin.h(q[2])
        squin.h(q[3])

        # Encode the logical qubit with CZ and H gates (equivalent to CNOT)
        squin.h(q[0])
        squin.cz(q[1], q[0])
        squin.cz(q[2], q[0])
        squin.h(q[4])
        squin.cz(q[2], q[4])
        squin.cz(q[6], q[4])
        squin.h(q[5])
        squin.cz(q[6], q[5])
        squin.cz(q[3], q[5])
        squin.cz(q[1], q[5])
        squin.h(q[5])
        squin.h(q[6])
        squin.cz(q[1], q[6])
        squin.cz(q[2], q[6])
        squin.h(q[6])
        squin.cz(q[3], q[4])
        squin.h(q[4])
        squin.cz(q[3], q[0])
        squin.h(q[0])

    # Create LineQubits for compatibility with existing code
    qubits = cirq.LineQubit.range(7)
    circuit = cirq_utils.emit_circuit(steane_kernel, circuit_qubits=qubits)
    return circuit


def build_steane_11cnot()-> cirq.Circuit:
    """Build the Steane code encoder (version b) with 11 CNOT gates - textbook version.

    This is the standard Steane code encoder circuit where:
    - Qubit 6 is the data qubit |ψ⟩ to be encoded
    - Qubits 0-5 are ancillas initialized to |0⟩
    - H gates prepare superposition on control qubits
    - 11 CNOT gates create the encoded state
    """

    @squin.kernel
    def steane_11cnot_kernel():
        q = squin.qalloc(7)

        # H gates on qubits 1, 2, 3 (ancilla preparation)
        squin.h(q[1])
        squin.h(q[2])
        squin.h(q[3])

        # 11 CNOT gates following textbook Steane code structure
        # First layer of CNOTs
        squin.cx(q[6], q[5])
        squin.cx(q[1], q[0])
        squin.cx(q[2], q[4])
        squin.cx(q[2], q[0])
        squin.cx(q[3], q[5])
        squin.cx(q[1], q[5])
        squin.cx(q[6], q[4])
        squin.cx(q[2], q[6])
        squin.cx(q[3], q[4])
        squin.cx(q[3], q[0])
        squin.cx(q[1], q[6])

    qubits = cirq.LineQubit.range(7)
    circuit = cirq_utils.emit_circuit(steane_11cnot_kernel, circuit_qubits=qubits)
    return circuit


def build_steane_9cnot()-> cirq.Circuit:
    """Build the optimized Steane code encoder (version c) with only 9 CNOT gates.

    This optimized version uses √Y and √Y† gates instead of some Hadamards,
    reducing the CNOT count from 11 to 9 while maintaining circuit equivalence.

    The optimization exploits the structure of the Steane code to eliminate
    redundant entangling operations.
    """

    @squin.kernel
    def steane_9cnot_kernel():
        q = squin.qalloc(7)

        # Initial √Y† layer on ancilla qubits (replaces H gates)
        # √Y† = Ry(-π/2)
        squin.ry(-np.pi / 2, q[0])
        squin.ry(-np.pi / 2, q[1])
        squin.ry(-np.pi / 2, q[2])
        squin.ry(-np.pi / 2, q[3])
        squin.ry(-np.pi / 2, q[4])
        squin.ry(-np.pi / 2, q[5])

        # First CZ layer (parallel)
        squin.cz(q[1], q[2])
        squin.cz(q[3], q[4])
        squin.cz(q[5], q[6])

        # √Y layer
        squin.ry(np.pi / 2, q[6])

        # Second CZ layer (parallel)
        squin.cz(q[0], q[3])
        squin.cz(q[2], q[5])
        squin.cz(q[4], q[6])

        # √Y layer from 2 to 6
        squin.ry(np.pi / 2, q[2])
        squin.ry(np.pi / 2, q[3])
        squin.ry(np.pi / 2, q[4])
        squin.ry(np.pi / 2, q[5])
        squin.ry(np.pi / 2, q[6])

        # Third CZ layer (parallel)
        squin.cz(q[0], q[1])
        squin.cz(q[2], q[3])
        squin.cz(q[4], q[5])

        # Final √Y layer
        squin.ry(np.pi / 2, q[1])
        squin.ry(np.pi / 2, q[2])
        squin.ry(np.pi / 2, q[4])

    qubits = cirq.LineQubit.range(7)
    circuit = cirq_utils.emit_circuit(steane_9cnot_kernel, circuit_qubits=qubits)
    return circuit


# %%
# Build all Steane circuit versions (reuse already defined noise models and simulator)
steane_seq = build_steane_code_circuit()  # Sequential CZ-based
steane_seq_auto = utils.parallelize(circuit=steane_seq)  # Auto-parallelized
steane_seq_auto = utils.remove_tags(steane_seq_auto)

steane_11cnot = build_steane_11cnot()  # 11 CNOT textbook
steane_11cnot_auto = utils.parallelize(circuit=steane_11cnot)  # Auto-parallelized
steane_11cnot_auto = utils.remove_tags(steane_11cnot_auto)

steane_9cz = build_steane_9cnot()  # 9 CZ optimized
steane_9cz_auto = utils.parallelize(circuit=steane_9cz)  # Auto-parallelized
steane_9cz_auto = utils.remove_tags(steane_9cz_auto)

# %% [markdown]
# ### seq: Sequential CZ-based Steane Circuit

# %%
SVGCircuit(steane_seq)

# %% [markdown]
# ### seq-auto: Auto-Parallelized Sequential Circuit

# %%
SVGCircuit(steane_seq_auto)

# %% [markdown]
# ### 11-CNOT: Textbook Steane Encoder

# %%
SVGCircuit(steane_11cnot)

# %% [markdown]
# ### 11-CNOT-auto: Auto-Parallelized 11-CNOT Circuit

# %%
SVGCircuit(steane_11cnot_auto)

# %% [markdown]
# ### 9-CZ: Optimized Steane Encoder

# %%
SVGCircuit(steane_9cz)

# %% [markdown]
# ### 9-CZ-auto: Auto-Parallelized 9-CZ Circuit

# %%
SVGCircuit(steane_9cz_auto)

# %% [markdown]
# ### Circuit Depths
# A lower depth is heuristically better than higher depth due to spectator errors on idle qubits.

# %%
print(f"seq:          {len(steane_seq)} moments")
print(f"seq-auto:     {len(steane_seq_auto)} moments")
print(f"11-CNOT:      {len(steane_11cnot)} moments")
print(f"11-CNOT-auto: {len(steane_11cnot_auto)} moments")
print(f"9-CZ:         {len(steane_9cz)} moments")
print(f"9-CZ-auto:    {len(steane_9cz_auto)} moments")


# %% [markdown]
# ### Noise Analysis

# %%
# Compute fidelities for all circuit versions
steane_circuits = {
    "seq": steane_seq,
    "seq-auto": steane_seq_auto,
    "11-CNOT": steane_11cnot,
    "11-CNOT-auto": steane_11cnot_auto,
    "9-CZ": steane_9cz,
    "9-CZ-auto": steane_9cz_auto,
}

steane_fidelities = {}
for name, circuit in steane_circuits.items():
    noisy = utils.noise.transform_circuit(circuit, model=noise_model)
    rho_ideal = simulator.simulate(circuit).final_density_matrix
    rho_noisy = simulator.simulate(noisy).final_density_matrix
    steane_fidelities[name] = np.trace(rho_ideal @ rho_noisy).real

# Print summary
print(f"{'Version':<15} {'Depth':<10} {'Fidelity':<10}")
print("-" * 35)
for name, circuit in steane_circuits.items():
    print(f"{name:<15} {len(circuit):<10} {steane_fidelities[name]:.4f}")

best_version = max(steane_fidelities, key=steane_fidelities.get)
print(f"\nBest: {best_version} ({steane_fidelities[best_version]:.4f})")

# %% [markdown]
# Fidelity comparison plot for all Steane code versions:

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

labels = ["seq", "seq\nauto", "11-CNOT", "11-CNOT\nauto", "9-CZ", "9-CZ\nauto"]
fidelity_vals = list(steane_fidelities.values())
depth_vals = [len(c) for c in steane_circuits.values()]
colors = ["#c0392b", "#e74c3c", "#d68910", "#f4d03f", "#1e8449", "#58d68d"]

for ax, vals, ylabel, title in [
    (ax1, fidelity_vals, "Fidelity", "Steane Code: Fidelity"),
    (ax2, depth_vals, "Circuit Depth", "Steane Code: Depth"),
]:
    bars = ax.bar(labels, vals, color=colors, edgecolor="black")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, vals):
        fmt = f"{v:.3f}" if isinstance(v, float) else str(v)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            fmt,
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
ax1.set_ylim(0, 1)
plt.tight_layout()
plt.show()

# %% [markdown]
# As expected, the manual and optimized circuits do better than their naively optimized counterparts. The "worst case" sequential circuit has the lowest fidelity, while the auto-optimized 9 CZ circuit has the highest fidelity. However, this also comes with a point of warning: the noise model is not a perfect representation of real hardware. In practice, the hand optimized 9-CZ circuit was implemented as part of QuEra's [magic state distillation paper](https://arxiv.org/abs/2412.15165), which suggests that the noise model is not aligned with hardware. The next steps after manual and automatic optimization should be implementation and tuning on real hardware.


# %%

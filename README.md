# QuEra Educational Materials

Welcome to QuEra's course on quantum computing and neutral-atom technologies! Following this material you will be brought from 0 to 1 (or ground to Rydberg!) on concepts and programming schemes for doing experiments and simulations with our quantum tech and beyond. Stay tuned for updates, exercises, homework, and future challenges!

## Launch Options

There are two possible ways to interact with the examples here, either via qBraid, a cloud-based environment for accessing different quantum computing platforms, or locally on your own machine.

### Local Usage

To run the tutorials in this repo you'll first need to install the Julia programming language on your system.

You can do this in a pinch with [`juliaup`](https://github.com/JuliaLang/juliaup) which allows you to easily maintain multiple versions of Julia (should you need it) as well as keeping Julia up-to-date.

Then you'll need to ensure that you have Jupyter Notebook/Jupyter Lab already installed on your computer. Jupyter Lab can be installed via `conda` or `pip` (see [here](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html)) but you can also use notebook: (https://jupyter.org/install#jupyter-notebook).

Finally, you'll need to install the packages the tutorials rely on. Type `julia` into your terminal and you should be greeted by:

```
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.9.1 (2023-06-07)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia>
```
Now copy and paste the following into the prompt:
```julia
using Pkg, Pkg.add(["Bloqade", "BloqadeExpr", "BloqadeSchema", "Graphs", "PythonCall", "IJulia", "JSON3"])
```
Once the installation is complete you can exit Julia and launch jupyter lab/jupyter notebook from your terminal (ensuring you're launching it from the folder where all these tutorials are). The `IJulia` package you installed earlier should have installed a jupyter kernel that enables Julia code to be executed inside Jupyter notebooks. 

Select the "Julia" kernel and have fun tinkering with the examples!

### Usage on qBraid

You can also launch this entire repo on qBraid by clicking on the button below:

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150">](https://account.qbraid.com?gitHubUrl=https://github.com/QuEraComputing/quera-education.git)

**NOTE:** Access to Bloqade on qBraid is still in development and requires a specific access code. Contact QuEra if you are interested in support for this access at info@quera.com.

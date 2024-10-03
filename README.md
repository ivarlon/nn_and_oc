# Neural Operator-based Optimal Control
This repo contains the code that was used to investigate a method of optimal control where the state and adjoint equations are solved by surrogate neural network models. This method was tested on three examples: a linear, scalar-valued ODE; the 1D heat equation; and the 2D Poisson equation. The code is heavily based on PyTorch, as well as NumPy and SciPy.

The models we trained and used are avaiable upon request.

**How to run**:
- Run `python run.py {file you want to run}`, e.g. `python run.py heat_equation/training_FNO_models.py`
- Alternatively, we recommend most of the code to be run in an interactive environment, such as Jupyter or Spyder. This was the way we wrote and ran most of the code, except for the code that trains the neural operators.

The repo is organised as follows.

## Root folder
The root folder contains the following files:
##### `FNO.py`
Define the class FNO (Fourier neural operator)

##### `DON.py`
Defines the DeepONet (Deep operator network) class 

##### `CustomDataset.py`
Defines custom dataset class for loading data.

##### `run.py`
Sets the correct work environment so that imports in other files are handled correctly.

There are also four sub-folders that contain the remaining code.

## simple_scalar_problem
This pertains to the scalar ODE that describes exponential decay with a source term:
$$
y' = -y + u
$$
The folder contains the following code:

##### `training_{}_models.py`
Sets up and trains FNO and DeepONet models to solve the state equation and the adjoint equation. The models are trained sequentially and saved as a list in pickle serialised format.

Additionally: 
##### `generate_data.py`
Contains functions to generate data $y,u$ and $p,y$ for training, based on an implementation of the improved Forward Euler method.

##### `OC_simple_scalar_problem.py`
Performs the OC optimisation loop. Includes a class that contains the necessary functions involved in the optimisation, including the cost function as methods to calculate state and adjoint variables. The trained neural operators are loaded as a list from a pickle file.

##### `bootstrap.py`
Performs a subsample bootstrap of OC runs based on drawing an $n$-sized ensemble from a list of neural operator models. It does this for a range of $n$-values.

##### `plotting_and_metrics.py`
Loads pickled lists of training data and plots the loss history.

##### (other files)
Deprecated but didn't want to remove from repo because they're my children and I love them.

## heat_equation
This pertains to the 1D heat equation:
$$
\frac{\partial y}{\partial t} = D\frac{\partial^2 y}{\partial x^2} + u
$$
The folder contains the following code

##### `training_{}_models.py`
Sets up and trains FNO and DeepONet models to solve the state equation and the adjoint equation. The models are trained sequentially and saved as a list in pickle serialised format.

Additionally: 
##### `crank_nicolson.py`
Solves the heat equation using the implicit Crank-Nicolson scheme. A test of the method is included.

##### `generate_data_heat_eq.py`
Contains functions to generate data $y,u$ and $p,y$ for training, based on solving the heat equation with Crank-Nicolson.

##### `OC_heat_eq.py`
Performs the OC optimisation loop. Includes a class that contains the necessary functions involved in the optimisation, including the cost function as methods to calculate state and adjoint variables. The trained neural operators are loaded as a list from a pickle file.

##### `plotting_and_metrics.py`
Loads pickled lists of training data and plots the loss history.


## poisson
This pertains to the 2D Poisson equation:
$$
\frac{\partial^2 y}{\partial x_1^2} + \frac{\partial^2 y}{\partial x_2^2} = - u
$$
The folder contains the following code:

##### `training_{}_models.py`
Sets up and trains FNO and DeepONet models to solve the state equation, and the state models may be used to solve the adjoint equation (which is identical to the state equation in our case). The models are trained sequentially and saved as a list in pickle serialised format.

Additionally: 
##### `solve_poisson.py`
Solves the Poisson equation with a direct scheme of inverting a matrix. A test of the method is included.

##### `generate_data_heat_eq.py`
Contains functions to generate data $y,u$ and $p,y$ for training, based on solving the Poisson equation. The source functions $u$ that are generated are Gaussian mixtures.

##### `OC_poisson.py`
Performs the OC optimisation loop. Includes a class that contains the necessary functions involved in the optimisation, including the cost function as methods to calculate state and adjoint variables. The trained neural operators are loaded as a list from a pickle file.

##### `plotting_and_metrics.py`
Loads pickled lists of training data and plots the loss history.

## utils
This defines a lot of general functions used in other files, namely:

##### `training_routines.py`
This contains functions to train the FNO and DeepONet models.

##### `training_routines_amp.py`
Like `training_routines.py` but uses mixed precision (32 and 16 bit) for reduced memory overhead. Used in the file `poisson/training_DON_models.py`.

##### `optimization_routines.py`
Defines gradient descent and conjugate gradient methods, as well as an Armijo line search.

##### `bootstrap_OC.py`
Defines a function to perform a bootstrap of OC runs by drawing $n$ models (with replacement) from a pickled list of neural operator models.


### Note for transparency
This repo contains the code that was used in our thesis. The repo has been updated after the thesis deadline with the code that we used, because the repo was out of date: among other things, for the final thesis we had changed some boundary conditions for the equations, and fixed a sign mistake in the Poisson DeepONet physics loss. To properly understand and reproduce the results in our thesis, the most recent commit should be used.

We weren't sure about whether updating the repo was a legitimate move, based on ambiguities in the guidelines surrounding the thesis. After two days of deliberation after the thesis deadline, we decided on updating the repo. The state of the repo at the time of deadline (Oct. 1. 2024, 14.00) can be found by pulling the relevant git commit id. After cloning the repo, use `git log` to find the ID of the desired repo commit, and then use `git reset --hard {commit ID}`. In this way, it is up to the individual to decide which version of the repo he or she wants to look at. 

However, we think it is most legitimate to just use the most recent commit. The objection to doing so is that the master's thesis had a deadline that is passed, and so no changes can be made. This objection hinges on an interpretation of what the term "master's thesis" refers to.

The broad interpretation is that "master's thesis" refers to the code used as much as the written paper. Therefore, any code must be handed in before the prescribed deadline just like the written report must be.

The narrow interpretation is that "master's thesis" primarily refers to the written report, and that any association with the code is secondary, namely, the results in the written report need to be reproducible, which implies the availability of the code that was used. This interpretation doesn't require code to be handed in in some prescribed format.

The problem with the broad interpretation is that it is unnatural to use the term "thesis" for anything but the written report. It doesn't make sense to say "I'm writing my thesis, but I haven't started writing the report yet". It makes more sense to say "I'm not writing my thesis just yet, I'm working on my code", which implies that the code is not part of the thesis per se.

An additional problem with the broad interpretation is that the regulations surrounding the master's thesis seem to treat it as a written report. The regulations say where and how the thesis is to be submitted: online on webpage P, by time T, in PDF format. This makes no mention of code and is most natural to think of as the written report. There is no clause that says "code has to be published on Github by time T and any subsequent updates of the repo are deemed illegitimate". There are courses where the code that was used in obligatory assignments must be published on Github by the assignment deadline, but then this is clearly specified in the assignment description or elsewhere.

For these reasons we prefer the narrower, more natural interpretation and consider it permissible to update the repo to make available the code that we used.

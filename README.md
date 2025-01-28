# py-data-models

This repository contains a collection of data models with simple and consistent implementations, ideal for lightweight and Python-based data modelling.

## Installation

This repository was primarily written in 3.11.3; compatibility with 3.x versions can be assumed.
This repository has the following dependencies:
- math (standard Python library)
- numpy (`pip install numpy` to install)

All models also require the `base_model.py` to inherit the base `Model` class, and some models additionally require the `matrix.py` file to be accessible.

## Usage

To use a model, import the model file. Initialise a model by creating an object using the desired model class.`sirMoel = SIR()`

1. Import model class from file (e.g. `from sir import SIR`)
2. Initialise model by instantiating an object with the desired model class (e.g. `sirModel = SIR()`)
3. Either:
   a. Externally set the model parameters (e.g. `sirModel.infRate = 0.5`)
   b. Execute model training with a set of inputs and desired outputs (e.g. `sirModel.train(inputs, targets)`)
4. Execute model prediction for a given input (e.g. `sirModel.predict(100)`)

### Full example

```
from sir import SIR # Import model class

sirModel = SIR(0.25, 0.1) # Initialise model and set parameters
print(sirModel.predict(25)) # Output model prediction for t = 25

targetModel = SIR(0.3, 0.7) # Create target model to replicate
inputs = [x for x in range(100)] # Create inputs for target model and training
targets = [targetModel.predict(x) for x in inputs] # Get target outputs from target model
sirModel.train(inputs, targets) # Train model with target outputs
print(sirModel.predict(50)) # Output model prediction for t = 50
```

Expected output:

```
[0.21004359203895354, 0.22565797639309043, 0.564298431567956] # t = 25, infRate = 0.25, recRate = 0.1
[0.8386886949115978, 1.8308298239615008e-14, 0.16131130508838373] # t = 50, infRate = ~0.3, recRate = ~0.7
```

> Additional input may be created with importing a model file; this is model testing output and can be ignored.

## `Model` structure

Every model inheriting the base `Model` class (`base_model.py`) implements a variation of these methods:
- `___init__`: A constructor that can take a series of parameters to define a specific model
- `train`: A method that takes a series of inputs and outputs and finds (near-)optimal parameters to model the given data
- `predict`: A method that takes an input and produces an output as predicted by the current parameters of the model or returns None if a parameter has not been set
## Model types
### Linear regression (`LinearRegression`)
The linear regression model is used to predict a single output variable using one or more input values with linear relationships to the output.
#### Model application (`LinearRegression.predict(x)`)

$y = \textbf{x}^T\textbf{w}$

- $y$: model output
- $\textbf{x}$: model inputs vector
- $\textbf{w}$: model weights vector

Each weight in the weights vector corresponds to and scales an input in the inputs vector, allowing for the implementation of linear relationships.
#### Model training (`LinearRegression.train(X, y)`)

$\textbf{w} = (\textbf{X}^T\textbf{X})^{-1}\textbf{X}^T\textbf{y}$

- $\textbf{w}$: model weights vector
- $\textbf{X}$: provided input matrix
- $\textbf{y}$: target output vector

The matrix-based structure of the model allows for the model weights to be calculated exactly from the input matrix and target output vector.

### Exponential regression (`ExponentialRegression`)

The exponential regression model is used to model a relationship between a single input and single output value, where the rate of increase of the output is proportional to the value of the input.

#### Model application (`ExponentialRegression.predict(x)`)

$y = ab^x$

- $y$: model output
- $x$: model input
- $a, b$: model parameters

As the input value increases, the rate of change of the output increases, producing an exponential relationship.

#### Model training (`ExponentialRegression.train(x, y, iterationLimit)`)

##### Parameter initialisation

$a = y_0$

$b = (\frac{y_0}{x_0})^{x_0^{-1}}, x_0 \neq 0; (\frac{y_1}{x_1})^{x_1^{-1}}, otherwise$

- $y$: model output
- $x$: model input
- $a, b$: model parameters

The first point with a non-zero zero input is used to calculate the initial parameters. The first input is assumed as 0 to generate the initial value for a, which is unchanged by later training.

##### Parameter optimisation

Since the value for the parameter $a$ is assumed to be correct, only the parameter $b$ is optimised by training.
During each epoch, $b$ is adjusted to minimise the model loss (total MSE across training data) using the Newton-Raphson method, as following:

$loss = \sum_{i=0}^{|\textbf{x}|}(ab^{\textbf{x}_i} - y_i)^2$

$b_n = b_{n-1} - \frac{1}{2a}loss\cdot(\sum_{i=0}^{|\textbf{x}|}(\textbf{x}_ib^{\textbf{x}_i - 1}(ab^{\textbf{x}_i} - \textbf{y}_i)))^{-1}$

- $\textbf{y}$: model outputs
- $\textbf{x}$: model inputs
- $a, b$: model parameters

After each adjustment, if the loss of the overall model has been reduced, the parameter is value is preserved. If the loss stagnates or the iteration limit `iterationLimit` is reached, training exits. After training is complete, the last preserved parameter value is used to minimise loss.

#### Naive model training (`ExponentialRegression.train_naive(x, y, initialPrecision, finalPrecision)`)

##### Parameter initialisation

$a = y_0$

$b = (\frac{y_0}{x_0})^{x_0^{-1}}, x_0 \neq 0; (\frac{y_1}{x_1})^{x_1^{-1}}, otherwise$

- $y$: model output
- $x$: model input
- $a, b$: model parameters

The same parameter initialisation is used as Newton-Raphson training, where the first output value is assumed as the target output for when the model input is 0.

##### Parameter optimisation

Since the parameter $a$ is assumed to be the correct value, only the parameter $b$ is varied. The parameter $b$ is varied incrementally, with the increment starting with the value `10^initialPrecision`. The increment is added to $b$ until the model loss increases, when the increment is reversed. Once this pass is completed for a increment, the increment is reduced and another pass is completed. Once the pass for the increment `10^finalPrecision` is complete, training is complete.

### Logarithmic regression (`LogarithmicRegression`)

#### Model application (`LogarithmicRegression.predict(x)`)

$y = a + b\ln{x}$

- $y$: model output
- $x$: model input
- $a, b$: model parameters

As the input value increases, the rate of change of the output decreases, producing a logarithmic relationship.

#### Model training (`LogarithmicRegression.train(x, y)`)

##### Parameter initialisation

$a = y_0$

$b = \frac{y_1 - a}{\log{x_1}}$

- $y$: model output
- $x$: model input
- $a, b$: model parameters

The first provided output is assumed to correspond to an input of 1, so the output is equal to the value for $a$. The second provided output and estimated value for $a$ are then used to estimate the value for $b$.

##### Parameter optimisation

Both $a$ and $b$ are adjusted during training using the Newton-Raphson method to minimise the loss (total MSE) of the model, as following:

$loss = \sum_{i=0}^{|\textbf{x}|}(ab^{\textbf{x}_i} - y_i)^2$

$a_n = a_{n-1} - \frac{1}{2}loss\cdot(\sum_{i=0}^{|\textbf{x}|}((a + b\ln{\textbf{x}_i}) - \textbf{y}_i))^{-1}$

$b_n = b_{n-1} - \frac{1}{2}loss\cdot(\sum_{i=0}^{|\textbf{x}|}((\ln{x_i}(a + b\ln{\textbf{x}_i}) - \textbf{y}_i)))^{-1}$

- $\textbf{y}$: model outputs
- $\textbf{x}$: model inputs
- $a, b$: model parameters

After each adjustment, if the loss of the overall model has been reduced, the parameter is value is preserved. If the loss stagnates or the iteration limit `iterationLimit` is reached, training exits. After training is complete, the last preserved parameter value is used to minimise loss.

### Logistic regression (`LogisticRegression`)

#### Model application (`LogisticRegression.predict(x)`)

$y = \frac{m}{1 + e^{a + bx}}$

- $y$: model output
- $x$: model input
- $a, b$: model parameters

The model output is within the range $(0,1)$, with large positive or negative inputs tending towards either 0 or 1.

#### Model training (`LogisticRegression.train(x, y)`)

##### Parameter initialisation

$a = -1.1$

$b = -1.1$

- $a, b$: model parameters

Both parameters are initialised to $-1.1$, which is an arbitrary constant that allows for training to optimise the values. While this will initially produce a considerably sub-optimal model, after a large number of training epochs, the impact of this arbitrary initialisation is minimal.

##### Parameter optimisation

The Newton-Raphson method is used to vary both parameters to minimuse the loss (total MSE) of the model:

$loss = \sum_{i=0}^{|\textbf{x}|}(ab^{\textbf{x}_i} - y_i)^2$

$m_n = m_{n-1} - \frac{1}{2}loss\cdot(\sum_{i=0}^{|\textbf{x}|}(e^{-a-b\textbf{x}_i}(m(1+e^{a+b\textbf{x}_i})^{-1} - \textbf{y}_i)))^{-1}$

$a_n = a_{n-1} + \frac{1}{2me^a}loss\cdot(\sum_{i=0}^{|\textbf{x}|}(e^{b\textbf{x}_i}(1+e^{a+b\textbf{x}_i})^{-2}(m(1+e^{a+b\textbf{x}_i})^{-1} - \textbf{y}_i)))^{-1}$

$b_n = b_{n-1} + \frac{1}{2me^a}loss\cdot(\sum_{i=0}^{|\textbf{x}|}(\textbf{x}_ie^{b\textbf{x}_i}(1+e^{a+b\textbf{x}_i})^{-2}(m(1+e^{a+b\textbf{x}_i})^{-1} - \textbf{y}_i)))^{-1}$

- $\textbf{y}$: model outputs
- $\textbf{x}$: model inputs
- $a, b$: model parameters

After each adjustment, if the loss of the overall model has been reduced, the parameter is value is preserved. If the loss stagnates or the iteration limit `iterationLimit` is reached, training exits. After training is complete, the last preserved parameter value is used to minimise loss.

### SIR (Susceptible-Infected-Recovered)

#### Model application (`SIR.predict(x)`)

$\frac{dS}{dt} = -\beta S I$

After each timestep, the susceptible population decreases by a change proportional to the infection rate $\beta$, the current susceptible population and the infected population. Generally, as $t \to \infty$, $S \to 0$, as all susceptible individuals become infected.

$\frac{dI}{dt} = \beta S I - \gamma I$

$\beta S I$ represents the susceptible individuals that are infected during the current timestep, and $\gamma I$ represents the infected individuals who recover during each timestep. As $t \to \infty$, $I \to 0$; however, depending on the values of $\beta$ and $\gamma$ used, $I$ may reach a maximum value, after which the rate of recovery becomes greater than the rate of new infections, or it may continuously decrease to $0$.

$\frac{dR}{dt} = \gamma I$

The rate of increase in the recovered population is equal to the decrease in the infected population during each timestep, as individuals that recover from the infected state enter the recovered state. With many combinations of parameters, as $t \to \infty$, $R \to 0$, as all susceptible individuals become infected and subsequently recover; otherwise, if $S \to S_{\infty}$, $R \to 1 - S_{\infty}$.

> ##### Note on model usage
> To improve the efficiency of repeated predictions, any calculated values are saved to `self.savedResults`. When changing any model parameters, this must be reset to an empty list (`self.savedResults = []`) to prevent fatal errors and incorrect further predictions.

#### Model training

##### Parameter initialisation

$\beta = 0.1$

$\gamma = 0.1$

- $\beta, \gamma$: model parameters

An arbitrary constant is used to initialise both $\beta$ (infection rate) and $\gamma$ (recovery rate), as the effect of any initialisaton on training accuracy quickly becomes negligible.

##### Standard gradient descent (`SIR.train(x, y, iterationLimit, initialInf, initialRec)`)

Gradient descent is used to vary both parameters with the following iterative formulae:

$\beta_{n+1} = \beta_{n} - \sum_{i=0}^{|x|}((y_i)_S \cdot (y_i)_I \cdot ((y_i)_S - p(x_i)_S + (y_i)_I - p(x_i)_I))$

$\gamma_{n+1} = \gamma_{n} - \sum_{i=0}^{|x|}((y_i)_I \cdot (y_i)_R \cdot ((y_i)_I - p(x_i)_I + (y_i)_R - p(x_i)_R))$

- $y_i$: target output at position $i$
- $x_i$: input at position $i$
- $p(x)$: prediction with input $x$ and current values for $\beta$ and $\gamma$
- $A_S$, $A_I$, $A_R$: $S$, $I$ and $R$ values from $A$
- $\beta$: infection rate
- $\gamma$: recovery rate

##### Gradient descent with loss checks (`SIR.train_(x, y, iterationLimit, intialInf, initialRec)`)

This uses the same iteration as the above gradient descent; however, after each change, the current model MSE loss is calculated. The parameter values with the lowest loss are then used as the final values, ensuring that, if training begins to diverge, the best known parameters are used.

> This training algorithm produces considerably less optimal results than `SIR.train()` due to internal calculation differences. Since the gradient descent used is unlikely to diverge, the functionality provided by `SIR.train_()` is generally unnecessary and provides no benefit to training accuracy.

##### Naive parameter variation (`SIR.train_naive(x, y, initialInf, initialRec, initialPrecision, finalPrecision)`)

The model parameters $\beta$ and $\gamma$ are changed by decreasing powers of 10, with the variation starting at `10**initialPrecision` and ending after  the variation reaches `10**finalPrecision`. Initially, a value of $\beta$ is set with the current variation; the optimal value for $\gamma$ is then found through a similar process. The combination of values for $\beta$ and $\gamma$ that produce the lowest model MSE loss are then used once training is complete.

> This algorithm is considerably less efficient than `SIR.train()`; however, if other algorithms fail to converge, this algorithm can be more reliable due to its complete exploration of possible values.
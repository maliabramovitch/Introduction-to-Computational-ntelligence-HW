# Mixed-Integer Quadratic Objective Functions Optimized by a (1+1) Evolution Strategy
This project provides basic mixed-integer quadratic objective functions for testing black-box optimization heuristics, as well as execution code for applying the (1+1) Evolution Strategy to a concrete setup of 3 problem instances in dimensionality D=64.

## Source files:
1. MixedVariableObjectiveFunctions.py - constitutes the Objective Function Classes module, whose primary class is MixedVarsEllipsoid that represents an unconstrained mixed-integer ellipsoid function. Its function evaluation is executed per a population of candidate solutions which is stored in a 2D numpy array X. 
2. ellipsoidFunctions.py - comprises a set of Hessian generating functions to construct a family of ellipsoid objective functions.
3. 1p1_ES_with_ObjectiveFunctionClass.py - includes an implementation of the renowned (1+1)-Evolution Strategy with the 1/5th success-rule, as well as a __main__ function which applies the (1+1)-ES to 3 instances of the mixed-integer quadratic function "MixedVarsEllipsoid", whose Hessian matrices are generated via "ellipsoidFunctions": the 'Cigar', 'RotateEllipse', and the 'HadamardEllipse' instances.
The ES does not handle the integer constraint in a particular manner, but lets the objective function evaluation round the values to the nearest integer. The experimental setup runs the ES NRUNS=30 times on each of the 3 problem instances.

## Concrete details:
Within the OnePlusOneEvolutionStrategy, the objective function calls use numpy's "reshape" since it deals with a singleton decision vector while the Objective Function assumes a population in a 2D numpy array as its input. 

## Usage on bash:
python3 1p1_ES_with_ObjectiveFunctionClass.py 

## Contact
Ofer Shir: ofersh@telhai.ac.il

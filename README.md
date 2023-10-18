# Minion

[Contact me](bowen.zhang23@outlook.com)

## Introduction

Minion is a extremely easy python package that performs several popular optimisation algorithms for 
finding the local minina of a given multi-dimensional objective function, 
including:

- Newton
- Gradient Descent
- Quasi-Newton (BFGS)
- Nelder-Mead.

The implementations almost strictly follow the descriptions in the corresponding wikipedia pages.
Hope they help beginners to understand what is under the hood of existing optimisation packages, i.e. 
gnu, scipy, matlab, etc.

## Usage

See jupyter notebooks.

## See also

[Wikipedia: mathematical optimization](https://en.wikipedia.org/wiki/Mathematical_optimization).

## Todo

- Auto beta for quasi-newton, refer to scipy.
- Test all methods with more functions
- Evaluate the speed of each methods.

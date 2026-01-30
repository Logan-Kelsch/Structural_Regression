# Structural_Regression

### Draft 1 of project structure refresh 1/29/2026

In this project, we will be taking in data points, 
time-series or population sample,
most often being time-series as this is the fronteir I would like to challenge.

Instead of simply using symbolic regression to eventually model this data, I want to 
iteratively solve for a structure or "grammar" that models this type of data at a maximal pace.

This requires two iterative solving loops, therefore data must contain a hiarchecal structure for where it is used.
In stock market data:
- Since we cannot differentiate the loss surface with this approach, we can iterate towards a solution given a grammar
  for each time window, evaluate on the next window, then iterate the grammar based on out of sample performance.
  Entirely independent testing exists here on either a different window of time WITH/WITHOUT a different symbol.
In population based data:
- Random partitioning of the population in several segments to allow for several fair iterations of grammar before one
  large developed grammar modeling iteration set, with another partition left out for a final validation.

We will use MEP data structure for instantiated data.
We will use standard GP mutation and reproduction strategies.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
We will have to define several candidate grammar structures of various complexities since this space is lightly explored.
examples: simple 1to1 transformation function probability matrix, slowly building structures?, some kind of probabalistic lgg? 

This program operation will resemble my previous projects very closely with the basic concept of grammar being passed into
the data modeling function to adjust the probabilities of structural development at any point.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
For stock market data, we also will need to come up with a more robust method for solution representation other than halfkelly.
examples: maybe stdvs going out, stdevs of moving averages, some kind of stdev exponential mass??, maybe a volume target instead? p value of >0 of results?

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
We also need to come up with other candidate methods of model interpretation, other than exact evaluation.
examples: wrist-finger concept, detached sigmoid output, rolling normalization values (will probably need this)
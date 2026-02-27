'''
    Logan Kelsch - 2/26/26

    Holy moly took long enough to get started on this file.
    This file will contain all overarching and helper functionality in the evaluation of genes and USAGE of gene instantiation in our framework.
    this file will also contain the evaluation of grammar
    this file will also contain the evaluation of a population in terms of itself

    In this text block here I will develop the instructure.

BASIC:

    We ultimately will have some 1d array of length N (or 2d array if we split along days, I'm sure we will)
    This vector will be our solutions that will be interpreted somehow
            expanding on "solutions" and "interpreted somehow"

    gene evaluation will provide information for selecting parent / parent quality as well as iterating grammar
    grammar evaluation will provide information that goes hand in hand with gene evaluation for iterating grammar
    population evaluation will provide information that removes redundancy or excessive inbreeding in a population


solutions:

    We will have several different angles that we could try to solve for, which I will write in likely order of development.
    A main objective of developing this consists of ensuring a modular approach for indiana jones style swapping of methods.

    - Boolean anomoly detection
            this allows us to interpret positive values as true prediction (ex: price moves 3st up from 20 candle EMA)
    - Moving average SD
            this allows for predictions to be made regarding directionality, while holding a normalized output
    - Volume moving average SD
            This allows for more generalized predictions to be made regarding behavior of volume (directionality removed)
    - certainly some more approaches 

Mod dev step:
    
    The first steps will be:
    - creating a solution array for boolean anomoly detection
    - creating a gene matrix evaluation functionality for 2d instantiated gene matrices
            these can be made modular by having a class or object mechanism that ties evaluation style mechanism to solution instantiation mechanism
            for example, for the boolean anomoly, classification approaches will not be as effective, so we will have a cost score
                in which considers the difference of that and the expected value of this solution space.
                this will allow for a negative score to be undesirable, zero to be equal to market performance, and >>0 to be ideal.
                Expected cost = c_FP x FP + c_FN x FN


'''


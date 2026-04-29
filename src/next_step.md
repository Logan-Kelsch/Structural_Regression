Logan Kelsch
4/28/26

Documentation for candidate approach for implementation of custom MCTS as a grammar structure.

for this project Ive already got a 1d and 2d functioning stochastic grammar structure, but I would like to add a full fledged MCTS approach that is as modular as possible.

current list of development priority.

1. x parent and tf id probabilistic matters with alpha as a constant and random d,dd,k
2. implementation of alpha probabilistic matters and random d,dd,k
3. POSSIBLY consider exploring the spaces of d,dd,k using gaussian process.

I expect to complete 1 and 2 by presentation time for 490.
I'd be interested to one day develop 3.


LOGIC OVERVIEW:

We want to build something one step at a time (with new materials) that we expect to work as we have seen before (with other materials).

We are only inferencing what we have already seen, this is our Grammar.

Depending on the level of development, we allow error in recreation to be subject to destruction, as we assume the stochastic nature of unexplored spaces to be potentially unfit for life. That is for us to learn and do with in the next iteration.

So, therefore, the loop is:
- We built something one step at a time
- We tried to take the steps we expect to be best
- We learned something
- We are now building something again, at each step, we add g to X:
    - We can see what steps we have taken and where we are (X structure so far)
    - note: We can compare it to what we know (projection of G (quantified quality) onto X)
    - note: We can make sense of where we are (X Expected quantified quality)
    - 
    - We can THINK about WHERE we think is best to step (Inference of candidate child nodes from existing states.)
    - We can DECIDE *WHERE* to take next step (selection thereof of new g edge location (x sensor obtained))
        - This is comparison of all expected quality scores of X and probabilistic selection (tree structure of UCT scores)
    - We can THINK about HOW we think is best to step (Inference of candidate child node states from existing states.)
    - We can decide *HOW* to take next step (inference thereof for new g tf id)
        - We are narrowed down to the information at one node present in X
        - This is comparison of all possible states to come off of this node (1d vector of UCB scores)

This described process above is fully encapsulating of #1 in development priority.
The construction of this should be pretty solid moving forward

for myself I will write it again but shorter.

We are building something one step at a time, for each step:
- We look at all the steps we have taken.
    - We may have steps we have taken before (in G) and we may have steps we have never taken (not in G)
- We can assign quality to all of the steps we just took (project G UCT scores onto X)
    - This consists of steps we have taken before (Exists in G and can be put onto X states)
    - This consists of steps we have not taken before (Does not exist in G and must be assigned mystery/NULL UCT score)
- We can then decide what step we would like to continue from
    - Evaluation of tree structure of UCT scores (FROM ROOTS UP????)
    - This brings us to a single state to continue from  (x sensor)
- We can then decide how we would like to take that next step
    - Evaluatoin of child node structure of UCB1 scores (FROM selected node)
    - This brings us to a single selected function (tf id)
- LATER DEV:
    - Figure out wtf alpha assignment would look like
- LATER DEV:
    - Figure out how to consider continuous space exploration for d,dd,k with gaussian regression
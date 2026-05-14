Final path of things to build out.

i = inside evolutionary sample
j = inside  validation  sample (outside evolutionary sample)
k = outside  any seen   sample

we evolve & learn & observe in sample i.
we          learn & observe in sample j.
we                  observe in sample k.

there are a few policies that we need to compare.

- A null  policy,   null       evolution,   in i,j,k sample.
    - is in  sample in no    samples.
    - is out sample in i,j,k samples.

- A null  policy,   structured evolution,   in i,j,k sample.
    - is in  sample in i     samples.
    - is out sample in j,k   samples.

- learned policy,   null       evolution,   in i,j,k sample.
    - is in  sample in i     samples.
    - is out sample in j,k   samples.

- learned policy,   structured evolution,   in i,j,k sample.
    - is in  sample in i,j   samples.
    - is out sample in k     samples.


We want to evaluate the learning and modeling capacity and statistical significance for these approaches when samples are stationary and when they are walk forward.

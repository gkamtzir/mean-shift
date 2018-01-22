This is the 3rd exercise of the Parallel and Distributed Systems class of Electrical Engineering and Computer Engineering department at Aristotle University of Thessaloniki. It is a parallel implemetation of the Mean Shift algorithm in CUDA.

You can pass the standard deviation as an argument. If you don't, the default value will be used (1.4565 for the seeds dataset and 1.0 for the other one).

    $ make
    $ ./shared
    $ ./global
    $ ./shared_seeds
    $ ./global_seeds
    
Georgios Kamtziridis 8542, Winter Semester 2017-2018

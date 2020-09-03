# Linear2Circular
### Dissertation for Msc Bioinformatics 2020

#### Overall Idea:
Atifical metabolism algorithm that takes strings of fixed length (n) of a given alphabet (a) e.g. [A,B,C,D] and recombines in the nature of chemical reactions. A commoditiy to be produced is a randomly created string of n length from the characters in the alphabet. The algorithm returns the set of reactions requred to produce this commodity from the resources. Initial resources are strings of each character in the alphabet of n length, e.g. if n=3 then resources = ["AAA", "BBB", "CCC", "DDD"]. Intermediate products produced by the chain of reactions can be added to the resources and used to create the commodity.

These linear metabolisms are then transformed inot circular ones using a Simulated Annealing algorithm, whereby waste produced by reactions are rewired and reused as reactants for other reactions within the system.

#### Required modules:
* Matplotlib
* Networkx
* Numpy
* Pandas

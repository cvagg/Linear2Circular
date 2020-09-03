"""
---------------------------------------------------
Transforming Linear Metabolisms into Circular ones
---------------------------------------------------
Author: ID 160346817

Generates linear metabolisms from string chemistry's and transforms them into circular ones
via heuristic methods of optimisation.
"""


from abc import ABC, abstractmethod
import copy
import collections
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
import pandas as pd
import random
import time
import math


class Commodity:
    """
    ---------------------------
    Creates Commodity(s) Xc:
    ---------------------------

    Randomly generates string(s) which can be denoted as desired commodities
    to be made by the aritifical linear metabolism.

    ---
    Inputs:
    ---
    n = fixed length of commodity, same as metabolic system (int)
    a = alphabet language for string chemistries, same as metabolic system (list of char as strings)
    n_commodity = the number of commodities to be made (int)

    ---
    Returns:
    ---
    n_commodity number of strings of fixed length n generated from given
    alphabet a, as a list.
    """

    @classmethod
    def create_commodity(cls, n, a, n_commodity):

        """
        ---------------------------
        Creates Commodity(s) Xc:
        ---------------------------

        Randomly generates string(s) which can be denoted as desired commodities
        to be made by the aritifical linear metabolism.

        ---
        Inputs:
        ---
        n = fixed length of commodity, same as metabolic system (int)
        a = alphabet language for string chemistries, same as metabolic system (list of char as strings)
        n_commodity = the number of commodities to be made (int)

        ---
        Returns:
        ---
        n_commodity number of strings of fixed length n generated from given
        alphabet a, as a list.
        """

        res = []

        for comm in range(0, n_commodity):

            commodity = []

            for x in range(0, n):
                r = random.choice(a)
                commodity.append(r)

            c = "".join([str(i) for i in commodity])
            res.append(c)

        return res


class Reactions(ABC):
    """
    --------------------------------------
    Parent Class for Reaction Construction:
    --------------------------------------

    Base function as a format for the reaction chains as strings, such that

    n: reactant1 + reactant2 -(i)-> product1 + product2

    ---
    Inputs:
    ---
    reaction_n = the n in the string above, to denote the number of reactions.
                 This (int) appears as label for node, as r(n).
    reactant1, reactant2 = (strings) that act as reactants whch are to be reconmbined as products.
    i = index which the reactatns are cut, to be recombined as products (int)


    ---
    Returns:
    ---
    Abstract method product().
    Reaction class cannot be used as its own instance, must be used in accordance with product method
    in child class which governs the recombination of reactant strings to products.
    """

    def __init__(self, reaction_n, left, right, i):
        self.left = left
        self.right = right
        self.i = i
        self.reaction_n = reaction_n

        a, b = self.product()
        self.left2 = a
        self.right2 = b

    def __str__(self):
        a, b = self.product()
        return "{}: {} + {}  --({})--> {} + {}".format(self.reaction_n, self.left, self.right, self.i, a, b)

    def __eq__(self, other):
        """Checks if attributes are equal in two different instances,
        so that two reactions that are the same reactants, products and cut index are given the same hash id"""
        return self.left == other.left and self.right == other.right and self.left2 == other.left2 and self.right2 == other.right2

    def __hash__(self):
        return hash((self.left, self.right))

    @abstractmethod
    def product(self): pass


class Recomb_1(Reactions):
    """
    -----------------------------------
    Recombination of Strings, method 1:
    -----------------------------------

    Where a, b, x and y are substrings of AB and XY, reactant1 and reactant2 respectively.
    The recombination of reactant strings to form products are as follows:
    where i is the index in which the strings are cut, denoted as "."

    a.b + x.y -(i)--> a.y + x.b

    ---
    Inputs:
    ---
    See __init__ in parent class Reactions()

    ---
    Returns:
    ---
    product1 and product2 (str) produced by recombination of substrings.
    """

    def product(self):
        """a.b + x.y -(1)--> a.y + x.b"""
        return self.left[:self.i] + self.right[self.i:], self.right[:self.i] + self.left[self.i:]


class Recomb_2(Reactions):
    """
    -----------------------------------
    Recombination of Strings, method 2:
    -----------------------------------

    Where a, b, x and y are substrings of AB and XY, reactant1 and reactant2 respectively.
    The recombination of reactant strings to form products are as follows:
    where i is the index in which the strings are cut, denoted as "."

    a.b + x.y -(i)--> y.a + b.x

    ---
    Inputs:
    ---
    See __init__ in parent class Reactions()

    ---
    Returns:
    ---
    product1 and product2 (str) produced by recombination of substrings.
    """

    def product(self):
        """a.b + x.y -(1)--> y.a + b.x"""
        return self.right[self.i:] + self.left[:self.i], self.left[self.i:] + self.right[:self.i]


class ProductionChain:
    """
    -------------------------------------------
    Linear Metabolism using String Chemistries:
    -------------------------------------------

    Produces Linear Metabolic system as a set of reaction strings in the Reaction class
    format.

    ---
    Inputs:
    ---
    commodity = (lst) of strings that the linear metabolism need to make (Xc)
    recombination = recombination of strings technique that is wished to be used
                    to generate commodities, must be RS1 or RS2 (str).

    ---
    Returns:
    ---
    Instance for linear system for Xc and recombination string technique.
    """

    def __init__(self, commodity, recombination=Recomb_1):
        self.commodity = commodity
        self.recombination = recombination
        self.reactions = None

    def chain(self):
        """
        -----------------------------------------------------------
        Produces linear chain of reactions to generated commodity(s)
        -----------------------------------------------------------

        Overwrites self.reactions initiatied in __init__

        ---
        Returns:
        ---
        Reactions in format denoted in Reactions class, as strings in a set.

        ---
        Implementation problems:
        ---
        Must be called after every instance to define the reaction chain, before
        attempts to call other methods in this class, e.g. network, greedy etc.

        """
        commodity = self.commodity
        reactions = set()
        reaction_count = 0

        for comm in commodity:

            n = len(comm)
            repeated = r2_index(comm)
            inloop_r_count = 0

            for x in range(0, n - 1):

                if self.recombination == Recomb_1:

                    i = x + 1

                    if comm[x] != comm[x + 1]:
                        reaction_count = reaction_count + 1
                        inloop_r_count = inloop_r_count + 1

                        if inloop_r_count == 1:  # inital reaction
                            left1 = [comm[x] for i in range(0, n)]
                            right1 = [comm[x + 1] for i in range(0, n)]
                            # reaction_n = "r{}".format(reaction_count)
                            r = self.recombination(reaction_count, join_str(left1), join_str(right1), i)
                            left2, right2 = r.product()
                            # react_str_.append(reaction_n, str(r))
                            reactions.add(r)
                            continue

                        else:
                            left1 = left2
                            right1 = [comm[x + 1] for i in range(0, n)]
                            # reaction_n = "r{}".format(reaction_count)
                            r = self.recombination(reaction_count, join_str(left1), join_str(right1), i)
                            left2, right2 = r.product()
                            reactions.add(r)


                elif self.recombination == Recomb_2:

                    reaction_count = reaction_count + 1
                    inloop_r_count = inloop_r_count + 1

                    if inloop_r_count == 1:  # inital reaction
                        left1 = [repeated[0][0] for i in range(0, n)]
                        right1 = [repeated[1][0] for i in range(0, n)]
                        i = repeated[1][1]
                        r = self.recombination(reaction_count, join_str(left1), join_str(right1), i)
                        left2, right2 = r.product()
                        # react_str_.append(reaction_n, str(r))
                        reactions.add(r)
                        continue

                    else:
                        if right2 == comm:
                            break
                        else:
                            left1 = right2
                            right1 = [repeated[inloop_r_count][0] for i in range(0, n)]
                            # reaction_n = "r{}".format(reaction_count)
                            i = repeated[inloop_r_count][1]
                            r = self.recombination(reaction_count, join_str(left1), join_str(right1), i)
                            left2, right2 = r.product()
                            reactions.add(r)

                # all same char in comm
                elif comm == n * comm[0]:
                    left1 = [comm[x] for i in range(0, n)]
                    right1 = [comm[x + 1] for i in range(0, n)]
                    # reaction_n = "r{}".format(reaction_count)
                    r = self.recombination(reaction_count, join_str(left1), join_str(right1), i)
                    left2, right2 = r.product()
                    # react_str_.append(reaction_n, str(r))
                    reactions.add(r)
                    break

            # reaction_comm[comm] = reactions
        self.reactions = reactions
        return self.reactions


    def network(self):
        """
        --------------------------------------------------------
        Generates network of the current self.reactions instance
        --------------------------------------------------------

        Takes nodes of reactant species and reaction evens from self.reactions.

        ---
        Returns:
        ---
        G = Networkx multiple directional network graph, (networkx_object)

        mapping = a dictionary of simple labels for nodes in the network, (dict)
        i.e. xn for species and rn for reactions,
        where n is numbered chronologically in the chain of reactions
        """
        G = nx.MultiDiGraph()
        reaction_hash = []
        product_count = 0
        mapping = {}
        reaction_count = 0

        for r in self.reactions:
            reaction_count += 1

            reaction_dict = r.__dict__
            G.add_edge(reaction_dict.get('left'), hash(r))
            G.add_edge(reaction_dict.get('right'), hash(r))
            G.add_edge(hash(r), reaction_dict.get('left2'))
            G.add_edge(hash(r), reaction_dict.get('right2'))

            product_count += 1
            mapping[reaction_dict.get('left')] = "x{}".format(product_count)
            product_count += 1
            mapping[reaction_dict.get('right')] = "x{}".format(product_count)
            product_count += 1
            mapping[reaction_dict.get('left2')] = "x{}".format(product_count)
            product_count += 1
            mapping[reaction_dict.get('right2')] = "x{}".format(product_count)

            mapping[hash(r)] = "r{}".format(reaction_dict.get("reaction_n"))
            reaction_hash.append(hash(r))

        return G, mapping


    def network_nodes_species(self):
        """
        ---------------------------------------------------------------
        Generates dictionary for each node and its component goods role
        ---------------------------------------------------------------
        ---
        Returns:
        ---
        node_dict = (dict) of reactant species as keys and goods role as value,
        i.e. {xn:w} etc.
        r = reaction event
        Xc = commodity
        w = waste
        Xr = resource
        InPr = intermediate product
        """
        G, mapping = self.network()
        waste, resources, intmed_products = self.amenities()

        node_dict = {}

        for nd in G:
            # print(nd)
            if isinstance(nd, int):
                node_dict[nd] = "r"
            elif nd in self.commodity:
                node_dict[nd] = "Xc"
            elif nd in waste:
                node_dict[nd] = "w"
            elif nd in resources:
                node_dict[nd] = "Xr"
            elif nd in intmed_products:
                node_dict[nd] = "InPr"

        return node_dict

    def amenities(self):
        """
        ---------------------------------------------------------
        Assigns goods roles to the nodes in the metabolic network
        ---------------------------------------------------------

        Uses the in- and out- edges of the network of the instance based on
        current metabolic system. Whereby;
        - Raw resources are denoted to nodes which have only out- edges and no in- edges;
        - Waste are denoted to nodes which have only in- edges and no out-edges;
        - Intermediate products are the remaining nodes.

        ---
        Returns:
        ---
        waste, resources, intermed_products = (list) of the nodes which belongs to each goods role.
        """
        G, mapping = self.network()
        waste = []
        resources = []
        intmed_products = []

        for nd in G:
            # if nd[0] != "r":
            if not isinstance(nd, int):
                if not G.in_edges(nd):
                    resources.append(nd)
                elif not G.out_edges(nd):
                    if nd != self.commodity:
                        waste.append(nd)
                else:
                    intmed_products.append(nd)

        return waste, resources, intmed_products


    def greedy_search(self, check=False):
        """
        ---------------------------------------------------------
        Identifies candidate reactions for rewiring
        ---------------------------------------------------------

        Identifies two reactions which can be rewired together. If when rewired,
        the same useful product is made then the rewire is accepted, and self.reactions
        of the system is updated.

        1. Selects two reactions from the reaction set (g1 and g2)
        2. Rewire attempt is made
        3. If waste of g1 can be rewired as a reactant for g2,
           it is added to reaction set and returned.
        4. If not, steps 1-3 are repeated. Or until 1000 atempts
           have been made.

        ---
        Inputs:
        ---
        check = bool (True/False), for developmental stages. If True, print statements
                for found rewire canditates.

        ---
        Returns:
        ---
        self.reactions = set of reaction objects with updated rewire reactions
        """

        rewire_bool = True
        rewire_count = 0

        if len(self.reactions) <= 1:
            print("Commodity only takes one reaction to product, rewiring is not possible")

        else:

            while rewire_bool:

                g = random.sample(self.reactions, 2)
                rewire_count += 1

                g1_str = str(g[0])
                g2_str = str(g[1])

                if "*" not in g1_str and "*" not in g2_str:

                    # 2 reactons dictionaries as instance. class objects
                    g1_dict = g[0].__dict__
                    g2_dict = g[1].__dict__

                    #                     left right strings
                    g1_right2 = g1_dict.get("right2")
                    g1_left2 = g1_dict.get("left2")

                    g2_left2 = g2_dict.get("left2")
                    g2_right2 = g2_dict.get("right2")
                    g2_right = g2_dict.get("right")

                    #                     cut index i
                    g1_i = g1_dict.get("i")
                    g2_i = g2_dict.get("i")

                    # Finding reactions which can be rewired
                    if g1_dict.get("i") <= g2_dict.get("i"):
                        if self.recombination == Recomb_1:
                            r_waste = g1_right2
                            r1_bool = True

                            g1_product = g1_right2[g2_i]
                            g2_product = g2_left2[g2_i]

                            desired_product = g2_left2

                            if g1_product == g2_product:
                                rewire_bool = False

                                if check:
                                    print("candidate rewiring found!")
                                    print(str(g[0]))
                                    print(str(g[1]))
                            else:
                                continue

                        elif self.recombination == Recomb_2:
                            r_waste = g1_left2
                            r1_bool = False

                            desired_product = g2_right2
                            n = len(desired_product)

                            if g1_left2[-g2_i] == g2_right[g2_i]:
                                rewire_bool = False

                                if check:
                                    print("candidate rewiring found!")
                                    print(str(g[0]))
                                    print(str(g[1]))

                            else:
                                continue

                        # Rewiring reactions and replacing them in reaction set
                        r_n = "{}*".format(g2_dict.get("reaction_n"))

                        r = self.recombination(r_n, g2_dict.get("left"), r_waste, g2_dict.get("i"))

                        if check:
                            print(str(r))

                        # (2) remove and replace
                        g1 = g[0]
                        g2 = g[1]

                        # check same product is made
                        rewire_dict = r.__dict__

                        if r1_bool:
                            rewire_product = rewire_dict.get("left2")
                        elif r1_bool is False:
                            rewire_product = rewire_dict.get("right2")

                        if rewire_product == desired_product:

                            if check:
                                print("Rewire accepted and old reaction replaced")

                            rewire_bool = False

                            for obj in self.reactions:
                                if obj == g2:
                                    self.reactions.remove(obj)
                                    self.reactions.add(r)

                elif rewire_count >= 1000:
                    if check:
                        print("Rewire max attempts reached, {} total".format(rewire_count))

                    rewire_bool = False

        return self.reactions


def join_str(lst, new_line=False):
    """function which takes lists and joins them into one string"""
    if new_line:
        j_str = "/n".join([str(i) for i in lst])
    else:
        j_str = "".join([str(i) for i in lst])
    return j_str


def r2_index(species):
    """
    -----------------------------------
    Calculates index i position for RS2
    -----------------------------------
    Essentail for use in producing reaction chain for RS2.

    --
    Input:
    ---
    species = desired commodity to be made by chain of reactions, as a single str

    ---
    Returns:
    ---
    List of (char,n) where n is the number of times that char is repeated adjacent to the first.

    e.g. AABBBCCC returns [(A,2), (B,3), (C,4)]
         AABBBAAA returns [(A,2), (B,3), (A,3)]
    """
    result = []
    curr_char = species[0]
    curr_len = 1
    for i in range(1, len(species)):
        if curr_char != species[i]:
            result.append((curr_char, curr_len))
            curr_char = species[i]
            curr_len = 0
        curr_len += 1
    result.append((curr_char, curr_len))
    return result


def format_reactions(instance, display=True):
    """
    -----------------------------------------
    Puts reaction objects into display format
    -----------------------------------------

    Sorts the reactions set that are created by the Productionchain into
    a printable formated list where they are in chronological order.

    ---
    Inputs:
    ---
    instance = Linear object for system
    display = bool (True/False), if True then the strings are printed in order,

    ---
    Returns:
    ---
    (list) of str(reactions) in display order.

    ---
    Implementation errors:
    ---
    instance.chain() must be called to first generation self.reactions
    before format_reactions(instance) is used.
    """

    str_reactions = []
    int_reactions = []

    for x in instance.reactions:
        dct = x.__dict__

        int_reactions.append(int(dct.get("reaction_n")))

    int_reactions = sorted(int_reactions)

    for i in int_reactions:

        for r in instance.reactions:
            dct = r.__dict__
            if dct.get("reaction_n") == i:

                str_reactions.append(str(r))

                if display:
                    print(str(r))
    return str_reactions


def check_comm(instance):
    """
    ------------------------------------------------------------------------
    Developing tool for checking that reaction set produce correct commodity
    ------------------------------------------------------------------------

    Checks that all commodities are made in the reactions set.
    Unit testing for when 100s of commodities.

    ---
    Returns:
    ---
    True if elements in commodity list are in reactions list.
    """

    comm = instance.__dict__.get("commodity")

    accounted_comm = set()

    for c in comm:

        for r in instance.reactions:
            r_dict = r.__dict__

            for label, species in r_dict.items():

                if instance.__dict__.get("recombination") == Recomb_1:
                    product = r_dict.get("left2")

                else:
                    product = r_dict.get("right2")

                if product == c:
                    accounted_comm.add(c)

    if set(comm) == accounted_comm:
        return True
    else:
        print("Commodity:", set(comm))
        print("Commodity products made:", accounted_comm)
        return False


def nodes_mapped(instance):
    """"
    ------------------------------------------------------------------------
    Translates mapping from network() method with the node labels of species
    -----------------------------------------------------------------------

    Maps the nodes in the network to managable labels i.e. xn for reactant
    species and rn for reaction events, to their purpose in the metabolism.

    ---
    Input:
    ---
    instance = reaction set class objects from Linear class

    ---
    Returns:
    ---
    Dictionary of mapped nodes to goods as {node:goods}
    """
    G, mapping = instance.network()
    node_dict = instance.network_nodes_species()

    node_dict_mapped = {}

    for old_label, new_label in mapping.items():
        for node, ammentity in node_dict.items():
            if old_label == node:
                node_dict_mapped[new_label] = ammentity

    return node_dict_mapped


def colour_node(instance, reaction_colour='darkgrey', Xc_colour='orange', waste_colour='red', res_colour='limegreen', InPr_colour='lightblue'):
    """
    --------------------------------------------
    Provides colour map list for colouring nodes
    --------------------------------------------

    List of colours in order of nodes for use as a parameter when calling nx.draw_networkx()

    ---
    Inputs:
    ---
    instance = reaction set class objects from Linear class

    ---
    Returns:
    ---
    List with colours added in order the nodes appear, to be used as a
    direct parameter inside nx.draw() when drawing the desired graph for
    the input instance.

    ---
    Implementation errors:
    ---
    Must have nodes_mapped() as it is integrated inside this function.
    """
    G, mapping = instance.network()

    # relabel
    G = nx.relabel_nodes(G, mapping)

    node_dict_mapped = nodes_mapped(instance)

    waste, resources, intmed_products = instance.amenities()

    colour_map = []

    for nd in G:
        # print("nd",nd)
        for nd_label, ammentity in node_dict_mapped.items():
            # print("nd_label",nd_label)
            if nd_label == nd:
                # print(nd, nd_label)

                if ammentity == "r":
                    colour_map.append(reaction_colour)

                elif ammentity == "Xc":
                    colour_map.append(Xc_colour)

                elif ammentity == "w":
                    colour_map.append(waste_colour)

                elif ammentity == "Xr":
                    colour_map.append(res_colour)

                elif ammentity == "InPr":
                    colour_map.append(InPr_colour)
    return colour_map


def percent_waste(waste, resources, intmed_products):
    """
    ---------------------------
    Percent Waste of the System
    ---------------------------

    ---
    Inputs:
    ---
    waste, resources, intmed_products = (lists) returned from
    goods() method from Linear object.

    ---
    Returns:
    ---
    Percent waste of all species inputted (int)
    """
    n_species = len(waste) + len(resources) + len(intmed_products)
    p = (len(waste) / n_species) * 100

    percent_waste = round(p, 2)
    return percent_waste


def waste_per_product(waste, n_commodity):
    """
    --------------------------
    Waste per Comodity Produced
    --------------------------

    ---
    Inputs:
    ---
    waste = (list) of waste from goods() method from Linear class
    n_commodity = number of Xc produced

    ---
    Returns:
    ---
    waste per commodity Xc produced
    """
    return len(waste) / n_commodity


def simulated_annealing(instance, n_commodity, temp, temp_step=False, iteration=1000, check=False, network_gif=False):
    """
    --------------------------------------------------------
    Simulated Annealing algorithm for optimising circularity
    --------------------------------------------------------

    Conducts rewires using the rewire_search() method in the Linear class,
    if the circularity of the system has been improved (i.e. the percent waste)
    then the rewire is accepted. However if hte circularity is not improved (i.e.
    percent waste increases), then weither the rewire is accepted or not depends
    on the constriants set.

    These constraints are set by the temperature (temp) which decreases by 10%
    each iteration, forcing the algorithm to 'home in' on the oopimum circularity
    possibly achieved by the system.

    ---
    Inputs:
    ---
    - instance = set of reaction objects,
    - n_commodity = int
    - temp = start temperature of algorithm
    - temp_step = bool default=False, True creates a step wise iteration of decreasing temperature.
                  temperature decreases by 1% of iterations inputs.
    - iterations = (default=1000), iterations for finding rewiring candidates.
    - check = operates with rewire_search() method's check parameter, for unit testing

    ---
    Returns:
    ---
    Returns pandas dataframe with algorithm data.
    circular = Linear class object containing new circular self.reactions
    opt_data = data for all rewires that were accepted by probability of acceptance
    all_data = data for all rewires attempted even those not accepted.

    ---
    Implementation errors:
    ---
    instance (Linear class object) inputed as parameter must have called chain() method before calling this method,
    so that there is an initial self.reactions().
    """
    if network_gif:
        fig = plt.figure()
        camera = Camera(fig)
        G, mapping = instance.network()
        node_dict = instance.network_nodes_species()
        colour_map = colour_node(instance)

        pos = nx.spring_layout(G, k=0.5, iterations=100)
        nx.draw_networkx(G, pos=pos, node_color=colour_map, with_labels=False, node_size=50, width=0.5)

        plt.annotate("0", xy=(0.9, 0.9))

        camera.snap()

    circ_isntance = copy.deepcopy(instance)

    ##opmtised data
    # empty array with columns for (rewire no., percent waste, waste per product)
    opt_arr = np.empty((0, 3), int)
    # empty array with columns for (rewire no. time, temperature, probability, r)
    opt_ad = np.empty((0, 4), int)

    ##all data
    # empty array with columns for (rewire no., percent waste, waste per product, time, temp, p)
    all_arr = np.empty((0, 6), int)

    rewire_count = 0
    count = 0

    # calc waste for rewire 0 and add to matrix's
    waste, resources, intmed_products = circ_isntance.amenities()
    opt_arr = np.append(opt_arr, np.array(
        [[rewire_count, percent_waste(waste, resources, intmed_products), waste_per_product(waste, n_commodity)]]),
                        axis=0)
    opt_ad = np.append(opt_ad, np.array([[0, temp, 1, 0]]), axis=0)

    # all
    all_arr = np.append(all_arr, np.array([[rewire_count, percent_waste(waste, resources, intmed_products),
                                            waste_per_product(waste, n_commodity), 0, temp, 1]]), axis=0)

    # temperature decrease iterations

    if temp_step:
        t_iter = round(iteration * 0.01)
    else:
        t_iter = 1

    # start iterations
    while count < iteration:
        if check:
            print("count:", count)

        for i in range(0, t_iter):

            start = time.time()

            # make copy of reactions
            rewire_test = copy.deepcopy(circ_isntance)

            # make copy of opt matrix
            opt_arr_cp = copy.deepcopy(opt_arr)

            # greedy search for reaction rewire
            rewire_test.greedy_search(check)

            rewire_count += 1
            count += 1

            # calculate new rewire waste and add to copy of matrix
            waste, resources, intmed_products = rewire_test.amenities()
            opt_arr_cp = np.append(opt_arr, np.array([[rewire_count, percent_waste(waste, resources, intmed_products),
                                                       waste_per_product(waste, n_commodity)]]), axis=0)

            # define old(g) and new circulaity (waste) measurements (g_)
            g = opt_arr_cp[rewire_count - 1, 1]
            g_ = opt_arr_cp[rewire_count, 1]

            if g_ <= g:
                p = 1
                r = 0
            #             stop = time.time()

            elif g_ > g:
                if temp == 0:
                    p = 0
                    r = 1
                else:
                    p = math.exp((-1 * (g_ - g)) / temp)
                    r = random.uniform(0, 1)

            #             if r < p:
            #                 optimise = True
            #                 stop = time.time()
            #             else:
            #                 p = 0

            if r < p:
                stop = time.time()
                duration = stop - start

                # time for rewire is the time for previous rewire + the duration of current rewire
                t = opt_ad[rewire_count - 1, 0] + duration

                # merge instance with copy
                circ_isntance = rewire_test

                # merge copy with original array
                opt_arr = opt_arr_cp

                # add data to addition data array
                opt_ad = np.append(opt_ad, np.array([[t, temp, p, r]]), axis=0)

                # add to all matrix for rewire 0
                all_arr = np.append(all_arr, np.array([[rewire_count, percent_waste(waste, resources, intmed_products),
                                                        waste_per_product(waste, n_commodity), t, temp, p]]), axis=0)

                if network_gif:
                    G, mapping = instance.network()
                    node_dict = instance.network_nodes_species()
                    colour_map = colour_node(instance)

                    pos = nx.spring_layout(G, k=0.5, iterations=100)
                    nx.draw_networkx(G, pos=pos, node_color=colour_map, with_labels=False, node_size=50, width=0.5)

                    plt.annotate("0", xy=(0.9, 0.9))

                    camera.snap()

            else:
                stop = time.time()
                duration = stop - start

                # time for rewire is the time for previous rewire + the duration of current rewire
                t = opt_ad[rewire_count - 1, 0] + duration

                all_arr = np.append(all_arr, np.array([[rewire_count, percent_waste(waste, resources, intmed_products),
                                                        waste_per_product(waste, n_commodity), t, temp, p]]), axis=0)
                rewire_count -= 1

        # decrease temperature
        temp = 0.9 * temp

    opt_arr_cat = np.concatenate((opt_arr, opt_ad), axis=1)

    # dataframes
    opt_data = pd.DataFrame(data=opt_arr_cat[0:, 0:], index=None,
                            columns=["Rewire no.", "Percent waste (%)", "Waste per product", "Time (s)", "Temperature",
                                     "Probability", "R"])
    all_data = pd.DataFrame(data=all_arr[0:, 0:], index=None,
                            columns=["Rewire no.", "Percent waste (%)", "Waste per product", "Time (s)", "Temperature",
                                     "Probability"])

    if network_gif:
        animation = camera.animate()
        return animate
    else:
        return circ_isntance, opt_data, all_data


def SA_data_display(opt_df, all_df):
    """
    --------------------------------------
    Graph displays for simulated annealing
    --------------------------------------
    Generates following graphs for optimal and all rewiring attempts from simulated annealing:
    - Percent waste change over time of algorithm
    - Acceptance probability over time of algorithm
    - Temperature decrease over time of algorithm

    ---
    Inputs:
    ---
    opt_df = pandas dataframe for optimal rewire attempts returned in simulated_annealing() function
    all_df = " same but for all rewire attempts

    ---
    Returns:
    ---
    Nothing directly returned, just plots in display
    """
    fig, axs = plt.subplots(2, 3)

    axs[0,0].set_title("Optimal rewire attempts for circularity")
    axs[0,0].set_ylabel("Percent waste %")
    axs[0,0].set_xlabel("Time (s)")
    axs[0,0].plot(opt_df["Time (s)"], opt_df["Percent waste (%)"])

    axs[0,1].set_title("Optimal rewire attempts acceptance probability")
    axs[0,1].set_ylabel("Acceptance Probability")
    axs[0,1].set_xlabel("Time (s)")  # time??
    axs[0,1].scatter(opt_df["Time (s)"], opt_df["Probability"])

    axs[0,2].set_title("Optimal rewire attempts temperature decrease")
    axs[0,2].set_ylabel("Temperature")
    axs[0,2].set_xlabel("Time (s)")  # time??
    axs[0,2].plot(opt_df["Time (s)"], opt_df["Temperature"])

    axs[1,0].set_title("All rewire attempts for circularity")
    axs[1,0].set_ylabel("Percent waste %")
    axs[1,0].set_xlabel("Time (s)")
    axs[1,0].plot(all_df["Time (s)"], all_df["Percent waste (%)"])

    axs[1,1].set_title("All rewire attempts acceptance probability")
    axs[1,1].set_ylabel("Acceptance Probability")
    axs[1,1].set_xlabel("Time (s)")  # time??
    axs[1,1].scatter(all_df["Time (s)"], all_df["Probability"])

    axs[1,2].set_title("All rewire attempts temperature decrease")
    axs[1,2].set_ylabel("Temperature")
    axs[1,2].set_xlabel("Time (s)")  # time??
    axs[1,2].plot(all_df["Time (s)"], all_df["Temperature"])

    return plt.show()


def percent_change(dataframe):
    p_w_0 = dataframe["Percent waste (%)"].iloc[0]
    p_w_end = dataframe["Percent waste (%)"].iloc[-1]

    p_diff = (p_w_end - p_w_0)
    p_change = (p_diff / p_w_end) * 100

    # print("Percentage change: %.2f%%" % p_change)
    return p_change


def what_node(instance, node):
    """
    -----------------------------
    What good maps to which node?
    -----------------------------
    ---
    Inputs:
    ---
    instance = ProductionChain object
    node = a node from that instance which you want the ammentity for

    ---
    Returns:
    ---
    node's asociated ammentity as str
    """
    map_dict = nodes_mapped(instance)

    for nd, ammentity in map_dict.items():
        if nd == node:
            return ammentity


def centrality_measures(G):
    """
    -------------------
    Centrailty measures
    -------------------
    ---
    Input:
    ---
    G = networkx object for metabolic network

    ---
    Returns:
    ---
    bet_cen, clo_cen, eig_cen = dictionaries for each centrality measure from networkx
    """
    G1_d = nx.Graph(G)
    G1_ud = G1_d.to_undirected()

    G1_components = nx.connected_components(G1_ud)
    S = [G.subgraph(c).copy() for c in nx.connected_components(G1_ud)]
    S = nx.Graph(S[0])

    G1_components = nx.connected_components(G1_ud)
    G1_mc = S

    # Betweenness Centrality
    bet_cen = nx.betweenness_centrality(G1_mc)
    # print("Betweenness Centrality",bet_cen)

    # Closeness centrality
    clo_cen = nx.closeness_centrality(G1_mc)
    # print("Closeness centrality",clo_cen)

    # Eigenvector centrality
    eig_cen = nx.eigenvector_centrality(G1_mc)
    # print("Eigenvector centrality",eig_cen)

    return bet_cen, clo_cen, eig_cen


def degree_distribution_scatter(G1, G2, G1_label="G1", G2_label="G2"):
    """
    ----------------------------------------
    Degree distribution of directed networks
    ----------------------------------------
    ---
    Inputs:
    ---
    G1, G2 = Network object for rs1 and rs2
    G1_label, G2_label = labels for legend on graph

    ---
    Returns:
    ---
    Scatter graph for in- out- degree distribution for each network
    """
    degs_c = {}

    for n in G2.nodes():
        deg_c = G2.degree(n)
        if deg_c not in degs_c:
            degs_c[deg_c] = 0
        degs_c[deg_c] += 1

    items_c = sorted(degs_c.items())

    degs = {}

    for n in G1.nodes():
        deg = G1.degree(n)
        if deg not in degs:
            degs[deg] = 0
        degs[deg] += 1

    items = sorted(degs.items())

    plt.figure("degree distribution of {} and {}".format(G1_label, G2_label))
    plt.scatter([k for (k, v) in items], [v for (k, v) in items], label=G1_label, color="lightgrey")
    plt.scatter([k for (k, v) in items_c], [v for (k, v) in items_c], label=G2_label, color="black")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True)
    plt.xlabel("Degree")
    plt.ylabel("P(K)")
    plt.legend()
    plt.show()

    return items, items_c


def main():

    # Generating commodity
    n_commodity = 100
    commodity = Commodity.create_commodity(8, ["A", "B", "C", "D"], n_commodity)

    print("Desired commodity(s):")
    [print(i) for i in commodity]

    # Original reaction instance chain
    rs1 = ProductionChain(commodity, Recomb_1)
    rs2 = ProductionChain(commodity, Recomb_2)

    reactions1 = rs1.chain()
    reactions2 = rs2.chain()

    # Reaction Key:
    print("\nRS1 linear metabolism to produce commodity:")
    format_reactions(rs1)
    print("\n")

    print("\nRS2 linear metabolism to produce commodity:")
    format_reactions(rs2)
    print("\n")

    G1, mapping1 = rs1.network()
    G2, mapping2 = rs2.network()

    # relabel
    G1 = nx.relabel_nodes(G1, mapping1)
    G2 = nx.relabel_nodes(G2, mapping2)

    colour_map1 = colour_node(rs1)
    colour_map2 = colour_node(rs2)

    # Bipartite Networks
    # top1 = nx.bipartite.sets(G1)[0]
    # pos1 = nx.bipartite_layout(G1, top1)

    # top2 = nx.bipartite.sets(G2)[0]
    # pos2 = nx.bipartite_layout(G2, top2)

    # Spring Layout
    pos1 = nx.spring_layout(G1, k=0.5, iterations=100)
    pos2 = nx.spring_layout(G2, k=0.5, iterations=100)

    plt.figure("RS1 Rl")
    nx.draw_networkx(G1, pos=pos1, node_color=colour_map1, with_labels=False, node_size=50, width=0.5)

    plt.figure("RS2 Rl")
    nx.draw_networkx(G2, pos=pos2, node_color=colour_map2, with_labels=False, node_size=50, width=0.5)

    # Circluarisation
    circ_react1, opt1, all_1 = simulated_annealing(rs1, n_commodity, temp=10, iteration=1000, temp_step=False)
    SA_data_display(opt1, all_1)

    circ_react2, opt2, all_2 = simulated_annealing(rs2, n_commodity, temp=10, iteration=1000, temp_step=False)
    SA_data_display(opt2, all_2)

    colour_map1c = colour_node(rs1c)
    colour_map2c = colour_node(rs2c)

    # Bipartite Networks
    # top1c = nx.bipartite.sets(G1c)[0]
    # pos1c = nx.bipartite_layout(G1c, top1c)

    # top2c = nx.bipartite.sets(G2c)[0]
    # pos2c = nx.bipartite_layout(G2c, top2c)

    # Spring Layout
    pos1c = nx.spring_layout(G1c, k=0.5, iterations=100)
    pos2c = nx.spring_layout(G2c, k=0.5, iterations=100)

    # Circular metabolic Networks
    plt.figure("RS1 Rc")
    nx.draw_networkx(G1c, pos=pos1c, node_color=colour_map1c, with_labels=False, node_size=50, width=0.5)

    plt.figure("RS2 c")
    nx.draw_networkx(G2c, pos=pos2c, node_color=colour_map2c, with_labels=False, node_size=50, width=0.5)

    plt.show()


if __name__ == "__main__":
    main()
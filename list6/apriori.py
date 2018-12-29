import numpy as np
import math

import sys
import itertools


class Apriori:

    def __init__(self, X, T, alpha=0.1, beta=0.2, gamma=0.3, debug=False):
        """
        Generate association rules
        :param X: set of products
        :param T: set of transactions (subsets of X) 
        :param alpha: minimum support
        :param beta: minimum confidance
        :param gamma: minimum lift
        :param debug: If True prints debug messages to sys.stderr
        """
        self.X = X ## a list of products?
        self.T = T ## i want set of frozenset
        self.n = len(T)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.debug = debug
        self.supp_by_set = {}

        self._generate_singleton_support()

        self.frequent = self._generate_frequent_sets()
        self.rules = self._generate_good_rules()

       


    def _get_support(self, s):
        
        if s in self.supp_by_set:
            return self.supp_by_set[s]
        is_in = 0
        for t in self.T:
            
            if s <= t:
                is_in += 1

        
        self.supp_by_set[s] = is_in/self.n

        return is_in/self.n
    

    def _get_stats(self, a, b):
        suppA = self._get_support(a)
        suppB = self._get_support(b)
        suppAB = self._get_support(a | b)

        conf = suppAB/suppA
        lift = conf/suppB

        return conf, lift, suppA, suppB, suppAB

    def _generate_singleton_support(self):

        for x in self.X:
            self.supp_by_set[frozenset([x])] = 0 

        for t in self.T:
            for x in t:
                self.supp_by_set[frozenset([x])] += 1

        for x in self.X:
            self.supp_by_set[frozenset([x])] /= self.n 
        


    def _generate_frequent_sets(self):
        
        if self.debug:
            print("X size: {}".format(len(self.X)), file=sys.stderr)

        freq_by_size = [set()]

        for item in self.X:
            i = frozenset([item])

            if self._get_support(i) >= self.alpha:
                freq_by_size[0].add(i)

        freq_by_size[0] = list(freq_by_size[0])

        if self.debug:
            print("Sets of size 1: {}".format(len(freq_by_size[0])), file=sys.stderr)

        for k in range(1, self.n):
            freq_by_size.append(set())

            for sec, fst in enumerate(freq_by_size[k-1]):
                for snd in freq_by_size[k-1][sec+1:]:
                    new_set = fst | snd
                    if len(new_set) == k+1 and self._get_support(new_set) >= self.alpha:
                        freq_by_size[k].add(new_set)
            
            freq_by_size[k] = list(freq_by_size[k])

            if self.debug:
                print("Sets of size {}: {}".format(k+1, len(freq_by_size[k])), file=sys.stderr)

            if len(freq_by_size[k]) == 0:
                break
        
        if self.debug:
            print("Max set length: {}".format(k), file=sys.stderr)
            print("Number of sets: ", sum([len(sets) for sets in freq_by_size]), file=sys.stderr)

        return frozenset([item for s in freq_by_size for item in s])

    def _generate_good_rules(self):
        
        rules = []

        for a in self.frequent:
            for b in self.frequent:
                if len(a & b) == 0 and a | b in self.frequent:
                    
                    conf, lift, _, _, _ = self._get_stats(a,b)

                    if self.debug:
                        print("Rule {} => {}".format(a,b), file=sys.stderr)
                        print("confidance: {}".format(conf), file=sys.stderr)
                        print("lift: {}".format(lift), file=sys.stderr)
                    
                    if conf >= self.beta and lift >= self.gamma:
                        rules.append((a,b))

        return rules


    def print_frequent_sets(self, verbose=False):

        for s in self.frequent:
            if verbose:
                print(tuple(s), self._get_support(s))
            else:
                print(tuple(s))




    def print_rules(self, verbose=False):

        for a,b in self.rules:
            print("{} => {}".format(tuple(a), tuple(b)))
            if verbose:
                conf, lift, sa, sb, sab = self._get_stats(a,b)
                print("confidance: {}\t lift: {}".format(conf, lift))
                print("supports: A: {}\t B: {} \t AvB: {}".format(sa,sb,sab))
        return

    
        


if __name__ == "__main__":
    
    X = ["marchew", "jabko", "zarowka"]
    T = [["marchew", "jabko"],
         ["marchew"],
         ["zarowka"],
         ["zarowka", "marchew", "jabko"],
         ["marchew", "jabko"]]    

    T = [frozenset(t) for t in T]



    a = Apriori(X, T, 0.5, debug=True)

    a.print_frequent_sets(True)
    a.print_rules(True)



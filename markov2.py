from collections import defaultdict, Counter
import math
from random import random

from graphviz import Digraph
from functools import partial, reduce
import json


def product(iterable):
    return reduce(float.__mul__, iterable, 1.0)


def marginal_probability(parameters, cluster):
    return parameters[0][cluster]


def statistical_model(parameters, cluster, sequence):
    """
    :param parameters: theta
    :param cluster: c_k
    :param sequence: |x|
    :return: p_k(|x| | c_k, theta)
    """
    head, *tail = sequence
    return parameters[1][cluster][head] \
           * product(
        parameters[2][cluster][previous_state][next_state]
        for previous_state, next_state in zip(sequence, tail))


def sequence_probability(clusters, parameters, sequence):
    return sum([
                   marginal_probability(parameters, cluster) \
                   * statistical_model(parameters, cluster, sequence)
                   for cluster in clusters
                   ])


# def make_markov(transitions):
#     markov = defaultdict(Counter)
#     for previous_state, next_state in transitions:
#         markov[previous_state].update([next_state])
#     return markov
#
#
def membership(clusters, parameters, sequence, cluster):
    return marginal_probability(parameters, cluster) \
           * statistical_model(parameters, cluster, sequence) \
           / sum(marginal_probability(parameters, cluster)
                 * statistical_model(parameters, cluster, sequence)
                 for cluster in clusters)


def new_mixture_weights(clusters, parameters, sequences):
    return {
        cluster: sum(
            membership(clusters, parameters, sequence, cluster)
            for sequence in sequences) / sum(
            sum(
                membership(clusters, parameters, sequence, cluster)
                for sequence in sequences
            )
            for cluster in clusters
        )
        for cluster in clusters
        }


def new_initial_probabilities(clusters, parameters, sequences, states):
    return {
        cluster: {
            state: sum(
                membership(clusters, parameters, sequence, cluster) + (
                    0.01 / len(states))
                for sequence in sequences
                if sequence[0] == state
            ) / sum(
                sum(
                    membership(clusters, parameters, sequence, cluster) + (
                    0.01 / len(states))
                    for sequence in sequences
                    if sequence[0] == state
                )
                for state in states
            )
            for state in states
            }
        for cluster in clusters
        }


def new_transition_matricies(clusters, parameters, sequences, states):
    return {
        cluster: {
            state_j: {
                state_l: sum(
                    membership(clusters, parameters, sequence, cluster) * n_count(
                        sequence, state_j, state_l) + (0.01 / len(states))
                    for sequence in sequences
                ) / sum(
                    sum(
                        membership(clusters, parameters, sequence,
                                   cluster) * n_count(
                            sequence, state_j, state) + (0.01 / len(states))
                        for sequence in sequences
                    )
                    for state in states
                )
                for state_l in states
                }
            for state_j in states
            }
        for cluster in clusters
        }


def n_count(sequence, state_j, state_l):
    #print(sequence)
    #print(list(zip(sequence, sequence[1:]) ))
    count = sum(1 for j, l in zip(sequence, sequence[1:]) if
               j == state_j and l == state_l)
    #print('Checking count between {} and {} and it comes to: {}'.format(state_j, state_l, count))
    return count


def train(sequences):
    states = reduce(set.union, map(set, sequences), set())
    clusters = ['A', 'B']

    parameters = initial_parameters(clusters, states)

    for _ in range(20):
        # print(json.dumps(parameters, indent=2))

        # for sequence in sequences:
        #     for cluster in clusters:
        #         m = membership(clusters, parameters, sequence, cluster)
        #         print('Sequence {} has membership value {} for cluster {}'.format(sequence, m, cluster))

        parameters = (
            new_mixture_weights(clusters, parameters, sequences),
            new_initial_probabilities(clusters, parameters, sequences, states),
            new_transition_matricies(clusters, parameters, sequences, states),
        )

    print('digraph g {')
    print('rankdir=LR;')
    for cluster in parameters[1]:
        print('subgraph cluster_{} {{'.format(cluster))
        print('label="Cluster {}"'.format(cluster))
        for state in parameters[1][cluster]:
            print('"{}_{}" [label="{}"];'.format(cluster, state, state))
            if parameters[1][cluster][state] >= 0.3:
                print('"{}" -> "{}_{}" [label="{}"];'.format(cluster, cluster, state,
                                                          parameters[1][
                                                              cluster][state]))
        for state1 in parameters[2][cluster]:
            for state2 in parameters[2][cluster][state1]:
                if parameters[2][cluster][state1][state2] >= 0.3:
                    print('"{}_{}" -> "{}_{}" [label="{}"];'.format(cluster, state1, cluster, state2,
                                                              parameters[2][
                                                                  cluster][
                                                                  state1][
                                                                  state2]))
        print('}')
    print('}')

    return None


def roll(iterable):
    r = [random() for _ in iterable]
    s = sum(r)
    r = [i / s for i in r]
    return zip(iterable, r)


def initial_parameters(clusters, states):
    mixture_weights = {
        cluster: 1.0 / len(clusters) for cluster in clusters
        }

    initial_probabilities = {
        cluster: {
            state: r for state, r in roll(states)
            } for cluster in clusters
        }

    transition_matricies = {
        cluster: {
            state: {
                state: r
                for state, r in roll(states)
                }
            for state in states
            }
        for cluster in clusters
        }
    parameters = (mixture_weights, initial_probabilities, transition_matricies)
    return parameters


if __name__ == '__main__':
    sequences = [
        ['a', 'b'],
        ['a', 'c', 'd'],
        ['a', 'c', 'e'],
        ['b', 'c', 'e']
    ]

    mixture_model = train(sequences)

import numpy as np
import pandas as pd

#data is in the matrix where rows are the observations and the columns are the variables,
#last column holds the classes


def get_tree_priors(classes):
    unique_classes = np.unique(classes)
    tree_priors = dict([(option, np.sum(classes == option) / classes.shape[0]) for option in unique_classes])
    return tree_priors


def get_node_priors(node_classes, tree_cases):
    unique_classes = np.unique(node_classes)
    node_priors = dict([(option, np.sum(node_classes == option) / tree_cases) for option in unique_classes])
    return node_priors


def get_class_given_node(node_priors, class_id):
    node_prior_sum = sum(node_priors.values())
    class_given_node = node_priors[class_id]/node_prior_sum
    return class_given_node


#####################################################################################


def get_all_splits(self, observations, classes):
    '''Only takes numerical variables. For categoricals, map them into binary variables.'''


# TODO: implement cost function


def get_class_prior(tree_prior, classes):
    unique_classes = np.unique(classes)
    # TODO: class priors is wrong, correct this
    class_priors = [tree_prior[tree_prior[:, 1] == option, 0] * np.sum(classes == option) / classes.shape[0] for option
                    in unique_classes]
    return class_priors


def get_class_given_node(class_priors, p):
    return [class_prior / p for class_prior in class_priors]


def get_gini_index(tree_prior, classes):
    class_priors = get_class_prior(tree_prior, classes)
    p = np.sum(class_priors)
    class_given_node = get_class_given_node(class_priors, p)
    classes = range(len(class_given_node))
    index = 0
    for i in classes:
        for j in classes:
            if i != j:
                index += class_given_node[i] * class_given_node[j]
    return index


def get_split_cost(split_value, variable_values, classes, tree_prior):
    # data = pd.DataFrame(np.hstack((variable_values, classes)))
    node_index, p = get_gini_index(tree_prior, classes)

    left_int = variable_values <= split_value
    left_classes = classes[left_int]
    left_index = get_gini_index(tree_prior, left_classes)
    left_prop = np.sum(left_int) / len(classes)

    right_int = variable_values > split_value
    right_classes = classes[right_int]
    right_index = get_gini_index(tree_prior, right_classes)
    right_prop = np.sum(right_int) / len(classes)

    impurity_decrease = node_index - (left_prop * left_index) - (right_prop * right_index)
    return impurity_decrease


class DecisionTree:
    """Decision tree class based on CART algorithm"""

    def __init__(self, training_data, test_data):
        self.training_data = training_data
        self.test_data = test_data


class Node:
    '''Node class that holds information on the observations it works on and the splits'''

    def __init__(self, training_data):
        self.training_data = training_data
        self.tree_prior = np.zeros()

    def get_variable_split(self, variable_values, classes):
        # data = pandas.DataFrame(np.hstack((variable_values,classes)))
        # sorted = data.sort_values(0)
        sorted = np.unique(variable_values)
        variable_splits = np.empty([len(sorted)-1, 1])
        for i in range(len(sorted)-1):
            variable_splits[i] = (sorted[i]+sorted[i+1])/2
        costs = [get_split_cost(split, variable_values, classes, self.tree_prior) for split in variable_splits]
        return variable_splits[costs == np.min(costs)]

    def get_split(self):
        classes = self.training_data[:, -1]
        variable_splits = [self.get_variable_split(self.training_data[:, column], classes) for column in range(self.training_data.shape[1] - 1)]
        # TODO: check the costs of the splits and return the best split and the training points for child nodes

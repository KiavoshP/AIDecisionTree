import numpy as np
from collections import Counter
import time


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.
        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.
        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        # """Get a child node based on the decision function.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
        # Args:͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
        #     feature (list(int)): vector for feature.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
        # Return:͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
        #     Class label if a leaf node, otherwise a child node.͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
        # """͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
        """Determine recursively the class of an input array by testing a value
           against a feature's attributes values based on the decision function.

        Args:
            feature: (numpy array(value)): input vector for sample.

        Returns:
            Class label if a leaf node, otherwise a child node.
        """

        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data.
    Tree is built fully starting from the root.

    Returns:
        The root node of the decision tree.
    """
    a4_r = DecisionNode(None, None, None, 1)
    a4_l = DecisionNode(None, None, None, 0)
    a4l_r = DecisionNode(None, None, None, 0)
    a4l_l = DecisionNode(None, None, None, 1)

    a3_r = DecisionNode(a4_l, a4_r, lambda a4: a4[3] == 0)
    a3_l = DecisionNode(a4l_l, a4l_r, lambda a4: a4[3] == 0)

    root_r = DecisionNode(None, None, None, 1)
    root_l = DecisionNode(a3_l, a3_r, lambda a3: a3[2] == 0)

    root_a1 = DecisionNode(root_l, root_r, lambda a1: a1[0] == 0)


    decision_tree_root = root_a1

    return decision_tree_root


def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.

    Classifier output vs true labels, which is equal to:
    Predicted  vs  Actual Values.

    Output will in the format:

                        |Predicted|
    |T|                
    |R|    [[true_positive, false_negative],
    |U|    [false_positive, true_negative]]
    |E|

    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        A two dimensional array representing the confusion matrix.
    """

    matrix = np.zeros((2,2))
    for i in range(len(true_labels)):
        if true_labels[i] == classifier_output[i]:
            matrix[true_labels[i]][true_labels[i]] += 1
        else: 
            matrix[true_labels[i]][classifier_output[i]] += 1
            
    return [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]


def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.
    Precision is measured as:
        true_positive/ (true_positive + false_positive)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The precision of the classifier output.
    """

    cm = confusion_matrix(classifier_output, true_labels)
    return cm[0][0] / (cm[0][0] +cm[1][0])

def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.
    Recall is measured as:
        true_positive/ (true_positive + false_negative)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The recall of the classifier output.
    """

    cm = confusion_matrix(classifier_output, true_labels)
    return cm[0][0] / (cm[0][0] +cm[0][1])


def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.
    Accuracy is measured as:
        correct_classifications / total_number_examples
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The accuracy of the classifier output.
    """
    cm = confusion_matrix(classifier_output, true_labels)
    return (cm[0][0] + cm[1][1]) / len(classifier_output)


def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.
    Returns:
        Floating point number representing the gini impurity.
    """
    if len(class_vector) <= 0:
        return 1
    false_zo_count = Counter(class_vector)
    leaf_gini_imp_false = 1 - np.power(false_zo_count[0] / len(class_vector) , 2) - np.power(false_zo_count[1] / len(class_vector) , 2)
    return leaf_gini_imp_false


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    
    return (gini_impurity(previous_classes) - np.sum([gini_impurity(i) * len(i) for i in current_classes]) / len(np.hstack(current_classes)))

    

class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        self.root = self.__build_tree__(features, classes)

    
    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """

        if depth == self.depth_limit :
            y = Counter(classes).most_common()[0][0]
            return DecisionNode(None, None, None, y)
    
        if len(classes) == 1 :
            y = Counter(classes).most_common()[0][0]
            return DecisionNode(None, None, None, y)
        
        # Finding the best split
        ledger = [0,0,0,None,None]
        for feature in range(features.shape[1]):

            featured_sorted = features[np.argsort(features[:, feature])]
            classes_sorted = classes[np.argsort(features[:, feature])]

            for dp in range(0, len(features) - 1):

                curr_gini_info = gini_gain(classes_sorted, [classes_sorted[:dp+1], classes_sorted[dp+1:]])

                if curr_gini_info > ledger[0]:
                    ledger[0] = curr_gini_info
                    ledger[1] = feature
                    ledger[2] = dp
                    ledger[3] = classes_sorted
                    ledger[4] = featured_sorted

        if ledger[0] <= 0:
            y = Counter(classes).most_common()[0][0]
            return DecisionNode(None, None, None, y)

        curr_features = ledger[4]
        curr_classes = ledger[3]

        
        split_value = curr_features[ledger[2], ledger[1]]
        left_leaf = self.__build_tree__(curr_features[:ledger[2]+1], curr_classes[:ledger[2]+1], depth = depth + 1)
        right_leaf = self.__build_tree__(curr_features[ledger[2]+1:], curr_classes[ledger[2]+1:], depth = depth + 1)

        curr_root = DecisionNode(left_leaf, right_leaf, lambda feat: feat[ledger[1]] <= split_value)

        return curr_root


    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """
        class_labels = [self.root.decide(feature) for feature in features]
        # class_labels = self.root.decide(feature=features)
        return class_labels
   

class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        for tree in range(self.num_trees):
            dp_sample_indx = np.random.choice(features.shape[0], round(self.example_subsample_rate * features.shape[0]), replace=True)
            feature_samples = features[dp_sample_indx]
            class_samples = classes[dp_sample_indx]

            dt = DecisionTree(depth_limit = self.depth_limit)
            dt.fit(feature_samples, class_samples)
            self.trees.append(dt)

    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        """

        ballet = np.array([dt.classify(features) for dt in self.trees])
        return([Counter(ballet[:,label]).most_common()[0][0] for label in range(features.shape[0])])


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """

        return (data[:][:]*data[:][:]) + data[:][:]

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """
        sums = np.sum(data[:100][:], axis=1)
        indx = np.where(sums == max(sums))[0][0]
        max_sum = sums[indx]
        return (max_sum, indx)

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        return(list(Counter(np.hstack(data)[np.hstack(data) > 0]).items()))

    
    
    def non_vectorized_glue(self, data, vector, dimension='c'):
        """Element wise array arithmetic with loops.
        This function takes a multi-dimensional array and a vector, and then combines
        both of them into a new multi-dimensional array. It must be capable of handling
        both column and row-wise additions.
        Args:
            data: multi-dimensional array.
            vector: either column or row vector
            dimension: either c for column or r for row
        Returns:
            Numpy array of data.
        """
        if dimension == 'c' and len(vector) == data.shape[0]:
            non_vectorized = np.ones((data.shape[0],data.shape[1]+1), dtype=float)
            non_vectorized[:, -1] *= vector
        elif dimension == 'r' and len(vector) == data.shape[1]:
            non_vectorized = np.ones((data.shape[0]+1,data.shape[1]), dtype=float)
            non_vectorized[-1, :] *= vector
        else:
            raise ValueError('This parameter must be either c for column or r for row')
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row, col] = data[row, col]
        return non_vectorized

    def vectorized_glue(self, data, vector, dimension='c'):
        """Array arithmetic without loops.
        This function takes a multi-dimensional array and a vector, and then combines
        both of them into a new multi-dimensional array. It must be capable of handling
        both column and row-wise additions.
        Args:
            data: multi-dimensional array.
            vector: either column or row vector
            dimension: either c for column or r for row
        Returns:
            Numpy array of data.
            
        """
        if dimension == 'c' and len(vector) == data.shape[0]:
            new_vec = vector.reshape(data.shape[0],1)
            data = np.append(data, new_vec, axis= 1)
        elif dimension == 'r' and len(vector) == data.shape[1]:
            new_vec = vector.reshape(1,data.shape[1])
            data = np.append(data, new_vec, axis= 0)
        else:
            raise ValueError('This parameter must be either c for column or r for row')
        return data
    
    def non_vectorized_mask(self, data, threshold):
        """Element wise array evaluation with loops.
        This function takes a multi-dimensional array and then populates a new
        multi-dimensional array. If the value in data is below threshold it
        will be squared.
        Args:
            data: multi-dimensional array.
            threshold: evaluation value for the array if a value is below it, it is squared
        Returns:
            Numpy array of data.
        """
        non_vectorized = np.zeros_like(data, dtype=float)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                val = data[row, col]
                if val >= threshold:
                    non_vectorized[row, col] = val
                    continue
                non_vectorized[row, col] = val**2

        return non_vectorized

    def vectorized_mask(self, data, threshold):
        """Array evaluation without loops.
        This function takes a multi-dimensional array and then populates a new
        multi-dimensional array. If the value in data is below threshold it
        will be squared. You are required to use a binary mask for this problem
        Args:
            data: multi-dimensional array.
            threshold: evaluation value for the array if a value is below it, it is squared
        Returns:
            Numpy array of data.
        """
        mask = np.ones((data.shape[0],data.shape[1]))
        mask[data < threshold] = 2
        data = np.power(data, mask)
        return data

def return_your_name():
    # return your name͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    # TODO: finish this͏︆͏󠄃͏󠄌͏󠄍͏󠄂͏️͏󠄇͏︊͏︃
    return 'Kiavosh Peynabard'

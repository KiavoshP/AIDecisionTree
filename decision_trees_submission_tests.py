import unittest
import submission as dt
import numpy as np
import time


def curr_time_millis():
    return int(round(time.time() * 1000))

class DecisionTreePart1Tests(unittest.TestCase):
    """Test tree example, confusion matrix, precision, recall, and accuracy.

    Attributes:
        hand_tree (DecisionTreeNode): root node of the built example tree.
        ht_examples (list(list(int)): features for example tree.
        ht_classes (list(int)): classes for example tree."""

    def setUp(self):
        """Setup test data.
        """

        self.hand_tree = dt.build_decision_tree()
        self.ht_examples = [[1, 0, 0, 0],
                            [1, 0, 1, 1],
                            [0, 1, 0, 0],
                            [0, 1, 1, 0],
                            [1, 1, 0, 1],
                            [0, 1, 0, 1],
                            [0, 0, 1, 1],
                            [0, 0, 1, 0]]
        self.ht_classes = [1, 1, 1, 0, 1, 0, 1, 0]

    def test_hand_tree_accuracy(self):
        """Test accuracy of the tree example.

        Asserts:
            decide return matches true class for all classes.
        """

        for index in range(0, len(self.ht_examples)):
            decision = self.hand_tree.decide(self.ht_examples[index])

            assert decision == self.ht_classes[index]

    def test_confusion_matrix(self):
        """Test confusion matrix for the example tree.

        Asserts:
            confusion matrix is correct.
        """

        answer = [1, 0, 0, 1, 0, 0, 0]
        true_label = [1, 1, 1, 0, 0, 0, 0]
        test_matrix = [[1, 2], [1, 3]]

        assert np.array_equal(test_matrix, dt.confusion_matrix(answer,true_label))

    def test_precision_calculation(self):
        """Test precision calculation.

        Asserts:
            Precision matches for all true labels.
        """

        answer = [0, 0, 0, 0, 0]
        true_label = [1, 0, 0, 0, 0]

        for index in range(0, len(answer)):
            answer[index] = 1
            precision = 1 / (1 + index)
            print(precision)
            assert dt.precision(answer, true_label) == precision

    def test_recall_calculation(self):
        """Test recall calculation.

        Asserts:
            Recall matches for all true labels.
        """

        answer = [0, 0, 0, 0, 0]
        true_label = [1, 1, 1, 1, 1]
        total_count = len(answer)

        for index in range(0, len(answer)):
            answer[index] = 1
            recall = (index + 1) / ((index + 1) + (total_count - (index + 1)))

            assert dt.recall(answer, true_label) == recall

    def test_accuracy_calculation(self):
        """Test accuracy calculation.

        Asserts:
            Accuracy matches for all true labels.
        """

        answer = [0, 0, 0, 0, 0]
        true_label = [1, 1, 1, 1, 1]
        total_count = len(answer)

        for index in range(0, len(answer)):
            answer[index] = 1
            accuracy = dt.accuracy(answer, true_label)

            assert accuracy == ((index + 1) / total_count)

class DecisionTreePart2Tests(unittest.TestCase):
    """Tests for Decision Tree Learning.

    Attributes:
        restaurant (dict): represents restaurant data set.
        dataset (data): training data used in testing.
        train_features: training features from dataset.
        train_classes: training classes from dataset.
    """

    def setUp(self):
        """Set up test data.
        """
        data_dir = './data/'
        self.restaurant = {'restaurants': [0] * 6 + [1] * 6,
                           'split_patrons': [[0, 0],
                                             [1, 1, 1, 1],
                                             [1, 1, 0, 0, 0, 0]],
                           'split_food_type': [[0, 1],
                                               [0, 1],
                                               [0, 0, 1, 1],
                                               [0, 0, 1, 1]]}

        self.dataset = dt.load_csv(data_dir +'part23_data.csv')
        self.train_features, self.train_classes = self.dataset
        self.time_limit = 5*60*1000

    def test_gini_impurity_max(self):
        """Test maximum gini impurity.

        Asserts:
            gini impurity is 0.5.
        """

        gini_impurity = dt.gini_impurity([1, 1, 1, 0, 0, 0])

        assert  .500 == round(gini_impurity, 3)

    def test_gini_impurity_min(self):
        """Test minimum gini impurity.

        Asserts:
            entropy is 0.
        """

        gini_impurity = dt.gini_impurity([1, 1, 1, 1, 1, 1])

        assert 0 == round(gini_impurity, 3)

    def test_gini_impurity(self):
        """Test gini impurity.

        Asserts:
            gini impurity is matched as expected.
        """

        gini_impurity = dt.gini_impurity([1, 1, 0, 0, 0, 0])

        assert round(4. / 9., 3) == round(gini_impurity, 3)

    def test_gini_gain_max(self):
        """Test maximum gini gain.

        Asserts:
            gini gain is 0.5.
        """

        gini_gain = dt.gini_gain([1, 1, 1, 0, 0, 0],
                                 [[1, 1, 1], [0, 0, 0]])

        assert .500 == round(gini_gain, 3)

    def test_gini_gain(self):
        """Test gini gain.

        Asserts:
            gini gain is within acceptable bounds
        """

        gini_gain = dt.gini_gain([1, 1, 1, 0, 0, 0],
                                 [[1, 1, 0], [1, 0, 0]])

        assert 0.056 == round(gini_gain, 3)

    def test_gini_gain_restaurant_patrons(self):
        """Test gini gain using restaurant patrons.

        Asserts:
            gini gain rounded to 3 decimal places matches as expected.
        """

        gain_patrons = dt.gini_gain(
            self.restaurant['restaurants'],
            self.restaurant['split_patrons'])

        assert round(gain_patrons, 3) == 0.278

    def test_gini_gain_restaurant_type(self):
        """Test gini gain using restaurant food type.

        Asserts:
            gini gain is 0.
        """

        gain_type = round(dt.gini_gain(
            self.restaurant['restaurants'],
            self.restaurant['split_food_type']), 2)

        assert gain_type == 0.00

    def test_decision_tree_all_data(self):
        """Test decision tree classifies all data correctly.

        Asserts:
            classification is 100% correct.
        """
        tree = dt.DecisionTree()
        self.start_time = curr_time_millis()
        tree.fit(self.train_features, self.train_classes)
        if curr_time_millis()-self.start_time >= self.time_limit:
                raise Exception("Took too long to fit")
        output = tree.classify(self.train_features)
    

        assert (output == self.train_classes).all()

class DecisionTreePart3Tests(unittest.TestCase):
    """Tests for RandomForest Decision Tree Learning.

    Attributes:
        restaurant (dict): represents restaurant data set.
        dataset (data): training data used in testing.
        train_features: training features from dataset.
        train_classes: training classes from dataset.
    """

    def setUp(self):
        """Set up test data.
        """
        data_dir = './data/'
        self.comp_bin_dataset = dt.load_csv(data_dir + 'mod_complex_binary.csv')
        self.train_features_full, self.train_classes_full = self.comp_bin_dataset

        shuffle_train_indices = np.arange(self.comp_bin_dataset[0].shape[0])
        np.random.shuffle(shuffle_train_indices)
        train_indices = shuffle_train_indices[:int(self.comp_bin_dataset[0].shape[0]*0.8)]
        test_indices = shuffle_train_indices[int(self.comp_bin_dataset[0].shape[0]*0.8):]

        self.train_features_split =  self.train_features_full[train_indices]
        self.train_classes_split = self.train_classes_full[train_indices]

        self.test_features_split =  self.train_features_full[test_indices]
        self.test_classes_split = self.train_classes_full[test_indices]

        self.rfb = None
        self.time_limit = 5*60*1000

    def test_binary_random_forest_complete(self):
        """Test random forest on the complete binary data. 

        Asserts:
            Accuracy is greater than 80%.
        """     
        self.rfb = dt.RandomForest(200, 3, .2, .3)
        self.start_time = curr_time_millis()
        self.rfb.fit(self.train_features_full, self.train_classes_full)
        
        time_elapsed = curr_time_millis()-self.start_time 

        if time_elapsed >= self.time_limit:
                raise Exception("Took too long to fit")
        
        
        train_votes = self.rfb.classify(self.train_features_full)
        result = (float(sum([1 if vote == self.train_classes_full[i] else 0 for i,
                                    vote in enumerate(train_votes)])) / float(len(self.train_classes_full))) 

        
        assert result > .80
        
    def test_binary_random_forest_split(self):
        """Test random forest on the split binary data. 

        Asserts:
            Accuracy is greater than 75%.
        """     
        self.rfb = dt.RandomForest(200, 3, .2, .3)
        self.start_time = curr_time_millis()
        self.rfb.fit(self.train_features_split, self.train_classes_split)
        if curr_time_millis()-self.start_time >= self.time_limit:
                raise Exception("Took too long to fit")
        test_votes = self.rfb.classify(self.test_features_split)
        result = (float(sum([1 if vote == self.test_classes_split[i] else 0 for i,
                                    vote in enumerate(test_votes)])) / float(len(self.test_classes_split)))    
        self.rfb.fit(self.train_features_split, self.train_classes_split)
   
        assert result > .75

class VectorizationWarmUpTests(unittest.TestCase):
    """Tests the Warm Up exercises for Vectorization.

    Attributes:
        vector (Vectorization): provides vectorization test functions.
        data: vectorize test data.
    """

    def setUp(self):
        """Set up test data.
        """
        data_dir = './data/'
        self.vector = dt.Vectorization()
        self.data = dt.load_csv(data_dir + 'vectorize.csv', 1)

    def test_vectorized_loops(self):
        """Test if vectorized arithmetic.

        Asserts:
            vectorized arithmetic matches looped version.
        """

        real_answer = self.vector.non_vectorized_loops(self.data)
        my_answer = self.vector.vectorized_loops(self.data)

        assert np.array_equal(real_answer, my_answer)

    def test_vectorized_slice(self):
        """Test if vectorized slicing.

        Asserts:
            vectorized slicing matches looped version.
        """

        real_sum, real_sum_index = self.vector.non_vectorized_slice(self.data)
        my_sum, my_sum_index = self.vector.vectorized_slice(self.data)

        assert real_sum == my_sum
        assert real_sum_index == my_sum_index

    def test_vectorized_flatten(self):
        """Test if vectorized flattening.

        Asserts:
            vectorized flattening matches looped version.
        """

        answer_unique = sorted(self.vector.non_vectorized_flatten(self.data))
        my_unique = sorted(self.vector.vectorized_flatten(self.data))

        assert np.array_equal(answer_unique, my_unique)

    def test_vectorized_glue(self):
        """Test if vectorized flattening.

        Asserts:
            vectorized array matches looped version.
        """
        answer_glue = self.vector.non_vectorized_glue(self.data[:,0:-1], self.data[:,-1], 'c')
        my_glue = self.vector.vectorized_glue(self.data[:,0:-1], self.data[:,-1], 'c')

        assert np.array_equal(answer_glue, my_glue)

        answer_glue = self.vector.non_vectorized_glue(self.data[0:-1,:], self.data[-1,:], 'r')
        my_glue = self.vector.vectorized_glue(self.data[0:-1,:], self.data[-1,:], 'r')

        assert np.array_equal(answer_glue, my_glue)

    def test_vectorized_mask(self):
        """Test if vectorized flattening.

        Asserts:
            vectorized mask matches looped version.
        """
        val = 99.
        answer_mask = self.vector.non_vectorized_mask(self.data, val)
        my_mask = self.vector.vectorized_mask(self.data, val)

        assert np.array_equal(answer_mask, my_mask)

    def test_vectorized_loops_time(self):
        """Test if vectorized arithmetic speed.

        Asserts:
            vectorized arithmetic is faster than expected gradescope time.
        """

        start_time = time.time() * 1000
        self.vector.vectorized_loops(self.data)
        end_time = time.time() * 1000

        assert (end_time - start_time) <= 0.09

    def test_vectorized_slice_time(self):
        """Test if vectorized slicing speed.

        Asserts:
            vectorized slicing is faster than expected gradescope time.
        """

        start_time = time.time() * 1000
        self.vector.vectorized_slice(self.data)
        end_time = time.time() * 1000

        assert (end_time - start_time) <= 0.1

    def test_vectorized_flatten_time(self):
        """Test if vectorized flatten speed.

        Asserts:
            vectorized flatten is faster than expected gradescope time.
        """
        start_time = time.time() * 1000
        self.vector.vectorized_flatten(self.data)
        end_time = time.time()  * 1000

        assert (end_time - start_time) <= 10.0


    def test_vectorized_glue_time(self):
        """Test vectorized glue speed.

        Asserts:
            vectorized flatten is faster than expected gradescope time.
        """

        start_time = time.time() * 1000
        self.vector.vectorized_glue(self.data[0:-1,:], self.data[-1,:], 'r')
        end_time = time.time()  * 1000

        assert (end_time - start_time) <= 3.0

    def test_vectorized_mask_time(self):
        """Test if vectorized flatten speed.

        Asserts:
            vectorized flatten is faster than expected gradescope time.
        """
        val = 100.
        start_time = time.time() * 1000
        self.vector.vectorized_mask(self.data, val)
        end_time = time.time()  * 1000

        assert (end_time - start_time) <= 4.0

class NameTests(unittest.TestCase):
    def setUp(self):
        """Set up test data.
        """
        self.to_compare = "George P. Burdell"


    def test_name(self):
        """Test if vectorized arithmetic.

        Asserts:
            Non Matching Name
        """

        self.name = dt.return_your_name()
        assert self.name != None
        assert self.name != self.to_compare

if __name__ == '__main__':
    unittest.main()

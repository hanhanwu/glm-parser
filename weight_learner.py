# -*- coding: utf-8 -*-
import feature_set, data_set, feature_vector
import eisner

class WeightLearner():
    """
    Learns the weight of the features using maximum perceptron algorithm
    """
    
    def __init__(self):
        self.MAX_ITERATE = 20
        pass
    
    def learn_weight_source(self, section_set=None, source=None):
        """
        Given the path and specified sections, 
        for each dep_tree in the source
        learn the weight for each feature in that feature set
        
        :param section_set: see section_set in DataSet
        :type section_set: list(int/tuple)
        
        :param source: see data_path in DataSet
        :type source: str
        """
        dataset = data_set.DataSet(section_set, source)
        # should be while
        while dataset.has_next_data():
            self.learn_weight(dataset.get_next_data())    
            
    def learn_weight(self,dep_tree):
        """
        Given one dependency tree, the function learns the weight 
        for each feature in the feature set
        
        :param dep_tree: a class contains the information about the
        word_list, pos_list, edge and the information of edge type
        :type dep_tree: DependencyTree
        
        :return: updated feature set
        :rtype: FeatureSet
        """
        print "learn_weight"
        fset = feature_set.FeatureSet('weight.db',dep_tree)
        word_list = dep_tree.get_word_list()
        gold_edge_set = \
            set([(head_index,dep_index) for head_index,dep_index,_ in dep_tree.get_edge_list()])
	print "gold set", gold_edge_set
        for i in range(self.MAX_ITERATE):
            _, current_edge_set = \
               eisner.EisnerParser(word_list).parse(fset.get_edge_score)
            # guarantee to converge ???????
            # will the eisner calculate the different trees?
            if current_edge_set == gold_edge_set:
                break
            
            # calculate the global score
            # assume the length of each local vector in the same sentanse is the same
            # the truth_global_vector will change because of the change in weights
            current_global_vector = self.get_global_vector(current_edge_set, dep_tree, fset)
	    print current_edge_set
            gold_global_vector = self.get_global_vector(gold_edge_set, dep_tree, fset)
            gold_global_vector.eliminate(current_global_vector)
	    # print gold_global_vector.feature_dict
            fset.update_weight_vector(gold_global_vector)
        fset.close()
        return 

    def get_global_vector(self, edge_set, dep_tree, fset):
        """
        Calculate the global vector with the current weight, the order of the feature
        score is the same order as the feature set

        :param edge_set: the set of edges represented as tuples
        :type: list(tuple(integer, integer))
        :param dep_tree: the DependencyTree which contains the word list
        :type: DependencyTree
        :param fset: the current using feature set
        :type fset: FeatureSet
        
        :return: The global vector of the sentence with the current weight
        :rtype: list
        """
        global_vector = feature_vector.FeatureVector()
        for head_index,dep_index in edge_set:
            local_vector = fset.get_local_vector(head_index,dep_index)
            global_vector.aggregate(local_vector)
        return global_vector

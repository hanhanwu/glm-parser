import logging
from hvector._mycollections import mydefaultdict
from hvector.mydouble import mydouble
from weight import weight_vector
from data import data_pool
import perceptron
import debug.debug
import sys,os,shutil,re
from os.path import isfile, join, isdir
from pyspark import SparkContext



logging.basicConfig(filename='glm_parser.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

class ParallelPerceptronLearner():

    def __init__(self, w_vector=None, max_iter=1):
        """
        :param w_vector: A global weight vector instance that stores
         the weight value (float)
        :param max_iter: Maximum iterations for training the weight vector
         Could be overridden by parameter max_iter in the method
        :return: None
        """
        logging.debug("Initialize ParallelPerceptronLearner ... ")
        self.w_vector = w_vector
        return
    def parallel_learn(self, max_iter=-1, dir_name=None, shards=1, fgen=None,parser=None,format_path=None,learner=None,sc=None,d_filename=None):
        '''
        This is the function which does distributed training using Spark


        :param max_iter: iterations for training the weight vector
        :param dir_name: the output directory storing the sharded data
        :param fgen: feature generator
        :param parser: parser for generating parse tree
        '''

        def create_dp(textString,fgen,format,sign):
            dp = data_pool.DataPool(textString=textString[1],fgen=fgen,format_list=format,comment_sign=sign)
            return dp


        def get_sent_num(dp):
            return dp.get_sent_num()

        fformat = open(format_path)
        format_list = []
        comment_sign = ''
        remaining_field_names = 0
        for line in fformat:
            format_line = line.strip().split()
            if remaining_field_names > 0:
                format_list.append(line.strip())
                remaining_field_names -= 1

            if format_line[0] == "field_names:":
                remaining_field_names = int(format_line[1])

            if format_line[0] == "comment_sign:":
                comment_sign = format_line[1]

        fformat.close()

        #nodes_num = "local[%d]"%shards
        #sc = SparkContext(master=nodes_num)
        train_files= sc.wholeTextFiles(dir_name,minPartitions=10).cache()
        dp = train_files.map(lambda t: create_dp(t,fgen,format_list,comment_sign)).cache()
        if learner.__class__.__name__== "AveragePerceptronLearner":
            fv = {}
            total_sent = dp.map(get_sent_num).sum()
            c = total_sent*max_iter
            for round in range(max_iter):
                #mapper: computer the weight vector in each shard using avg_perc_train
                print "keys: %d"%len(fv.keys())
                feat_vec_list = dp.flatMap(lambda t: learner.parallel_learn(t,fv,parser))
                #reducer: combine the weight vectors from each shard
                feat_vec_list = feat_vec_list.combineByKey(lambda value: (value[0], value[1], 1),
                                 lambda x, value: (x[0] + value[0], x[1] + value[1], x[2] + 1),
                                 lambda x, y: (x[0] + y[0], x[1] + y[1], x[2]+y[2])).collect()
                fv = {}
                for (feat, (a,b,c)) in feat_vec_list:
                    fv[feat] = (float(a)/float(c),b)

            self.w_vector.clear()
            for feat in fv.keys():
                self.w_vector[feat] = fv[feat][1]/c
        if learner.__class__.__name__== "PerceptronLearner":
            fv = {}
            for round in range(max_iter):
                print "round: %d"%round
                #mapper: computer the weight vector in each shard using avg_perc_train
                feat_vec_list = dp.flatMap(lambda t: learner.parallel_learn(t,fv,parser))
                #reducer: combine the weight vectors from each shard
                feat_vec_list = feat_vec_list.combineByKey(lambda value: (value, 1),
                                 lambda x, value: (x[0] + value, x[1] + 1),
                                 lambda x, y: (x[0] + y[0], x[1] + y[1])).collect()
                #fv = feat_vec_list.map(lambda (label, (value_sum, count)): (label, value_sum / count)).collectAsMap()

                fv = {}
                for (feat,(a,b)) in feat_vec_list:
                    fv[feat] = float(a)/float(b)
            self.w_vector.clear()
            self.w_vector.iadd(fv)
            #dump the weight vector
            #d_filename: change the full path to hdfs file name
        if d_filename is not None:
            self.w_vector.dump(d_filename + "_Iter_%d.db"%max_iter)

from __future__ import division
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
import gzip # use compressed data files
import copy, operator, optparse
from feature import pos_fgen

def get_maxvalue(viterbi_dict):
    maxvalue = (None, None) # maxvalue has tuple (tag, value)
    for tag in viterbi_dict.keys():
        value = viterbi_dict[tag] # value is (score, backpointer)
        if maxvalue[1] is None:
            maxvalue = (tag, value[0])
        elif maxvalue[1] < value[0]:
            maxvalue = (tag, value[0])
        else:
            pass # no change to maxvalue
    if maxvalue[1] is None:
        raise ValueError("max value tag for this word is None")
    return maxvalue

def perc_test(feat_vec, labeled_list, tagset, default_tag):
    output = []
    labels = copy.deepcopy(labeled_list)
    # add in the start and end buffers for the context
    labels.insert(0, '_B_-1')
    labels.insert(0, '_B_-2') # first two 'words' are B_-2 B_-1
    labels.append('B_+1')
    labels.append('B_+2') # last two 'words' are B_+1 B_+2

    # size of the viterbi data structure
    N = len(labels)

    # Set up the data structure for viterbi search
    viterbi = {}
    for i in range(0, N):
        viterbi[i] = {} # each column contains for each tag: a (value, backpointer) tuple

    # We do not tag the first two and last two words
    # since we added B_-2, B_-1, B_+1 and B_+2 as buffer words 
    viterbi[0]['B_-2'] = (0.0, '')
    viterbi[1]['B_-1'] = (0.0, 'B_-2')
    # find the value of best_tag for each word i in the input
    # feat_index = 0
    pos_feat = pos_fgen.Pos_feat_gen(labels)
    for i in range(2, N-2):
        word = labels[i]
        # if len(feats) == 0:
        #     print >>sys.stderr, " ".join(labels), " ".join(feat_list), "\n"
        #     raise ValueError("features do not align with input sentence")

        #fields = labels[i].split()
        #(word, postag) = (fields[0], fields[1])
        found_tag = False
        for tag in tagset:
            #has_bigram_feat = False
            # weight = 0.0
            # # sum up the weights for all features except the bigram features
            # for feat in feats:
            #     #if feat == 'B': has_bigram_feat = True
            #     if (feat, tag) in feat_vec:
            #         weight += feat_vec[feat, tag]
            #         print >>sys.stderr, "feat:", feat, "tag:", tag, "weight:", feat_vec[feat, tag]
            prev_list = []
            for prev_tag in viterbi[i-1]:
                feats = []
                (prev_value, prev_backpointer) = viterbi[i-1][prev_tag]
                pos_feat.get_pos_feature(feats,i,prev_tag,prev_backpointer)
                weight = 0.0
                # sum up the weights for all features except the bigram features
                for feat in feats:
                #if feat == 'B': has_bigram_feat = True
                    if (feat, tag) in feat_vec:
                        weight += feat_vec[feat, tag]
            #         print >>sys.stderr, "feat:", feat, "tag:", tag, "weight:", feat_vec[feat, tag]
                prev_tag_weight = weight
                # if has_bigram_feat:
                #     prev_tag_feat = "B:" + prev_tag
                #     if (prev_tag_feat, tag) in feat_vec:
                #         prev_tag_weight += feat_vec[prev_tag_feat, tag]
                prev_list.append( (prev_tag_weight + prev_value, prev_tag) )
            (best_weight, backpointer) = sorted(prev_list, key=operator.itemgetter(0), reverse=True)[0]
            #print >>sys.stderr, "best_weight:", best_weight, "backpointer:", backpointer
            if best_weight != 0.0:
                viterbi[i][tag] = (best_weight, backpointer)
                found_tag = True
        if found_tag is False:
            viterbi[i][default_tag] = (0.0, default_tag)


    # recover the best sequence using backpointers
    maxvalue = get_maxvalue(viterbi[N-3])
    best_tag = maxvalue[0]
    for i in range(N-3, 1, -1):
        output.insert(0,best_tag)
        (value, backpointer) = viterbi[i][best_tag]
        best_tag = backpointer

    return output


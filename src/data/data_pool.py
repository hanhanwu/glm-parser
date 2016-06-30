# -*- coding: utf-8 -*-
from __future__ import division
import os, re
from sentence import Sentence
import logging
import re
from data_prep import *
from file_io import *


logging.basicConfig(filename='glm_parser.log',
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

def getClassFromModule(attr_name, module_path, module_name,
                          silent=False):
    """
    This procedure is used by argument processing. It returns a class object
    given a module path and attribute name.

    Basically what it does is to find the module using module path, and then
    iterate through all objects inside that module's namespace (including those
    introduced by import statements). For each object in the name space, this
    function would then check existence of attr_name, i.e. the desired interface
    name (could be any attribute fetch-able by getattr() built-in). If and only if
    one such desired interface exists then the object containing that interface
    would be returned, which is mostly a class, but could theoretically be anything
    object instance with a specified attribute.

    If import error happens (i.e. module could not be found/imported), or there is/are
    no/more than one interface(s) existing, then exception will be raised.

    :param attr_name: The name of the desired interface
    :type attr_name: str
    :param module_path: The path of the module. Relative to src/
    :type module_path: str
    :param module_name: Name of the module e.g. file name
    :type module_name: str
    :param silent: Whether to report existences of interface objects
    :return: class object
    """
    # Load module by import statement (mimic using __import__)
    # We first load the module (i.e. directory) and then fetch its attribute (i.e. py file)
    module_obj = getattr(__import__(module_path + '.' + module_name,
                                    globals(),   # Current global
                                    locals(),    # Current local
                                    [],          # I don't know what is this
                                    -1), module_name)
    intf_class_count = 0
    intf_class_name = None
    for obj_name in dir(module_obj):
        if hasattr(getattr(module_obj, obj_name), attr_name):
            intf_class_count += 1
            intf_class_name = obj_name
            if silent is False:
                print("Interface object %s detected\n\t with interface %s" %
                      (obj_name, attr_name))
    if intf_class_count < 1:
        raise ImportError("Interface object with attribute %s not found" % (attr_name, ))
    elif intf_class_count > 1:
        raise ImportError("Could not resolve arbitrary on attribute %s" % (attr_name, ))
    else:
        class_obj = getattr(module_obj, intf_class_name)
        return class_obj

class DataPool():
    """
    Data object that holds all sentences (dependency trees) and provides
    interface for loading data from the disk and retrieving them using an
    index.

    Data are classified into sections when stored in the disk, but we
    do not preserve such structural information, and all sentences
    will be loaded and "flattened" to be held in a single list.

    The instance maintains a current_index variable, which is used to
    locate the last sentence object we have read. Calling get_next()
    method will increase this by 1, and calling has_next() will test
    this index against the total number. The value of the index is
    persistent during get_next() and has_next() calls, and will only
    be reset to initial value -1 when reset() is called (manually or
    during init).
    """
    def __init__(   self,
                    section_regex = '',
                    data_path     = "./penn-wsj-deps/",
                    fgen          = None,
                    format_path   = None,
                    textString    = None,
                    format_list   = None,
                    comment_sign  = '',
                    prep_path     = 'data/prep/',
                    shardNum      = 1,
                    sc            = None,
                    hadoop        = False):

        """
        Initialize the Data set

        :param section_regex: the sections to be used.
        A regular expression that indicates which sections to be used e.g.
        (0[0-9])|(1[0-9])|(2[0-1])/.*tab
        :type section_regex: str

        :param data_path: the relative or absolute path to the 'penn-wsj-deps' folder
        (including "penn-wsj-deps")
        :type data_path: str

        :param format_path: the file that describes the file format for the type of data
        :type format_path: str
        """
        if isinstance(fgen, basestring):
            self.fgen = getClassFromModule('get_local_vector', 'feature', fgen)
        else:
            self.fgen = fgen
        self.hadoop   = hadoop
        self.sc       = sc
        self.shardNum = shardNum
        self.reset_all()

        if textString is not None:
            self.load_stringtext(textString,format_list,comment_sign)
        else:
            self.data_path     = data_path
            self.section_regex = section_regex
            self.prep_path     = prep_path
            self.dataPrep      = DataPrep(dataPath     = self.data_path,
                                          dataRegex    = self.section_regex,
                                          shardNum     = self.shardNum,
                                          targetPath   = self.prep_path,
                                          sparkContext = sc)
            self.load(format_path, sc)
        return

    def loadedPath(self):
        if self.dataPrep:
            if self.hadoop == True:
                return self.dataPrep.hadoopPath()
            else:
                return self.dataPrep.localPath()
        else:
            raise RuntimeError("DATAPOOL [ERROR]: Data has not been loaded by DataPrep, cannot retrieve data path.")
        return

    def load_stringtext(self, textString, format_list, comment_sign):
        lines = textString.splitlines()
        column_list = {}
        for field in format_list:
            if not(field.isdigit()):
                column_list[field] = []

        length = len(format_list)

        for line in lines:
            entity = line.split()
            if len(entity) == length and entity[0] != comment_sign:
                for i in range(length):
                    if not(format_list[i].isdigit()):
                        column_list[format_list[i]].append(str(entity[i].encode('utf-8')))
            else:
                if not(format_list[0].isdigit()) and column_list[format_list[0]] != []:
                    sent = Sentence(column_list, format_list, self.fgen)
                    self.data_list.append(sent)

                column_list = {}

                for field in format_list:
                    if not (field.isdigit()):
                        column_list[field] = []

    def reset_all(self):
        """
        Reset the index variables and the data list.

        Restores the instance to a state when no sentence has been read
        """
        self.reset_index()
        self.data_list = []

        return

    def reset_index(self):
        """
        Reset the index variable to the very beginning of
        sentence list
        """
        self.current_index = -1

    def has_next_data(self):
        """
        Returns True if there is still sentence not read. This call
        does not advence data pointer. Call to get_next_data() will
        do the job.

        :return: False if we have reaches the end of data_list
                 True otherwise
        """
        i = self.current_index + 1
        if i >= 0 and i < len(self.data_list):
            return True
        else:
            return False

    def get_next_data(self):
        """
        Return the next sentence object, which is previously read
        from disk files.

        This method does not perform index checking, so please make sure
        the internal index is valid by calling has_next_data(), or an exception
        will be raise (which would be definitely not what you want)
        """
        if(self.has_next_data()):
            self.current_index += 1
            # Logging how many entries we have supplied
            if self.current_index % 1000 == 0:
                logging.debug("Data finishing %.2f%% ..." %
                             (100 * self.current_index/len(self.data_list), ))

            return self.data_list[self.current_index]
        raise IndexError("Run out of data while calling get_next_data()")

    def get_format_list(self):
        if self.field_name_list == []:
            raise RuntimeError("DATAPOOL [ERROR]: format_list empty")
        return self.field_name_list

    def get_comment_sign(self):
        return self.comment_sign

    def load(self, formatPath, sparkContext=None):
        """
        For each section in the initializer, iterate through all files
        under that section directory, and load the content of each
        individual file into the class instance.

        This method should be called after section regex has been initalized
        and before any get_data method is called.
        """
        logging.debug("Loading data...")
        print("Loading data...")

        # Load format file
        print("Loading dataFormat from: " + formatPath)
        if self.hadoop == True:
            fformat = fileRead(formatPath, sparkContext=sparkContext)
        else:
            fformat = open(formatPath)

        self.field_name_list = []
        self.comment_sign = ''

        remaining_field_names = 0
        for line in fformat:
            format_line = line.strip().split()

            if remaining_field_names > 0:
                self.field_name_list.append(line.strip())
                remaining_field_names -= 1

            if format_line[0] == "field_names:":
                remaining_field_names = int(format_line[1])

            if format_line[0] == "comment_sign:":
                self.comment_sign = format_line[1]

        if self.field_name_list == []:
            raise RuntimeError("DATAPOOL [ERROR]: format file read failure")

        if self.hadoop == False:
            fformat.close()

        # Load data
        if self.hadoop == True:
            self.dataPrep.loadHadoop()
        else:
            self.dataPrep.loadLocal()

        # Add data to data_list
        # If using yarn mode, local data will not be loaded
        if self.hadoop == False:
            for dirName, subdirList, fileList in os.walk(self.dataPrep.localPath()):
                for file_name in fileList:
                    file_path = "%s/%s" % ( str(dirName), str(file_name) )
                    self.data_list += self.get_data_list(file_path)
        else:
            aRdd = sparkContext.textFile(self.dataPrep.hadoopPath()).cache()
            tmp  = aRdd.collect()
            tmpStr = ''.join(str(e)+"\n" for e in tmp)
            self.load_stringtext(textString  = tmpStr,
                                format_list  = self.field_name_list,
                                comment_sign = self.comment_sign)
        return


    def get_data_list(self, file_path):
        """
        Form the DependencyTree list from the specified file.

        :param file_path: the path to the data file
        :type file_path: str

        :return: a list of DependencyTree in the specified file
        :rtype: list(Sentence)
        """

        f = open(file_path)
        data_list = []

        column_list = {}

        for field in self.field_name_list:
            if not(field.isdigit()):
                column_list[field] = []

        length = len(self.field_name_list)

        for entity in f:
            entity = entity[:-1].split()
            if len(entity) == length and entity[0] != self.comment_sign:
                for i in range(length):
                    if not(self.field_name_list[i].isdigit()):
                        column_list[self.field_name_list[i]].append(entity[i])

            else:
                # Prevent any non-mature (i.e. trivial) sentence structure
                if not(self.field_name_list[0].isdigit()) and column_list[self.field_name_list[0]] != []:

                    # Add "ROOT" for word and pos here
                    sent = Sentence(column_list, self.field_name_list, self.fgen)
                    data_list.append(sent)

                column_list = {}

                for field in self.field_name_list:
                    if not (field.isdigit()):
                        column_list[field] = []

        f.close()

        return data_list

    def get_sent_num(self):
        return len(self.data_list)

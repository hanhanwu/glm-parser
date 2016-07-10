import logging
import sys
import os
import re
import os.path


def fileReadHDFS(filePath=None, sparkContext=None):
    if filePath is None:
        raise ValueError("FILEIO [ERROR]: Reading file not specified")

    # Initialising
    if sparkContext is None:
        raise RuntimeError('FILEIO [ERROR]: SparkContext not initialised')
    sc = sparkContext

    aRdd = sc.textFile(filePath).cache()
    aRdd = aRdd.map(lambda x: str(x)).cache()
    fileContent = aRdd.collect()

    return fileContent


def fileWriteHDFS(filePath=None, contents=None, sparkContext=None):
    '''
    Acceptable contents:
        list(array) of data
    '''
    if filePath is None:
        raise ValueError("FILEIO [ERROR]: Saving file not specified")
    if not isinstance(contents, list):
        raise ValueError("FILEIO [ERROR]: Contents to be saved should be a list(an array)")
    # Initialising
    if sparkContext is None:
        raise RuntimeError('FILEIO [ERROR]: SparkContext not initialised')
    sc = sparkContext

    try:
        aRdd = sc.parallelize(contents, 1).cache()
        aRdd.coalesce(1, True).cache()
        aRdd.saveAsTextFile(filePath)
    except:
        raise RuntimeError('FILEIO [ERROR]: Unable to save file to HDFS: ' + filePath)

    return filePath + "/part-00000"


def fileRead(filePath=None, sparkContext=None):
    if filePath is None:
        raise ValueError("FILEIO [ERROR]: File not specified")
    if (filePath[:7] == "file://"):
        try:
            contents = []
            f = open(filePath[7:])
            for line in f:
                contents.append(line.rstrip('\n'))
            return contents
        except:
            raise RuntimeError('FILEIO [ERROR]: Unable to read from local directory: ' + filePath)
    return fileReadHDFS(filePath=filePath, sparkContext=sparkContext)


def fileWrite(filePath=None, contents=None, sparkContext=None):
    '''
    Acceptable contents:
        list(array) of data
    '''
    if filePath is None:
        raise ValueError("FILEIO [ERROR]: saving path not specified")
    if not isinstance(contents, list):
        raise ValueError("FILEIO [ERROR]: Contents to be saved should be a list(an array)")
    if (filePath[:7] == "file://"):
        try:
            f = open(filePath[7:], "w")
            for line in contents:
                f.write(line + "\n")
            return filePath
        except:
            raise RuntimeError('FILEIO [ERROR]: Unable to save to local directory: ' + filePath)
    return fileWriteHDFS(filePath=filePath, contents=contents, sparkContext=sparkContext)

from datetime import datetime
from collections import OrderedDict
import pathlib
"""
Configuration file for Text Guide.
How to use:
Define folder containing data file, filename of the data file, name of the column in the data file containing
unstructured textual data, select the truncation method and the name of the file (dictionary) containing
pairs of "key: value" where key == important token, and value == value of feature importance. Select the sorting
method for the important token dictionary.
"""


class ExperimentConfig(OrderedDict):

    def __init__(self):
        super().__init__()
        self.start_time = datetime.utcnow()
        self.config_name = f"{pathlib.Path(__file__).stem}"
        self.df = None

        # specify files and names
        self.data_folder = 'data'
        self.filename = "dmoz_100_instances.ftr"
        self.text_column = "Text"
        self.nrows = 7   # the number of instances to convert. The Whole file dmoz_100_instances.ftr has 100 instances.
        self.truncation_method = 'text_guide'   # or 'beginning'
        self.feature_importance_file = f"dmoz_30_1500_sITFL.p"      # the attached file was obtained from a BoW model
        # and a gradient boosting classifier. For different data, different file is needed.
        self.feature_importance_sort = 'descending'

        # specify other Text Guide parameters
        self.desired_length = 300       # the desired length of the new text instance.
        self.one_side_neighbours = 3    # number of tokens surrounding the important token to be used for creating the
        # pseudo sentence
        self.beg_part = 0.1     # the part of the beginning of the original text instance to be used by Text Guide to
        # create the new text instance
        self.end_part = 0.2     # the part of the ending of the original text instance to be used by Text Guide to
        # create the new text instance
        self.over_length = 1    # if set to 1, Text Guide will be used for all tex instances. Text Guide will be used
        # only for instances with length greater than desired_length*over_length tokens.
        self.number_of_important_token_occurrences = 1   # number of important token occurrences to be used by
        # Text Guide to create pseudo sentences.
        self.fill_up_to_limit = True   # some text instances created by Text Guide will be shorter than the
        # desired_length. Set to "True" if you wish to fill the remaining space by initial tokens of the original text
        # instance

        self.cpu_threads = 14   # number of cpu threads to be used by Dask when computing Text Guide instances

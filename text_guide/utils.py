import random
import numpy as np
import os
import feather
import hashlib
import re
import time
import heapq
import pickle
import dask.dataframe as dd


def read_data(data_folder, filename, nrows, text_column, config):
    """
    Reads the dataframe with all data instances.
    :param data_folder: str, name of the folder containing source data
    :param filename: str, name of the file containing source data
    :param nrows: int, number of data instances sampled for analysis
    :param text_column: str, name of the column containing text instances in the data file
    :param config: an instance of ExperimentConfig class
    :return: pandas data frame, an instance of ExperimentConfig class
    """

    # define the file path
    filepath = f'./{data_folder}/{filename}'

    # read the data file
    df = feather.read_dataframe(filepath)

    # use only the defined part of the data set
    if nrows >= len(df):
        config.nrows = len(df)

    df = df.sample(n=config.nrows, random_state=17)

    # compute chech sum of the original text instances and add it to the data frame
    def compute_check_sum(x):
        x = hashlib.md5(x.encode("utf-8")).hexdigest()
        return x

    df['check_sum'] = df[text_column].map(lambda x: compute_check_sum(x))
    df = df.set_index('check_sum', drop=False)

    # store the data frame
    columns_to_return = [text_column, "check_sum"]
    config.df = df[columns_to_return]

    return config

        
def return_selected_window_of_tokens(config):
    """
    A function which enables pre-selection of a part of the original text instance.
    It implements the 'keep beginning tokens' naive method and the Text Guide method.
    :param config: an instance of config.ExperimentConfig(). It includes the original data set.
    :return: config: an instance of config.ExperimentConfig(). It includes the modified data set.
    """

    def create_pseudo_instance(token_list, full_tokenized_instance, desired_length, selected_words_list,
                               one_side_neighbours, number_of_important_token_occurrences):
        x = full_tokenized_instance
        # create a dictionary of position indexes of each token
        idx_dict = dict()
        for token in selected_words_list:
            index_list = [i for i, x in enumerate(x) if x == token]
            if len(index_list) != 0:
                idx_dict[token] = index_list

        # create pseudo sentences from the important token and surrounding tokens
        selected_neighbourhoods = list()
        for token in idx_dict:
            count = 0
            for token_index in idx_dict[token]:
                selected_neighbourhoods.append(x[(token_index - one_side_neighbours):
                                                (token_index + one_side_neighbours + 1)])
                count += 1
                if number_of_important_token_occurrences <= count:
                    break

        # create final text instance
        for pseudo_sentence in selected_neighbourhoods:
            token_list.extend(pseudo_sentence)
            # sometimes the resulting instance would be too long, so we are taking only first n pseudo sentences,
            # the order does matter as the first ones come from the more important tokens
            if len(token_list) >= desired_length - (2 * one_side_neighbours + 1):
                break

        # sometimes, the resulting text instance is not using the 510 token limit.
        if config.fill_up_to_limit:
            # If so, the new text instance is filled with first k tokens until the 510 token limit is reached, even if
            # this doubles some sentences
            if len(token_list) < desired_length:
                number_missing_tokens = desired_length - len(token_list)
                token_list.extend(x[:number_missing_tokens])
        return token_list

    def prepare_for_sentence_feature_importance(config):
        """
        A function which prepares a word list sorted according to selected method.
        :param config: a config.ExperimentConfig() instance
        :return: list of str (sITFL i.e., selected words used by the Text Guide method).
        """
        # if the configuration file didn't provide a sorting method, use 'descending'
        if not hasattr(config, 'feature_importance_sort'):
            config.feature_importance_sort = 'descending'

        # read the file with precomputed feature importance
        filepath = os.path.join(config.data_folder, config.feature_importance_file)
        with open(filepath, 'rb') as f:
            feature_importances = pickle.load(f)

        # get n most important features sorted according to the selected method
        important_features = dict()

        # this eliminates non-token features that were used by the ML Classifier
        for key in feature_importances:
            if key.find("f_") == -1:
                important_features[key] = feature_importances[key]

        if config.feature_importance_sort == 'descending':
            # this sorts the features by importance value
            feature_importances = {k: v for k, v in heapq.nlargest(4000, important_features.items(),
                                                                   key=lambda i: i[1])}

        elif config.feature_importance_sort == 'nsmallest':
            # this sorts the features by importance value
            feature_importances = {k: v for k, v in heapq.nsmallest(4000, important_features.items(),
                                                                    key=lambda i: i[1])}

        elif config.feature_importance_sort == 'random':
            # this shuffles the features randomly
            l = list(feature_importances.items())
            random.shuffle(l)
            feature_importances = dict(l)

        # return the sITFL i.e., sorted list of important feature tokens
        return list(feature_importances.keys())

    def text_guide(original_tokenized_instance: list, desired_length: int, selected_words_list: list,
                   one_side_neighbours: int, beg_part: float, end_part: float, over_length: float,
                   number_of_important_token_occurrences: int):
        """
        A function which implements the Text Guide text preselection method useful for long text classification.
        :param original_tokenized_instance: list of str. The tokenized text instance.
        :param desired_length: int. Defines the length of final text instance by the number of tokens.
        :param selected_words_list: list of str. List of tokens used a guides for selecting informative text parts.
        :param one_side_neighbours: int. Number of tokens to be taken as neighbours providing context from one side
         of the selected word.
        :param beg_part: float. len(x)*beg_part defines the number of tokens from the beggining of the original
        text instance to be used also in the final text instance.
        :param end_part: float. len(x)*end_part defines the number of tokens from the end of the original
        text instance to be used also in the final text instance.
        :param over_length: float. The Text Guide method will be applied only if a condition is met:
         instance_length < int(desired_length * over_length)
        :param number_of_important_token_occurrences: int. For example, if 1, for each token from selected_words_list
        only the first occurrence of that token in the original text instance will be used. If 2,
        two first occurrences and so on.
        :return: str. Modified text instance.
        """
        x = original_tokenized_instance
        instance_length = len(x)
        if instance_length < int(desired_length * over_length):
            return " ".join(x[:desired_length])
        else:
            # create the final text instance
            n_first_tokens_to_keep = int(desired_length * beg_part)
            first_part = x[:n_first_tokens_to_keep]

            n_last_tokens_to_keep = int(desired_length * end_part)
            if not n_last_tokens_to_keep == 0:
                ending_part = x[-n_last_tokens_to_keep:]
                remainder_x = x[n_first_tokens_to_keep:-n_last_tokens_to_keep]

            else:
                ending_part = list()
                remainder_x = x[n_first_tokens_to_keep:]

            first_part.extend(ending_part)

            x = remainder_x
            final_text = create_pseudo_instance(token_list=first_part,
                                                full_tokenized_instance=x,
                                                desired_length=desired_length,
                                                selected_words_list=selected_words_list,
                                                one_side_neighbours=one_side_neighbours,
                                                number_of_important_token_occurrences=
                                                number_of_important_token_occurrences)
            return " ".join(final_text)

    # tokenize all text instances
    tokenized_col_name = 'tokenized'
    config.df[tokenized_col_name] = config.df[config.text_column].map(lambda x: x.split(' '))

    # define the name of the column with new text instances
    new_text_column = "new_text"

    # read the desired token length of new text instances
    desired_length = config.desired_length

    if config.truncation_method == "beginning":
        print("New text instances are created according to the naive 'keep beginning tokens' method.")
        config.df[new_text_column] = config.df[tokenized_col_name].map(lambda x: " ".join(x[:desired_length]))

    elif config.truncation_method == "text_guide":
        print("New text instances are created according to Text Guide.")
        selected_words_list = prepare_for_sentence_feature_importance(config)

        # this method uses dask for improved performance. Adapt npartitions to the number of cores
        # available on your machine
        one_side_neighbours = config.one_side_neighbours
        number_of_important_token_occurrences = config.number_of_important_token_occurrences
        ddata = dd.from_pandas(config.df, npartitions=config.cpu_threads)
        beg_part = config.beg_part
        end_part = config.end_part
        over_length = config.over_length

        config.df[new_text_column] = ddata \
            .map_partitions(lambda df: df.apply((lambda row: text_guide(
                original_tokenized_instance=row[tokenized_col_name],
                desired_length=desired_length,
                selected_words_list=selected_words_list,
                one_side_neighbours=one_side_neighbours,
                beg_part=beg_part,
                end_part=end_part,
                over_length=over_length,
                number_of_important_token_occurrences=number_of_important_token_occurrences)), axis=1)) \
            .compute(scheduler='processes')

    config.df[config.text_column] = config.df[new_text_column]
    config.df.drop(tokenized_col_name, axis=1, inplace=True)
    config.df.drop(new_text_column, axis=1, inplace=True)
    return config
   
        



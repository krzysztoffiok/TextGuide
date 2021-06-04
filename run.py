from datetime import datetime
import pathlib
from datetime import datetime
import text_guide
import importlib.util
import time
from os import listdir
from os.path import isfile, join


if __name__ == '__main__':
    # iterating over configuration files found in ./configs
    config_path = './configs'
    config_file_names = sorted([f for f in listdir(config_path) if isfile(join(config_path, f))])
    experiment_start = datetime.utcnow()

    time_list = list()
    for config_file_name in config_file_names:
        spec = importlib.util.spec_from_file_location('configs', join(config_path, config_file_name))
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        config = config.ExperimentConfig()

        print(f"\nThe analyzed config file name is: {config.config_name}\n")
        # the experiment begins
        print("Reading source data.")
        config = text_guide.utils.read_data(data_folder=config.data_folder, filename=config.filename,
                                            nrows=config.nrows, text_column=config.text_column, config=config)

        # text guide
        stime = datetime.utcnow()
        config = text_guide.utils.return_selected_window_of_tokens(config=config)
        config.df.reset_index(drop=True).to_feather(f"./{config.data_folder}/{config.config_name}.ftr")
        time_list.append((config.config_name, round(((datetime.utcnow() - stime).total_seconds() / 60), 2)))
        print(f"Time list: {time_list}")

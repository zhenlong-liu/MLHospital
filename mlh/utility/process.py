import re
import os
import pandas as pd
import yaml
# extract train mean train variance
def extract_metrics(log_file):
    with open(log_file, 'r') as file:
        text = file.read()
    
    pattern = r"(Entropies|Loss):\s+(\w+):\s+([\d.]+)\s+(\w+):\s+([\d.]+)\s+(\w+):\s+([\d.]+)\s+(\w+):\s+([\d.]+)"
    #pattern =r"(Entropies|Loss):\s+train_mean: ([\d.]+)\s+train_variance: ([\d.]+)\s+test_mean: ([\d.]+)\s+test_variance: ([\d.]+)"
    matches = re.findall(pattern, text)
    metrics = {}
    for match in matches:
        metric_type = match[0]
        metric_name1 = match[1]
        value1 = float(match[2])
        metric_name2 = match[3]
        value2 = float(match[4])
        metric_name3 = match[5]
        value3 = float(match[6])
        metric_name4 = match[7]
        value4 = float(match[8])

        key1 = f"{metric_type}_{metric_name1}"
        key2 = f"{metric_type}_{metric_name2}"
        key3 = f"{metric_type}_{metric_name3}"
        key4 = f"{metric_type}_{metric_name4}"

        metrics[key1] = value1
        metrics[key2] = value2
        metrics[key3] = value3
        metrics[key4] = value4

    return metrics


# get the ancestor directory name
def get_grandparent_directory_name(path, n=4):
    for _ in range(n):
        path = os.path.dirname(path)
    # Get the parent directory of the provided path
    return os.path.basename(path)

def extract_last_line_logging_info(log_file):
    with open(log_file, 'r') as file:
        lines = file.readlines()
        #print(lines)
        if lines:
            last_line = lines[-1].strip()
            #print(last_line)
            pattern = r"Train Epoch: (\d+), Total Sample: \d+, Train Acc: ([\d.]+), Test Acc: ([\d.]+), Loss: [\d.-]+, Total Time: [\d.]+s"

            match = re.search(pattern, last_line)
            
            if match:
                epoch = int(match.group(1))
                train_acc = float(match.group(2))
                test_acc = float(match.group(3))
                return epoch, train_acc, test_acc
    return None, None, None

def extract_mia_metrics(log_file):
    # Read the logging data from the log_file
    with open(log_file, 'r') as file:
        log_data = file.read()

    # Regular expression pattern to extract test names, acc, and auc values
    pattern = r"(correct|confidence|entropy|modified entropy|cross entropy loss) (test) acc:([\d.]+),.+auc:([\d.]+)"

    # Initialize a dictionary to store the extracted values
    test_data = {}

    # Find all occurrences of the pattern in the log data
    matches = re.findall(pattern, log_data)

    # Iterate through the matches and extract values
    for match in matches:
        test_category = match[0]
        test_type = match[1]
        acc_value = float(match[2])
        auc_value = float(match[3])

        test_name_acc = f"{test_category} {test_type} acc"
        test_name_auc = f"{test_category} {test_type} auc"

        test_data[test_name_acc] = acc_value
        test_data[test_name_auc] = auc_value

    # Create a DataFrame from the extracted data
    

    return test_data

def process_files(root_dir, output_excel, var= None):
    # sourcery skip: dict-assign-update-to-union
    output_folder = os.path.dirname(output_excel)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    data = []
    for subdir, dirs, files in os.walk(root_dir):
        if 'logging.log' in files:
            for file in files:
                if file == 'logging.log':
                    log_file_path = os.path.join(subdir, file)
                    # print(log_file_path)
                    epoch, train_acc, test_acc = extract_last_line_logging_info(log_file_path)
                    if epoch is not None:
                        target_dir = get_grandparent_directory_name(subdir, n=3)
                        alpha = os.path.basename(subdir)
                        beta = get_grandparent_directory_name(subdir, n=1)
                        mia_metrics_file = os.path.join(subdir, 'mia_metric-based.log')
                        
                        row_data = {
                            'Loss type': target_dir,
                            'Epochs': epoch,
                            'alpha' : alpha,
                            'beta': beta,
                            'Train Acc': train_acc,
                            'Test Acc': test_acc
                        }
                        
                        if os.path.exists(mia_metrics_file):
                            mia_metrics = extract_mia_metrics(mia_metrics_file)
                            distribution =  extract_metrics(mia_metrics_file)
                            row_data.update(mia_metrics)
                            row_data.update(distribution)
                        
                            #`row_data.update(mia_metrics)` is a method that updates the `row_data` dictionary with the key-value pairs from the `mia_metrics` dictionary.
                            
                        data.append(row_data)
                

    if data:
        df = pd.DataFrame(data)
        df = df.round(3)
        if var != None:
            df = df[var]
        df.to_excel(output_excel, index=False, float_format='%.3f') 




def process_files_yaml(root_dir, output_excel, var= None, if_round = True, dataframe = False):
    # sourcery skip: dict-assign-update-to-union
    output_folder = os.path.dirname(output_excel)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    data = []
    for subdir, dirs, files in os.walk(root_dir):
        if 'logging.log' in files:
            for file in files:
                if file == 'config.yaml':
                    log_file_path = os.path.join(subdir, file)
                    with open(log_file_path, 'r') as f:
                        data_config = yaml.safe_load(f)
                    log_file_path_train_log = os.path.join(subdir, "train_log.yaml")
                    with open(log_file_path_train_log, 'r') as f:
                        data_train_log = yaml.safe_load(f)
                    # print(log_file_path)
                    
                    data_config.update(data_train_log)
                    
                    
                    
                    mia_metrics_file = os.path.join(subdir, 'mia_metric-based.log')
                    if os.path.exists(mia_metrics_file):
                        mia_metrics = extract_mia_metrics(mia_metrics_file)
                        distribution =  extract_metrics(mia_metrics_file)
                        data_config.update(mia_metrics)
                        data_config.update(distribution)
                    
                        #`row_data.update(mia_metrics)` is a method that updates the `row_data` dictionary with the key-value pairs from the `mia_metrics` dictionary.
                        
                    data.append(data_config)
                
    
    if data:
        if if_round:
            df = pd.DataFrame(data)
            df = df.round(3)
            # return(df)
            if var != None:
                df = df[var]
        else:
            df = pd.DataFrame(data)
            # return(df)
            if var != None:
                df = df[var]
        if dataframe:
            return df
        else:
            df.to_excel(output_excel, index=False) 

def load_yaml_to_dict(yaml_path):
    with open(yaml_path, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    return yaml_data

def df_files(root_dir):
    # sourcery skip: dict-assign-update-to-union
    data = []
    for subdir, dirs, files in os.walk(root_dir):
        if 'logging.log' in files:
            for file in files:
                if file == 'logging.log':
                    log_file_path = os.path.join(subdir, file)
                    args_path = os.path.join(subdir, 'config.yaml')
                    # print(log_file_path)
                    epoch, train_acc, test_acc = extract_last_line_logging_info(log_file_path)
                    if epoch is not None:
                        target_dir = get_grandparent_directory_name(subdir, n=4)
                        alpha = os.path.basename(subdir)
                        beta = get_grandparent_directory_name(subdir, n=1)
                        mia_metrics_file = os.path.join(subdir, 'mia_metric-based.log')
                        
                        row_data = {
                            'Loss type': target_dir,
                            'Epochs': epoch,
                            'alpha' : alpha,
                            'beta': beta,
                            'Train Acc': train_acc,
                            'Test Acc': test_acc
                        }
                        
                        if os.path.exists(mia_metrics_file):
                            mia_metrics = extract_mia_metrics(mia_metrics_file)
                            distribution =  extract_metrics(mia_metrics_file)
                            row_data.update(mia_metrics)
                            row_data.update(distribution)
                        
                            #`row_data.update(mia_metrics)` is a method that updates the `row_data` dictionary with the key-value pairs from the `mia_metrics` dictionary.
                            
                        data.append(row_data)
    if data:
        df = pd.DataFrame(data)
        return df.round(3)
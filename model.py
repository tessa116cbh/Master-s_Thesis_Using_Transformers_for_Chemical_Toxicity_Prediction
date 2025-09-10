import json
import os
import torch
import numpy as np
import random
import pandas as pd
import copy
from torch.utils.data import Dataset
from deepchem.feat.smiles_tokenizer import SmilesTokenizer
from datasets import Dataset as HFDataset
import transformers
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import RobertaConfig, RobertaModel #, AutoTokenizer, AutoModel
from sklearn.model_selection import GroupKFold, KFold
import wandb

import paramiko
from scp import SCPClient, SCPException
import shutil

with open("/cephyr/users/clarabe/Alvis/Final_models_heads_1/config.json", 'r') as file:
    config = json.load(file)

mode = config['mode']
project_name = config["project_name"]
directory_path = config['paths']['directory_path']
vocab_path = config['paths']['vocab_path']
saved_folds_path = config['paths']['saved_folds_path']

n_folds = config['folds']['n_folds'] # Do not change

max_length = config['data']['max_length'] # Do not change

hidden_size_roberta = config['roberta']['hidden_size_roberta']
num_hidden_layers_roberta_list = config['roberta']['num_hidden_layers_roberta']
num_attention_heads_roberta = config['roberta']['num_attention_heads_roberta']

input_size = hidden_size_roberta + 1
# input_size = 768 + 1 # Dubbelkollat denna # Do not change
hidden_size = 512 # Do not change

learning_rate = config['dnn_model']['learning_rate'] 

num_epochs = config['training']['num_epochs']
batch_size = config['training']['batch_size']

seed = 20 # 10 innan
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# ------------------------------------------------------------- Utils. ------------------------------------------------------
def get_current_hyperparameters(hidden_size_roberta,num_hidden_layers_roberta,num_attention_heads_roberta):
    return f"encoder_{num_hidden_layers_roberta}_heads_{num_attention_heads_roberta}_emb_{hidden_size_roberta}"

def create_folders_and_get_weight_saving_path(directory_path, mode, current_hyperparameters):
    weight_saving_path = os.path.join(directory_path,current_hyperparameters)
    os.makedirs(weight_saving_path, exist_ok=True)
    return weight_saving_path
    # parameter_folder_name = f"{current_hyperparameters}"
    # mode_path = os.path.join(directory_path,mode)
    # weights_saving_path = os.path.join(directory_path,parameter_folder_name)
    # os.makedirs(mode_path, exist_ok=True)
    # return weights_saving_path


def load_saved_fold(saved_folds_path,fold_iteration,train_or_val):
    if train_or_val == "train":
        name = f"train_rows_fold_{fold_iteration}.csv"
    elif train_or_val == "val":
        name = f"validation_rows_fold_{fold_iteration}.csv"
    else:
        print("Not train nor val")
    data = pd.read_csv(os.path.join(saved_folds_path,name))
    return data

def get_SMILES_weights(train_rows): 
    smiles_counts = train_rows["SMILES_Canonical_RDKit"].value_counts()
    weights = 1.0/smiles_counts
    weights = weights**0.5
    return weights

def map_weights(smiles_weights, train_rows):
    dict = smiles_weights.to_dict()
    all_weights = train_rows["SMILES_Canonical_RDKit"].map(dict).values
    all_weights_tensor = torch.tensor(all_weights)
    return all_weights_tensor


def tokenize(data,tokenizer, max_length, pad_token='[PAD]'):
    pad_token_id = tokenizer.vocab[pad_token]
    processed_data = []

    for smiles in data:
        encoding = tokenizer.encode(smiles)
        padding_length = max_length - len(encoding)

        if padding_length > 0:
            encoding += [pad_token_id] * padding_length
        else:
            encoding = encoding[:max_length]
        attention_mask = [1 if token != pad_token_id else 0 for token in encoding]
        processed_data.append({'input_ids': encoding, 'attention_mask': attention_mask})
    
    return processed_data

def decode(token_ids,tokenizer):
    return tokenizer.decode(token_ids)

def convert_to_hf_dataset(tokenized_data, dataframe):
    # input_ids = tokenized_data["input_ids"]
    # attention_mask = tokenized_data["attention_mask"]
    input_ids = [d['input_ids'] for d in tokenized_data]
    attention_mask = [d['attention_mask'] for d in tokenized_data]
    mgperL_log10 = dataframe["mgperL_log10"].tolist()
    duration_value_log10 = dataframe["Duration_Value_log10"].tolist()
    
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    mgperL_log10 = torch.tensor(mgperL_log10, dtype=torch.float)
    duration_value_log10 = torch.tensor(duration_value_log10, dtype=torch.float)

    hf_dataset = HFDataset.from_dict({'input_ids': input_ids, 'attention_mask': attention_mask,
                                    'mgperL_log10': mgperL_log10, 
                                    'duration_value_log10': duration_value_log10})
    hf_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "mgperL_log10", "duration_value_log10"])

    return hf_dataset


# ---------------------------------------------------------- LOSS CALCULATIONS ----------------------------------------------------

def get_smiles(previous_index, current_index, last_full_batch, validation_rows):
    batch_size = current_index - previous_index
    if current_index > last_full_batch*batch_size:
        smiles = validation_rows.iloc[previous_index:]
    else:
        smiles = validation_rows.iloc[previous_index:current_index]
    
    return smiles

def create_output_df(smiles,duration_value_log10,mgperL_log_10,predictions):
    output_df = pd.DataFrame({"smiles" : smiles,
                                          "duration_value_log10" : duration_value_log10.numpy(),
                                          "true_mgperL_log_10" : mgperL_log_10.numpy(),
                                          "pred_mgperL_log_10" : predictions.unsqueeze(1).numpy().flatten()})
    return output_df

def calculate_residuals_and_absolute_error(output_df):
    residuals = output_df["true_mgperL_log_10"] - output_df["pred_mgperL_log_10"]
    absolute_errors = abs(residuals)
    squared_errors = residuals**2

    output_df["residuals"] = residuals
    output_df["absolute_errors"] = absolute_errors
    output_df["squared_errors"] = squared_errors

    return output_df

def aggregate_median_median(dataframe): 
    median_df = dataframe.groupby(["smiles","duration_value_log10"]).median()

    median_df = median_df.groupby("smiles").median()
    return median_df

def aggregate_mean_mean(dataframe): # Flera kortare experiment (obs ej samma duration time) --> större mean
    mean_df = dataframe.groupby(["smiles","duration_value_log10"]).mean()
    mean_df = mean_df.groupby("smiles").mean()
    return mean_df

def aggregate_median_mean(dataframe):
    median_df = dataframe.groupby(["smiles","duration_value_log10"]).median()
    final_df = median_df.groupby("smiles").mean()
    return final_df

# ----------------------------------------------- SAVE DATA ----------------------------------------------------
def get_logging_dict():    
    methods = ["median_median", "mean_mean", "median_mean"]
    logging_dict = {method: {} for method in methods}

    return logging_dict

def calculate_logging_data(df,inner_logging_dict):
    inner_logging_dict["mean_AE"] = df["absolute_errors"].mean()
    inner_logging_dict["mean_SE"] = df["squared_errors"].mean()
    inner_logging_dict["median_AE"] = df["absolute_errors"].median()
    inner_logging_dict["median_SE"] = df["squared_errors"].median()
    inner_logging_dict["std_AE"] = df["absolute_errors"].std()
    inner_logging_dict["std_SE"] = (df["squared_errors"].std())
    inner_logging_dict["10th_percentile_AE"] = (np.percentile(df["absolute_errors"],10))
    inner_logging_dict["10th_percentile_SE"] = (np.percentile(df["squared_errors"],10))
    inner_logging_dict["90th_percentile_AE"] = (np.percentile(df["absolute_errors"],90))
    inner_logging_dict["90th_percentile_SE"] = (np.percentile(df["squared_errors"],90))


def save_to_wandb(logging_dict, average_validation_loss, epoch):
    wandb.log({"epoch_raw":epoch, "raw_avg_val_loss": average_validation_loss})
    for method in logging_dict.keys():
        inner_logging_dict = logging_dict[method]
        wandb.log({f"mean_AE ({method})": inner_logging_dict["mean_AE"],
                f"mean_SE ({method})": inner_logging_dict["mean_SE"],
                f"median_AE ({method})": inner_logging_dict["median_AE"],
                f"median_SE ({method})": inner_logging_dict["median_SE"],
                f"std_AE ({method})": inner_logging_dict["std_AE"],
                f"std_SE ({method})": inner_logging_dict["std_SE"],
                f"10th_percentile_AE ({method})": inner_logging_dict["10th_percentile_AE"],
                f"10th_percentile_SE ({method})": inner_logging_dict["10th_percentile_SE"],
                f"90th_percentile_AE ({method})": inner_logging_dict["90th_percentile_AE"],
                f"90th_percentile_SE ({method})": inner_logging_dict["90th_percentile_SE"],
                f"epoch ({method})": epoch})

def save_all_outputs_df(all_outputs_df,directory_path,fold_iteration,epoch,current_hyperparameters):
    folder_name = f"all_outputs_df_fold_{fold_iteration}"
    folder_path = os.path.join(directory_path,current_hyperparameters)
    folder_path = os.path.join(folder_path,folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_name = f"all_outputs_epoch_{epoch}.csv"
    all_outputs_df.to_csv(os.path.join(folder_path,file_name), index=False)

# ------------------------------- Transfer folder ---------------------------------------------------------------


def create_remote_folder(hostname, username, password, key_path, remote_path, port=22):

    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        if key_path:
            ssh.connect(hostname, port, username, key_filename=key_path)
        else:
            ssh.connect(hostname, port, username, password)

        stdin, stdout, stderr = ssh.exec_command(f'mkdir -p {remote_path}')
        error = stderr.read().decode().strip()
        ssh.close()

        if error:
            return False, f"Error: {error}"
        else:
            return True, f"Folder '{remote_path}' created successfully."

    except Exception as e:
        return False, f"Exception occurred: {str(e)}"


def create_ssh_client(server, port, username, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port=port, username=username, password=password)
    return client

def scp_transfer_folders_and_cleanup(server, port, username, password, local_folder, remote_folder):
    ssh = None
    transfer_success = False  # flag to track if transfer was successful

    try:
        ssh = create_ssh_client(server, port, username, password)
        with SCPClient(ssh.get_transport()) as scp:
            scp.put(local_folder, remote_path=remote_folder, recursive=True)
        print(f"{local_folder} transferred successfully!")
        transfer_success = True  # set flag if transfer succeeded

    except (paramiko.SSHException, SCPException) as e:
        print(f" {local_folder} Transfer failed!")
        print(f"Error: {e}")
    except FileNotFoundError as e:
        print(f"{local_folder} folder not found!")
        print(f"Error: {e}")
    finally:
        if ssh:
            ssh.close()

        if transfer_success:
            try:
                shutil.rmtree(local_folder)  # safely delete the local folder
                print(f"Local folder '{local_folder}' deleted after successful transfer.")
            except Exception as e:
                print(f" Failed to delete {local_folder} folder: {e}")
        else:
            print(f"Local folder '{local_folder}' kept because transfer failed.")


def scp_transfer_files_and_cleanup(server, port, username, password, local_files, remote_folder):
    ssh = None
    transfer_success = True  

    try:
        ssh = create_ssh_client(server, port, username, password)
        with SCPClient(ssh.get_transport()) as scp:
            for file_path in local_files:
                if os.path.isfile(file_path):
                    scp.put(file_path, remote_path=remote_folder)
                    print(f"{file_path} transferred successfully!")
                else:
                    print(f"File not found: {file_path}")
                    transfer_success = False

    except (paramiko.SSHException, SCPException) as e:
        print(f"Transfer failed!")
        print(f"Error: {e}")
        transfer_success = False

    finally:
        if ssh:
            ssh.close()

        if transfer_success:
            for file_path in local_files:
                try:
                    os.remove(file_path)
                    print(f"Deleted local file: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")
        else:
            print("Files kept because transfer failed.")



# ------------------------------- MODELS ---------------------------------------------------------------

def get_roberta(vocab_size, hidden_size_roberta, num_hidden_layers_roberta, num_attention_heads_roberta):
    config = transformers.RobertaConfig(
                                    vocab_size=vocab_size,
                                    hidden_size=hidden_size_roberta, # Embedding size
                                    num_hidden_layers=num_hidden_layers_roberta, # Block size/encoder layers
                                    num_attention_heads=num_attention_heads_roberta # Number of parallel attention heads
    )
    roberta = transformers.RobertaModel(config)
    return roberta

class EC50Regressor(torch.nn.Module):
    def __init__(self, input_size, hidden_size, roberta, output_size=1):
        super(EC50Regressor, self).__init__()
        self.roberta = roberta
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),  # First fully connected layer
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)   # Output layer (1 output for EC50)
        )

    def forward(self, input_ids, attention_mask, duration_value_log10):
        x = self.roberta(input_ids,attention_mask = attention_mask)['last_hidden_state'][:,0,:]
        duration_value_log10 = duration_value_log10.unsqueeze(1)
        x = torch.cat([x, duration_value_log10], dim=1)
        return self.model(x)

def save_parameters(epoch, fold_iteration, model,save_path):
    folder =  f"parameters_fold_{fold_iteration}"
    folder_path = os.path.join(save_path,folder)
    os.makedirs(folder_path, exist_ok=True)  
    
    new_weight_path = os.path.join(folder_path,f"ec50_model_fold_{fold_iteration}_epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), new_weight_path) 

# ------------------------------------------ TRAINING -------------------------------------------------------------------

folder_created = False
inner_csv_folder_created = False
inner_parameter_folder_created = False

for num_hidden_layers_roberta in num_hidden_layers_roberta_list:
    current_hyperparameters = get_current_hyperparameters(hidden_size_roberta,num_hidden_layers_roberta,num_attention_heads_roberta)
    weights_saving_path = create_folders_and_get_weight_saving_path(directory_path,mode, current_hyperparameters)
    

    tokenizer = SmilesTokenizer(vocab_path)

    for fold_iteration in range(1,n_folds+1):
        print(f"Fold {fold_iteration}/{n_folds}")
        train_rows = load_saved_fold(saved_folds_path,fold_iteration,"train")
        validation_rows = load_saved_fold(saved_folds_path,fold_iteration,"val")
            
        tokenized_train_data = tokenize(train_rows["SMILES_Canonical_RDKit"], tokenizer, max_length, pad_token='[PAD]')
        hf_dataset_train = convert_to_hf_dataset(tokenized_train_data, train_rows)
        
        tokenized_validation_data = tokenize(validation_rows["SMILES_Canonical_RDKit"], tokenizer, max_length, pad_token='[PAD]')
        hf_dataset_validation = convert_to_hf_dataset(tokenized_validation_data, validation_rows)
        
        smiles_weights = get_SMILES_weights(train_rows)
        mapped_weights = map_weights(smiles_weights, train_rows) 
        sampler = torch.utils.data.WeightedRandomSampler(mapped_weights,num_samples=len(mapped_weights),replacement=True)
        
        train_loader = DataLoader(hf_dataset_train, batch_size=batch_size, sampler=sampler)
        validation_loader = DataLoader(hf_dataset_validation,batch_size=batch_size,shuffle=False)
        
        roberta = get_roberta(tokenizer.vocab_size, hidden_size_roberta, num_hidden_layers_roberta, num_attention_heads_roberta) 
        ec50_model = EC50Regressor(input_size=input_size, hidden_size=hidden_size, roberta=roberta)

        loss_function = torch.nn.L1Loss()
        optimizer = torch.optim.Adam(ec50_model.parameters(), lr=learning_rate)
        ec50_model.to(device)
        
        scaler = torch.cuda.amp.GradScaler() #torch.GradScaler(device=str(device))

        wandb.init( project = f"{project_name}",
                    name = f"encoder_{num_hidden_layers_roberta}_heads_{num_attention_heads_roberta}_emb_{hidden_size_roberta}_fold_{fold_iteration}",
                    config = {"learning_rate": learning_rate,
                            "batch_size": batch_size,
                            "num_hidden_layers_roberta": num_hidden_layers_roberta,
                            "embedding_size": hidden_size_roberta,
                            "num_attention_heads_roberta": num_attention_heads_roberta,
                            "epochs": num_epochs},
                    reinit = True)

        
        # Evaluation before training 
        ec50_model.eval()
        val_loss = 0    
        previous_index = 0
        current_index = batch_size
        last_full_bactch = len(validation_rows) // batch_size
        
        logging_dict = get_logging_dict()

        all_outputs_df = pd.DataFrame(columns = ["smiles","duration_value_log10","true_mgperL_log_10",
                                                    "pred_mgperL_log_10","residuals","absolute_errors","squared_errors"])
        with torch.no_grad(): 
            
            for batch in validation_loader:
                
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                duration_value_log10 = batch["duration_value_log10"].to(device)
                mgperL_log_10 = batch['mgperL_log10'].to(device)

                predictions = ec50_model(input_ids, attention_mask, duration_value_log10)
                val_loss += loss_function(predictions.squeeze(1), mgperL_log_10).item()

                smiles = get_smiles(previous_index, current_index, last_full_bactch, validation_rows)["SMILES_Canonical_RDKit"].tolist()
                output_df = create_output_df(smiles,duration_value_log10.cpu(),mgperL_log_10.cpu(),predictions.cpu())
                
                output_df = calculate_residuals_and_absolute_error(output_df)
                all_outputs_df = pd.concat([all_outputs_df, output_df])

                previous_index = current_index
                current_index += batch_size
        

        median_median_df= aggregate_median_median(all_outputs_df.copy(deep=True))
        mean_mean_df = aggregate_mean_mean(all_outputs_df.copy(deep=True))
        median_mean_df = aggregate_median_mean(all_outputs_df.copy(deep=True))
        calculate_logging_data(median_median_df,logging_dict["median_median"])
        calculate_logging_data(mean_mean_df,logging_dict["mean_mean"])
        calculate_logging_data(median_mean_df,logging_dict["median_mean"])

        average_validation_loss = val_loss / len(validation_loader)
        print(f"Before training validation Loss: {average_validation_loss}")
        save_to_wandb(logging_dict, average_validation_loss, 0)
        save_all_outputs_df(all_outputs_df,directory_path,fold_iteration,0,current_hyperparameters)

        if folder_created == False:
                # create folder on saga 
                folder_created = create_remote_folder(
                                    hostname='saga.math.chalmers.se',
                                    username='clarabe',
                                    password="amsterdam123",
                                    key_path=None,
                                    remote_path=f"/storage/clarabe/{current_hyperparameters}")
                
        if inner_csv_folder_created == False:
            inner_csv_folder_created = create_remote_folder(
                                hostname='saga.math.chalmers.se',
                                username='clarabe',
                                password="amsterdam123",
                                key_path=None,
                                remote_path=f"/storage/clarabe/{current_hyperparameters}/all_outputs_df_fold_{fold_iteration}")
            
        if inner_parameter_folder_created == False:
            inner_parameter_folder_created = create_remote_folder(
                                hostname='saga.math.chalmers.se',
                                username='clarabe',
                                password="amsterdam123",
                                key_path=None,
                                remote_path=f"/storage/clarabe/{current_hyperparameters}/parameters_fold_{fold_iteration}")
        
        scp_transfer_files_and_cleanup(
                server='saga.math.chalmers.se',
                port=22,
                username='clarabe',
                password="amsterdam123",
                local_files=[os.path.join(weights_saving_path,f"all_outputs_df_fold_{fold_iteration}/all_outputs_epoch_0.csv")],
                remote_folder=f'/storage/clarabe/{current_hyperparameters}/all_outputs_df_fold_{fold_iteration}')

        for epoch in range(num_epochs):
            
            # Train
            ec50_model.train()
            total_loss = 0

            for batch in tqdm(train_loader):
                
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                duration_value_log10 = batch["duration_value_log10"].to(device)
                mgperL_log_10 = batch['mgperL_log10'].to(device)

                with torch.autocast(device_type=str(device), dtype=torch.float16): 
                # with torch.autocast(device_type=str(device),dtype=torch.bfloat16): 
                # with torch.cuda.amp.autocast():
                    predictions = ec50_model(input_ids,attention_mask,duration_value_log10)
                    loss = loss_function(predictions.squeeze(1), mgperL_log_10)
                
                total_loss += loss.item()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ec50_model.parameters(), 1.0) # scale down parameters proportionally if > 1 so norm of them are 1
                scaler.step(optimizer)
                scaler.update() # if gradient s NaN/Inf --> reduce scaling factor 
                optimizer.zero_grad(set_to_none=True)

                wandb.log({"batch_loss": loss.item()})

            average_total_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}, Loss: {average_total_loss}")
            wandb.log({"epoch (train)":epoch+1,"raw_avg_train_loss":average_total_loss})
            
            # Validation
            ec50_model.eval()
            val_loss = 0    
            previous_index = 0
            current_index = batch_size
            last_full_bactch = len(validation_rows) // batch_size
            
            all_outputs_df = pd.DataFrame(columns = ["smiles","duration_value_log10","true_mgperL_log_10",
                                                    "pred_mgperL_log_10","residuals","absolute_errors","squared_errors"])
            with torch.no_grad(): 
            
                for batch in validation_loader:
                    
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    duration_value_log10 = batch["duration_value_log10"].to(device)
                    mgperL_log_10 = batch['mgperL_log10'].to(device)

                    predictions = ec50_model(input_ids, attention_mask, duration_value_log10)
                    val_loss += loss_function(predictions.squeeze(1), mgperL_log_10).item()

                    smiles = get_smiles(previous_index, current_index, last_full_bactch, validation_rows)["SMILES_Canonical_RDKit"].tolist()
                    output_df = create_output_df(smiles,duration_value_log10.cpu(),mgperL_log_10.cpu(),predictions.cpu())
                    
                    output_df = calculate_residuals_and_absolute_error(output_df)
                    all_outputs_df = pd.concat([all_outputs_df, output_df], ignore_index=True)


                    previous_index = current_index
                    current_index += batch_size

            median_median_df= aggregate_median_median(all_outputs_df.copy(deep=True))
            mean_mean_df = aggregate_mean_mean(all_outputs_df.copy(deep=True))
            median_mean_df = aggregate_median_mean(all_outputs_df.copy(deep=True))
            calculate_logging_data(median_median_df,logging_dict["median_median"])
            calculate_logging_data(mean_mean_df,logging_dict["mean_mean"])
            calculate_logging_data(median_mean_df,logging_dict["median_mean"])

            average_validation_loss = val_loss / len(validation_loader)
            print(f"Epoch {epoch+1}, Validation Loss: {average_validation_loss}")
            
            save_to_wandb(logging_dict, average_validation_loss, epoch=epoch+1)        
            save_parameters(epoch, fold_iteration, ec50_model, weights_saving_path)
            save_all_outputs_df(all_outputs_df,directory_path,fold_iteration,epoch+1,current_hyperparameters)
            
            


            # scp_transfer_folders_and_cleanup(
            # server='saga.math.chalmers.se',
            # port=22,
            # username='clarabe',
            # password='amsterdam123',
            # local_folder=os.path.join(weights_saving_path,f"all_outputs_df_fold_{fold_iteration}/all_outputs_epoch_{epoch+1}.csv"),
            # remote_folder=f'/storage/clarabe/{current_hyperparameters}/parameters_fold_{fold_iteration}')  
            
            scp_transfer_files_and_cleanup(
                server='saga.math.chalmers.se',
                port=22,
                username='clarabe',
                password="amsterdam123",
                local_files=[os.path.join(weights_saving_path,f"all_outputs_df_fold_{fold_iteration}/all_outputs_epoch_{epoch+1}.csv")],
                remote_folder=f'/storage/clarabe/{current_hyperparameters}/all_outputs_df_fold_{fold_iteration}')

            scp_transfer_files_and_cleanup(
                server='saga.math.chalmers.se',
                port=22,
                username='clarabe',
                password="amsterdam123",
                local_files=[os.path.join(weights_saving_path,f"parameters_fold_{fold_iteration}/ec50_model_fold_{fold_iteration}_epoch_{epoch+1}.pth")],
                remote_folder=f'/storage/clarabe/{current_hyperparameters}/parameters_fold_{fold_iteration}')
        
        folder_created = False
        inner_csv_folder_created = False
        inner_parameter_folder_created = False

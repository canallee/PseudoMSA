import argparse
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from utils.dataloader import *
from utils.GFP import *
from utils.torch_utils import *
from Groundtruth_model.CNN import *

import warnings
warnings.filterwarnings("ignore")

def eval_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--model_id', type=int, default=0)
    parser.add_argument('-S', '--seed', type=int, default=0)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-E', '--epoch', type=int, default=300)
    parser.add_argument('-B', '--batch_size', type=int, default=32)
    parser.add_argument('--train_full', type=bool, default=True)
    args = parser.parse_args()
    return args

def save_checkpoints(CNN, model_id, epoch):
    torch.save({
        'epoch': epoch,
        'model_id': model_id,
        'CNN_state_dict': CNN.state_dict(),
    }, os.path.join('GFP_data/ground_model/'+'CNN' + str(model_id) + \
        '_epoch_{}'.format(epoch)))
    return

# def load_checkpoints(CNN, model_id, epoch):
#     checkpoints = torch.load(
#         os.path.join('GFP_data/ground_model/'+'CNN' + str(model_id) + \
#         '_epoch_{}'.format(epoch)))  
#     CNN.load_state_dict(checkpoints['CNN_state_dict'])
#     return

def main():
    args = eval_parse()
    Epoch = args.epoch; lr = args.learning_rate 
    batch_size = args.batch_size; seed = args.seed
    # train_full: whether whole dataset is used for training
    train_full = args.train_full
    device = torch.device("cuda")
    #################################################
    seed_everything(seed=seed)
    df = pd.read_csv('GFP_data/gfp_data.csv')
    X, y = get_gfp_X_y_aa(df, large_only=True, ignore_stops=True)
    if train_full:
        train_set = torch.utils.data.TensorDataset(X, y)
        X_test, y_test = X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, shuffle=True)
        train_set = torch.utils.data.TensorDataset(X_train, y_train)
        
    # init model 
    loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle=True)
    CNN = CNN_ground(seq_len = X.shape[1], hidden_fc=128, hidden_conv=12, n_chars = 20)
    CNN = CNN.to(device=device)
    criterion = NLL_loss 
    MSE = nn.MSELoss()
    optimizer = torch.optim.Adam(CNN.parameters(), lr=lr)
    #
    def train(epoch, loss_tr, loss_te, train_full=False):
        train_loss_running = 0
        for batch, (X_b, y_b) in enumerate(loader):
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            y_pred = CNN(X_b).squeeze(-1)
            loss = criterion(y_b, y_pred)
            loss.backward()
            optimizer.step()
            train_loss_running += loss.item()
        # get avg training loss 
        train_loss = train_loss_running/batch
        # eval
        if train_full == False:
            with torch.no_grad():
                y_pred = CNN(X_test.to(device)).squeeze(-1)
                loss_test = MSE(y_pred[:, 0], y_test.to(device)) 
                print("Epoch:", epoch)
                print("training loss is:", train_loss, 
                  "; testing MSE is:", loss_test.item())
                loss_te.append(loss_test.item())
        else:
            print("Epoch:", epoch, "loss for training is:", train_loss)
            if epoch == Epoch - 1:
                y_pred = CNN(X.to(device)).squeeze(-1)
                mse_full = MSE(y_pred[:, 0], y.to(device)) 
                print("Epoch:", epoch, "MSE for training is:", mse_full.item())
        
        loss_tr.append(train_loss)
        return loss_tr, loss_te
    loss_tr, loss_te = [], []
    print("#=================================================================#")
    print("Start training for GFP dataset; model id is:", args.model_id)
    print("Arguments used:", args)
    for e in range(Epoch):
        loss_tr, loss_te = train(e, loss_tr, loss_te, train_full=train_full)
    # save model weights for ensemble
    save_checkpoints(CNN, args.model_id, Epoch)    
    

if __name__ == '__main__':
    main()

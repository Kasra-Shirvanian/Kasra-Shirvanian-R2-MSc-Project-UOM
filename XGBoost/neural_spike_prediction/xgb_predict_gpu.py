import numpy as np
import xgboost as xgb 
from scipy import io
from operator import itemgetter

prefix1 = "mouse5"  # Change this for each run
prefix2 = "evt22" 

def main():
      # Set your prefixes here
     
    filepath = list(["C:\\Users\\kasra\\Desktop\\XGBOOST\\"])
    filename = list([f"{prefix1}_{prefix2}_data_for_prediction.mat"])
    
    for n in range(len(filepath)):
        temp_path = ''.join(map(str,filepath[n]))
        for m in range(len(filename)):
            temp_name = ''.join(map(str,filename[m]))
            predict(temp_path,temp_name)

def predict(filepath, filename):
    print(filepath + filename)
      
    # Load data
    data = io.loadmat(filepath + filename)
    X = itemgetter('X')(data)
    Y = itemgetter('Y')(data)
    which_fold = np.squeeze(itemgetter('which_fold')(data))
    Kfold = np.squeeze(itemgetter('K')(data))
    
    # Set parameters for GPU
    Ncell = np.size(Y, axis=1)
    Nt = np.size(Y, axis=0)
    Nfeature = np.size(X, axis=1)
    Nsh = np.size(X, axis=2)
    
    xgb_params = {
        'objective': "count:poisson",  # Objective function for count data
        'eval_metric': "logloss",  # Evaluation metric
        'learning_rate': 0.025,
        'subsample': 1,
        'max_depth': 3,
        'gamma': 1,
        'tree_method': 'hist',  # Use histogram-based algorithm, suitable for GPU
        'device': 'cuda'  # Specify to use GPU
    }
    num_round = 500
    
    # Initialize prediction matrix and importance matrix
    Ypred = np.zeros((Nt, Ncell, Nsh))
    importance = np.zeros((Ncell, Nfeature))
    
    # Run XGBoost
    for s in range(Nsh):
        print("Shuffle " + str(s+1) + " of " + str(Nsh))
        for c in range(Ncell):
            print("Cell " + str(c+1) + " of " + str(Ncell))
            temp_imp = np.zeros((1, Nfeature))
            for k in range(Kfold):
                print("Fold " + str(k+1))
                id1 = which_fold != (k+1)
                id2 = which_fold == (k+1)
                xgb1 = xgb.DMatrix(np.squeeze(X[id1,:,s]), label=np.squeeze(Y[id1,c]))
                model = xgb.train(xgb_params, xgb1, num_round)
                xgb2 = xgb.DMatrix(np.squeeze(X[id2,:,s]))
                Ypred[id2, c, s] = model.predict(xgb2)
                if s == 0:
                    score = model.get_fscore()
                    temp_imp[0, 0:len(score)] = temp_imp[0, 0:len(score)] + list(score.values())
            if s == 0:
                importance[c, :] = temp_imp
    
    # Save results
    results = {
        'Ypred': Ypred,
        'xgb_params': xgb_params,
        'num_round': num_round,
        'importance': importance
    }
    save_name = f"{filename[0:len(filename)-4]}_results.mat"
    io.savemat(save_name, results)

import pandas as pd
import os


def get_file(index, path):
    '''
    Returns the dataframe for one of the 2000 instances 
    
    Merges on id.practice
    
    #######
    
    Input: File number
    
    Output: Dataframe
    '''

    practice = [_ for _ in os.listdir(path + "practice") if _.endswith(".csv")]
    practice_year =  [_ for _ in os.listdir(path + "practice_year") if _.endswith(".csv")]
    
    if index < 0 or index > len(practice):
        raise Exception("Index must lie in the number of files given")
    file_num = practice[index-1][-8:-4]
    practice_file = pd.read_csv(path + "practice/" + practice[index-1])
    practice_year_file = pd.read_csv(path + "practice_year/" + practice_year[index-1])
    output = practice_year_file.merge(practice_file, on = "id.practice")
    output = output.rename_axis(file_num, axis = 1)
    return output

def reformat_data(index, path):
    '''
    Takes in an "index" which is just an integer that represents
    the numbers at the end of the csv file and spits out the observed
    outcomes, the initial covariates, and the covariates with the Z 
    variable flipped. 
    
    Edit this if we wish to include more covariates in our classification. 
    
    #######
    
    Input: Index for the file we wish to get variables for 
    
    Output: Outcomes, Covariates, Counterfactual Covariates
    '''
    data = get_file(index, path)
    y = data["Y"]
    X = pd.get_dummies(data.drop(["id.practice", "Y"], axis = 1), columns = ['year', 'X2', 'X4'], drop_first = True) 
    cf_X = X.copy()
    cf_X["Z"] = 1 - X["Z"] 
    return y, X, cf_X

def compute_SATT(index, path, y_cf, by_year = False):
    '''
    Computes the sample average treatment for treated for an index 
    and the predicted values from a model's prediction for the counter
    factual - Comes from just predicting the opposite treatment given 
    a set of covariates
    
    #######
    
    Input: index and counterfactual outcomes
    
    Output: SATT over both years 3 and 4
            SATT for year 3 and year 4 if by_year
    '''
    
    data = get_file(index, path) 
    if len(y_cf) != data.shape[0]:
        raise Exception("Counterfactual terms do not match")
    data["y_cf"] = y_cf
    treated = data[(data["Z"] == 1) & (data["post"] == 1)].copy()
    treated["dY"] = treated["Y"] - treated["y_cf"] 
    treated["weighted_dY"] = treated["n.patients"] * treated["dY"]
    if by_year:
        year_3 = treated[treated["year"] == 3]
        year_4 = treated[treated["year"] == 4]
        ATT_3 = sum(year_3["weighted_dY"]) / sum(year_3["n.patients"])
        ATT_4 = sum(year_4["weighted_dY"]) / sum(year_4["n.patients"])
        return [ATT_3, ATT_4]
    else:
        return sum(treated["weighted_dY"]) / sum(treated["n.patients"])
    
    
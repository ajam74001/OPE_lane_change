import pandas as pd
from ope.methods import doubly_robust
from d3rlpy.dataset import MDPDataset
import numpy as np
import gym
import highway_env
import d3rlpy 
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos import DiscreteCQL
from sklearn.linear_model import LogisticRegression
from d3rlpy.models.encoders import VectorEncoderFactory
import pickle
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('-OldP' , type= str, default= "../lane-change/DqnPolicy-4lanes-gamma0.99-100ksteps.h5", help= 'the path to the dataset collected using the old policy')
# parser.add_argument('-NewP' , type= str, default= "../lane-change/cql64_g0.99-4lanes-DqN/cql_model_dqndata_4lanes_seed_50_ep300gamma0.pt", help= 'the path to the new policy model')

# args = parser.parse_args()
# OldP_path = args.OldP
# NewP_path = args.NewP



def action_probs(X,y):
    # fiting a LR to the observations and predict the probs of each actions X: observation , y: actions taken by the policy 
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(X, y)

    yhat = model.predict_proba(X)
    act_prob = [yhat[i,y[i]] for i in range(len(yhat))] # getting the prob of the taken action
    # print("act_prob", act_prob[0])
    return yhat, act_prob

def MDP_np(dataset):
    # dataset = MDPDataset.load("../lane-change/DqnPolicy-4lanes-gamma0.99-100ksteps.h5")
    observations = []
    rewards=[]
    actions =[]
    terminals= []
    for episode in dataset.episodes:
        observations.append(np.array(episode.observations))
        actions.append(np.array(episode.actions))
        rewards.append(np.array(episode.rewards))
        terminals.append(episode.terminal)
    
    rew = np.concatenate(rewards)
    acts = np.concatenate(actions)
    obs= np.concatenate(observations, axis=0)
    return rew, acts, obs

def data_preparation(dataset): # to do : path root
    # the previous policy 
    rew, acts, obs = MDP_np(dataset)
    obs_dict = pd.DataFrame(obs).to_dict(orient='index')
    obs_s = pd.Series(obs_dict)
    yhat, act_prob = action_probs(obs,acts)
    df = pd.concat([obs_s , pd.Series(acts),pd.Series(act_prob), pd.Series(rew)], axis=1)
    df.columns= ['context', 'action', 'action_prob', 'reward']
    return df 

def policy_preparation(dataset):
    rew, acts, obs = MDP_np(dataset)
    encoder_factory = VectorEncoderFactory(hidden_units=[64,64])
    model = d3rlpy.algos.DiscreteCQL( encoder_factory = encoder_factory, gamma =0)
    model.build_with_dataset(dataset)
    model.load_model('../lane-change/cql64_g0.99-4lanes-DqN/cql_model_dqndata_4lanes_seed_50_ep300gamma0.pt')
    actions_cql = model.predict(obs)
    # print("actions _cql", type(actions_cql[0]))
    yhat,_ = action_probs(obs,actions_cql)
    # print(yhat.shape)
    obs_dict = pd.DataFrame(obs).to_dict(orient='index')
    obs_s = pd.Series(obs_dict)
    res = pd.concat([obs_s,pd.DataFrame(yhat)], axis = 1)
    res.columns= ['context', 'action0', 'action1', 'action2', 'action3', 'action4']
    res['context'] = res['context'].apply(lambda x: str(x))
    # Set 'context' as the index
    res.set_index('context', inplace=True)
    # print(res)
    return res



if __name__ == "__main__":
    # This function gives the new probabilties based on the new policy and the action probs are for the expert (policy first), the library expects only one input to this function that is why i wrote it this way
    def action_probabilities(context):
        context = str(context)
        if context in df_newP.index:
            return {0: df_newP.loc[context]['action0'],
                    1: df_newP.loc[context]['action1'],
                    2: df_newP.loc[context]['action2'],
                    3: df_newP.loc[context]['action3'],
                    4: df_newP.loc[context]['action4']}
        else:
            # Return some default value if the context is not found
            return {0: 0.2, 1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}
    
    dataset = MDPDataset.load("../lane-change/DqnPolicy-4lanes-gamma0.99-100ksteps.h5")

    df = data_preparation(dataset)
    # print(df.head())
    df_newP = policy_preparation(dataset)
    result = doubly_robust.evaluate(df, action_probabilities, num_bootstrap_samples=5)
    print(result)

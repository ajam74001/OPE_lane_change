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
    # print(res)
    return res



if __name__ == "__main__":
    def action_probabilities(context): # This function gives the new probabilties based on the new policy and the action probs are for the expert (policy first), the library expects only one input to this function that is why i wrote it this way
    # if context what then i will return a
#     pass
    # df_newP[df_newP['context']==context]
    # return {0:df_newP.iloc[context,1], 1:df_newP.iloc[context,2], 2:df_newP.iloc[context,3], 3:df_newP[4], 4:df_newP[5]}
        return {0: df_newP[df_newP['context']==context]['action0'], 1: df_newP[df_newP['context']==context]['action1'], 2:df_newP[df_newP['context']==context]['action2'], 3:df_newP[df_newP['context']==context]['action3'], 4:df_newP[df_newP['context']==context]['action4']}

    dataset = MDPDataset.load("../lane-change/DqnPolicy-4lanes-gamma0.99-100ksteps.h5")

    df = data_preparation(dataset)
    # print(df.head())
    df_newP = policy_preparation(dataset)
    result = doubly_robust.evaluate(df, action_probabilities, num_bootstrap_samples=50)
    
    with open('result.pickle', 'wb') as f: 
        pickle.dump(result, f)
    print(result)
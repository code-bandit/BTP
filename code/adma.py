import numpy as np
import pandas as pd


""" Get Current State by ISVM"""
def ISVM(params):
    return params

""" Function to find action with maximum Q-value """
def max(q_t, s_t):
    return q_t[s_t].argmax()

""" Reward Function of Env """
def rewardFunction(s_t, a_t):
    #this is made just for reference to environments reward function
    return 0


""" Main Function """
def main(gamma=0.9, alpha=0.2):

    """ Definitions """
    s_t = ISVM(1) # current state calculated by ISVM 
    a_t = 0 # desicion scheme with max Q-value
    T_max = 1000 # max competion time / time of one episode 
    rewards = np.array([]) # array of rewards
    q_t = np.array([[10, 20, 40], [60, 20, 30]])
    # q_t = np.zeros((state_size, action_size)) # We then update and store our q-values after an episode. This q-table becomes a reference table for our agent to select the best action based on the q-value
    n_s = 0 # new state
    reward = 0 # immediate reward
    sum_qti = 0 # sigma(k=0 to lambda-1 => Q_t+k)
    q_tl = 0
    
    # print(max(q_t, 0))  # check if max works
    T_t = 1
    while T_t < T_max:
        s_t = ISVM(1)
        a_t = max(q_t, s_t)
        r = rewardFunction(s_t, a_t)

        # Command to use a_t to run the model and after this we get "n_s"(new state) and "reward"
        """ Code Here """
        
        # update Q-value
        q_tl = reward+gamma*(np.max(q_t[n_s, :]))
        sum_qti = q_tl
        for r in rewards[::-1]:
            sum_qti = sum_qti+r+gamma*(sum_qti)
        
        np.append(rewards, reward)
        q_t[s_t][a_t] = q_t[s_t][a_t] + alpha*((sum_qti/gamma - q_t[s_t, a_t]))
        
        # s_t = n_s
        T_t = T_t+1


if __name__ == "__main__":
    main()
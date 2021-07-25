[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Report for Project 3: Collaboration and Competition 

## Introduction

In this `Report.md`, you can see an implementation for the third project of the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893).

![Trained Agent][image1]
## Implementation specs

### 1. Summary

I implement the [Twin Delayed Deep Deterministic policy gradient algorithm (TD3)](https://arxiv.org/abs/1802.09477) and apply TD3 to multi-agent 

--------

### 2. Details

#### 2-1. Concepts for TD3

I referred to the [reference site](https://spinningup.openai.com/en/latest/algorithms/td3.html) and the [paper](https://arxiv.org/abs/1802.09477) for the concepts of Twin Delayed DDPG(TD3) 

While DDPG can achieve great performance sometimes, it is frequently brittle with respect to hyperparameters and other kinds of tuning. A common failure mode for DDPG is that the learned Q-function begins to dramatically overestimate Q-values, which then leads to the policy breaking, because it exploits the errors in the Q-function. Twin Delayed DDPG (TD3) is an algorithm that addresses this issue by introducing below critical tricks.

##### 2-1-1. Clipped Double-Q Learning

TD3 learns two Q-functions instead of one (hence “twin”), and uses the smaller of the two Q-values to form the targets in the Bellman error loss functions

- **Target policy smoothing** : Actions used to form the Q-learning target are based on the target policy, $\mu_{\theta_{\text{targ}}}$, but **with clipped noise added on each dimension of the action**. **After adding the clipped noise, the target action is then clipped to lie in the valid action range** (all valid actions, a, satisfy $a_{Low} \leq a \leq a_{High}$). The target actions are thus:

    $$a'(s') = \text{clip}\left(\mu_{\theta_{\text{targ}}}(s') + \text{clip}(\epsilon,-c,c), a_{Low}, a_{High}\right), \;\;\;\;\; \epsilon \sim \mathcal{N}(0, \sigma)$$

     **Target policy smoothing essentially serves as a regularizer for the algorithm**. It addresses a particular failure mode that can happen in DDPG: **if the Q-function approximator develops an incorrect sharp peak for some actions, the policy will quickly exploit that peak and then have brittle or incorrect behavior.** This can be averted by **smoothing out the Q-function over similar actions**, which target policy smoothing is designed to do.
- **clipped double-Q learning** : Both Q-functions use a single target value $y(r, s', d)$, **calculated using whichever of the two Q-functions gives a smaller target value**

    $$y(r,s',d) = r + \gamma (1 - d) \min_{i=1,2} Q_{\phi_{i, \text{targ}}}(s', a'(s')),$$

    and then both are learned by regressing to this target:

    ![https://spinningup.openai.com/en/latest/_images/math/7d5c18f49a242cc3eec554f717fe4f3bfc119bab.svg](https://spinningup.openai.com/en/latest/_images/math/7d5c18f49a242cc3eec554f717fe4f3bfc119bab.svg)

    ![https://spinningup.openai.com/en/latest/_images/math/cd73726a8a3845ade467aed57714912f868f6b36.svg](https://spinningup.openai.com/en/latest/_images/math/cd73726a8a3845ade467aed57714912f868f6b36.svg)

    Using the smaller Q-value for the target, and regressing towards that, helps fend off overestimation in the Q-function.

##### 2-1-2. Delayed Policy Updates : 

**TD3 updates the policy (and target networks) less frequently than the Q-function**. The paper recommends **one policy update for every two Q-function updates**.

Policy is learned just by maximizing $Q_{\phi_1}$:

$$\max_{\theta} \underset{s \sim {\mathcal D}}{{\mathrm E}}\left[ Q_{\phi_1}(s, \mu_{\theta}(s)) \right]$$

which is pretty much unchanged from DDPG. However, in TD3, the **policy is updated less frequently than the Q-functions are**. This helps damp the volatility that normally arises in DDPG because of how a policy update changes the target.

##### 2-1-3. Pseudocode

![https://spinningup.openai.com/en/latest/_images/math/b7dfe8fa3a703b9657dcecb624c4457926e0ce8a.svg](https://spinningup.openai.com/en/latest/_images/math/b7dfe8fa3a703b9657dcecb624c4457926e0ce8a.svg)

#### 2-2. Concepts for multi-agent


I referred [this](https://arxiv.org/abs/1706.02275) for multi-agent concepts and applied TD3 to multi-agent concepts.

Concepts are represented by the following figure.

![image](https://user-images.githubusercontent.com/73100569/126896252-a00ca014-ca56-44a5-b0bf-74e98b92fb9e.png)

Psuedocode for Multi-Agent Actor-Critic(**Here pseudocode used DDPG**) for Mixed Cooperative-Competitive Environments.

![image](https://user-images.githubusercontent.com/73100569/126896173-966a5613-d045-41a1-b746-a08d6cc525ff.png)

By referring to **2-1. Concepts for TD3** and **2-2. Concepts for multi-agent**, I combined two concepts for implementing my alogorithm to complete this third project. For critic of TD3, it uses **full states(states of all agents)** and **full actions(actions of all agents)**. For actor of TD3, it used only **local states(states of agent that will be updated at current step)** and **local actions(actions of agent that will be updated at current step)**.


------------
#### 2-3. Networks

The network structure is as follows:

##### 2-3-1. Actor

state -> BatchNorm -> Linear(state_size, 256) -> BatchNorm -> LeakyRelu -> Linear(256, 128) -> BatchNorm -> LeakyRelu -> Linear(128, action_size) -> tanh

##### 2-3-2. Critic

state -> BatchNorm -> Linear(state_size * **agent_num**, 256) -> Relu -> (concat with action) -> Linear(256 + action_size * **agent_num**, 128) -> Relu -> Linear(128, 1) 

#### 2-4. Hyperparameters

Agent hyperparameters are passed as constructor arguments to `MultiAgent`.  The default values, used in this project, are:

| parameter    | value  | description                                                                   |
|--------------|--------|-------------------------------------------------------------------------------|
| BUFFER_SIZE  | 1e6    | Number of experiences to keep on the replay memory for the TD3                |
| BATCH_SIZE   | 256    | Minibatch size used at each learning step                                     |
| GAMMA        | 0.99   | Discount applied to future rewards                                            |
| TAU          | 4e-2   | Scaling parameter applied to soft update                                      |
| LR_ACTOR     | 6e-4   | Learning rate for actor used for the Adam optimizer                           |
| LR_CRITIC    | 2e-3   | Learning rate for critic used for the Adam optimizer                          |
| NUM_LEARN    | 8      | Number of learning at each step                                               |
| NUM_TIME_STEP| 10     | Every NUM_TIME_STEP do update                                                 |
| EPSILON      | 1.0    | Epsilon to noise of action                                                    |
| EPSILON_DECAY| 2e-5   | Epsilon decay to noise epsilon of action                                      |
| POLICY_DELAY | 3      | Delay for policy update (TD3)                                                 |
| AGENT_NUM    | 2      | Number of agents for multi-agent                                              |


Training hyperparameters are passed to the training function `train` of `multi_agent`, defined below.  The default values are:

| parameter                     | value            | description                                                             |
|-------------------------------|------------------|-------------------------------------------------------------------------|
| n_episodes                    | 3000             | Maximum number of training episodes                                     |
| max_t                         | 3000             | Maximum number of steps per episode                                     |


-----------

### 3. Result and Future works

#### 3-1. Reward

<img src = "https://user-images.githubusercontent.com/73100569/126896626-393519df-43f1-489e-8cbc-e7d9463f87fd.png" width="400" height="300">

Here x-axis is the episode and y-axis is the score. Environment solved in **505 episodes**. You can see relatively stable and fast learning since TD3 is the improved version of DDPG compared to the benchmark that Udactiy provided where it used DDPG.(**see the below figure**)

<img src = "https://video.udacity-data.com/topher/2018/August/5b75ef77_screen-shot-2018-08-16-at-4.37.07-pm/screen-shot-2018-08-16-at-4.37.07-pm.png" width="415" height="300">

#### 3-2. Future works

1. Parameters tuning for TD3. 
2. Implement this project by other algorithms like **PPO(Proximal Policy Optimization)** which is on-policy algorithm or **SAC(Soft Actor Critic)** which is off policy with entropy maximization to enable stability and exploration.  
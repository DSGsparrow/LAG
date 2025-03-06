# train: dodge missile
记录训练设置、结果和总结  
train_jsbsim.py 设置参数为下，


## 发射规则 
shoot_flag = agent.is_alive   
np.sum(self.lock_duration[agent_id]) >= self.lock_duration[agent_id].maxlen：保持锁定超过1秒  
distance <= self.max_attack_distance：进入最大距离14000  
self.remaining_missiles[agent_id] > 0：有弹  
shoot_interval >= self.min_attack_interval：距离上次发射间隔大于25s  















## 参数：
--env-name
SingleCombat
--algorithm-name
ppo
--scenario-name
1v1/ShootMissile/HierarchySelfplay
--experiment-name
v1
--seed
1
--n-training-threads
1
--n-rollout-threads
32
--cuda
--log-interval
1
--save-interval
1
--use-selfplay
--selfplay-algorithm
fsp
--n-choose-opponents
1
--use-eval
--n-eval-rollout-threads
1
--eval-interval
1
--eval-episodes
1
--num-mini-batch
5
--buffer-size
3000
--num-env-steps
1e8
--lr
3e-4
--gamma
0.99
--ppo-epoch
4
--clip-params
0.2
--max-grad-norm
2
--entropy-coef
1e-3
--hidden-size
128
128
--act-hidden-size
128
128
--recurrent-hidden-size
128
--recurrent-hidden-layers
1
--data-chunk-length
8
--use-prior
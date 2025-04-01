# train: dodge missile 2
train_dodge_missile.py  
主要用来记录数据  
用来对状态进行攻击评估和模仿学习

## 智能体
我方：dodge1  
敌方：pursue + 发射规则  
+ 打弹前加速
+ 基于**规则**来发射
+ 也可以计算命中率

## 网络
net_shoot_missile.py
CustomPolicy


# train: dodge missile
train_shoot_missile.py

用sb3来训练  
## config: HierarchyVsBaselineSelf.yaml
file path:  
render_path: "render_train/dodge"  
enm_ver: "shoot1"

A 0 颗弹
B 1 颗弹，使用训练好的shoot模型

## 环境：singlecombat_env_shoot.py
step:加入render，render_path在config中

## 任务：
HierarchicalSingleCombatShootTask  
singlecombat_with_missle_task.py  
我不知道为什么它的step没有调用baseline  
都要加上调用baseline  
在normalize_action函数里实现，  
+ 动作由PPO产生，传进环境后给normalize_action
+ 函数里既处理自己的动作：
+ 从离散到杆量
+ 还产生敌方的
+ 不再调用上层的，直接写自己的

## 敌方智能体
训练过的shoot  
enm_ver的数字表示版本号  
singlecombat_task.py  
ShootAgent  
sb3的PPO调用模型竟然可以不输入环境，实在是太好了  
集成好的果然牛逼  
本来也不应该给这个新的  
状态：21位，包括弹状态
动作：4位，最后是打弹


## 奖励和终止条件
奖励增加了打弹间隔小于10 给-25的惩罚  
event_driven_reward加了躲弹成功给100的奖励  
避免掩盖其他奖励  

终止修改了dodge_safe_return，之前容易上了没打弹就结束了  

## 状态和动作
状态：21位，包括弹状态  
动作：3位，分层，


## 
好像改差不多了，后面加上动作记录，用来模仿学习

这怎么敌机这么抽象？



记录训练设置、结果和总结  
train_jsbsim.py 设置参数为下，
> 北纬1度大概为111公里，0.17度大概18.9公里  
> 北纬60度附近，经度差1度大约111.32×cos(60°)=111.32×0.5=55.66km  
> 南北纬度距离基本上不变，东西经度要看维度，因为有个cos  


## 发射规则 
shoot_flag = agent.is_alive   
np.sum(self.lock_duration[agent_id]) >= self.lock_duration[agent_id].maxlen：保持锁定超过1秒  
distance <= self.max_attack_distance：进入最大距离14000m  
self.remaining_missiles[agent_id] > 0：有弹  
shoot_interval >= self.min_attack_interval：距离上次发射间隔大于25s  


## 设置：
+ 似乎不需要从远方飞过来或者对准之类的
+ 直接打就完了
+ 随机方向，距离，高度，速度，
+ 记录：

1. 设置初始条件：在singlecombat_env.py中reset_simulator函数中实现
2. 自己的随机初始高度和速度吗？先不随机
3. 敌方距离在9000到14000米，朝向对准圆心，速度从400到1000英尺每秒
4. 在singlecombat_with_missile_task.py中修改了shoot_flag产生规则：
5. 自己存活且有弹就发射

奖励：
+ Posture_reward: 
  + range_reward: ![距离奖励图](../ppt/range_reward_v3.png)
  + orientation: 视线角越小，奖励越大，敌方视线角小于pi/2，给负奖励，
+ missile_reward:
  + 奖励同向拉开，导弹降速就给奖励，降的越快给越多
  + 惩罚反向对冲，导弹降速就惩罚小一点
+ AltitudeReward 低于安全高度给负奖励，还向下飞就给速度的负奖励
+ event 被击中给-200的大惩罚
+ EndRelativeAltitude: 
  + 敌方导弹完成加速后，且速度低于1.2*150=180后
  + 或导弹飞行48秒后，即指导能力只有0.2后
  + 在simulator.py missile中加了一个导弹的能量耗尽标志函数

记录奖励曲线：
+ 修改了reward_function_base.py中的get_reward_trajectory函数
+ 将奖励曲线以json形式传回来了
+ ~~修改了env_base.py中step函数，在回合结束时调用get_reward_trajectory函数~~
+ ~~然后保存奖励曲线到~~
+ 不对，还是不能在env_base里保存，怎么想都不合适
+ 改为在每次eval时记录测试的结果
+ 然后还想把运行时所有的logging.info都存到文件里

+ train_jsbsim.py中添加了设置logging的函数，main函数会调用
+ 现在可以将info都保存到运行路径中的run.log里

+ 然后在jsbsim_runner中添加了保存reward_trajectory的
+ 现在每个episode测试都会把这个episode的reward_trajectory保存起来
+ 到运行路径中的reward_trajectory_x里

检查一下EndRelativeAltitude

## 在task_base中get_termination中添加了当done时，在info中添加了
info['success'] = success

+ 搞错了，经纬度还搞反了
+ 可以用render来测试


## 测试各个情况下危险程度
完成
dodge_missile_test.py  
调用singlecombat_env_test中SingleCombatEnvTest  
区别在于在init中加入了enemy_positions的字典数据列表，从外部对敌方进行初始化  
reset太坑了，每次step之后自动调用，还不能加输入  
所以提前全部输进去，自己遍历  
等训练完就可以测试，估计要测很久

reset 返回值：ndarray(1,1,21)

## 躲弹结束回合标志
只要敌方弹全部失效，而且自己存活，无论敌方死活，dodgemissile safe return 回true  
然后回进env_base 的step  
最后进了个_pack，对done这种大小只有1的数据只取自己的，因此A结束，回合结束  
设置的确实牛逼  
现在问题是状态没了，回合一结束肯定是reset了  
保存上个回合的状态  
然后是每个回合仿真完就保存result 和数据
为什么这么多302？为什么会有2？
2的原因是上来就失速了，因为最低是150米每秒，也就是490英尺每秒  
render一下看看  

render有个标志位self._create_records = False  
因为之前没有reset，结果导致每次新回合的render都有问题，现在把重置加到了reset里

## 尝试多环境测试，随机初始条件，不然太慢了
在singlecombat_env_test.py中修改新的step









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
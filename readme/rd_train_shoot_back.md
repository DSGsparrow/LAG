# train shoot back t2
加上了速度奖励，减小了高度奖励，奖励高度范围也减小了  
另外这个打弹有点太严了，几千场一次都没打  
看来这个打弹和机动还必须得适配？  
可能确实是得把中间奖励去掉？  
基本上还是挺好的

# train shoot back t
~~用shoot back 1的继续训练~~  
使用transformer模型，因此不再继续训练  
解决问题：  
随着对方降高太多

使用solo相同的adapter  
使用相同transformer  




# train shoot back gail3
这次全部调用了，包括前三维的动作输出层  
只能说是一般




# train shoot back gail2
先调用之前的模型，~~包括前三维在内~~  
失败了，只调用了前面的
看这样会不会好一点  
如果再不行我要BC了  
log_file='./train/result/train_shoot_back2_gail2.log',  
render_path: "./render_train/shoot_back2_gail2"  

效果还是不好，还是直接打弹，太让人失望了



# train shoot back gail
从shoot back2 的测试结果中学习  
0.9 的模仿 0.1 的奖励  

## env
gail训练同样需要运行环境，  
task 修改为4维动作，最后一维作为打弹输出  

log_file='./train/result/train_shoot_back2_gail.log',  
config_name='1v1/ShootMissile/HierarchyVsBaselineShootBack',  
render_path: "./render_train/shoot_back2_gail"  

## result
训练的不好，没有回转，没有控制





# train shoot back
躲弹后的回转反击

## 初始智能体
我机：state_enm  
~~shoot_imi~~  shoot back  失败了  
改为从头训练，但只训练飞行，打弹由规则决定 shoot back2

敌机：state-my  
dodge4  
增加了高度的奖励，避免降高太多


## net
同模仿学习后训练的网络：  
net/net_shoot_imitation.py  

## 环境
### env 
要重新再来一个，毕竟初始化是完全不一样的  
config: 
LAGmaster/envs/JSBSim/configs/1v1/ShootMissile/
HierarchyVsBaselineShootBack.yaml  

render_path: "./render_train/shoot_back"  
state_path: "./test_result/result/states_imi_dodge2.jsonl"  
log_file = "./train/result/train_shoot_back.log"  
model_path = "./trained_model/shoot_imitation/ppo_air_combat_imi.zip"  

### task
也重写了，感觉奖励什么的应该是需要重新修改一下的  
比如距离就不能太近，感觉是吧起码  

### result
训练效果很差
































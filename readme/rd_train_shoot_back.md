# train shoot back
躲弹后的回转反击

## 初始智能体
我机：state_enm  
shoot_imi

敌机：state-my  
dodge2


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



































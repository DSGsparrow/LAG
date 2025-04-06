# test all models
## shoot_imitation vs dodge
test_log: "./test_result/log/test_shoot_imi_vs_dodge.log"  
render: "./test_result/render/shoot_imi_vs_dodge"  

model: ~~"./trained_model/imitation_shoot/imitation_pretrained_pytorch.pt"~~  
不对，搞错了，这么引用没有训练的线性层
trained_model/shoot_imitation/ppo_air_combat_imi.zip

### adapter
win_rate and episode added  
use get_attr to get these records  

init: enemy basically targets at self

### result
+ 目前看tacview： 
  + 高度老是下降
  + 角度也一般
  + 但加速了挺好
  + 基本上也都可以解释，但的确不是很完美
  + 高速时会有时忽略距离
  + 经常确实没有兼顾高度和距离
  + 高度速度也没有每次都控制住
  + 躲得也还可以吧
  + 有时候会有点激进地转向，
  + 但也有反向S，也有侧向拉开
  + 轨迹经常有点固定，喜欢反向后很快回转，这个时候容易被击中
  + 有时候确实看起来博弈的还不错^_^
  + 打完弹之后的的确得再练练
  + 没打中的时候确实有点决策的还不够好
  + 48.8%

## shoot_imitation_origin vs dodge
test_log: './test_result/log/test_shoot_imi_origin_vs_dodge.log'  
render: "./test_result/render/shoot_imi_origin_vs_dodge"  

model: './trained_model/imitation_shoot/imitation_pretrained_pytorch.pt'  
imitation policy model predict  

### result:
+ 目前来看实在是太强了，命中率过90了都，超高速，超近距，确实没办法
  + dodge完全没办法，
  + 还会出现过早回头，提前爬升，这种追求姿态奖励的动作
  + 躲弹确实躲得太放松了
  + 57.3%









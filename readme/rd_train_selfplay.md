# test self play
## 环境
## task
终止条件
- 有弹在飞就没结束
- 飞机都没被击落且任意一个还有弹就没结束
- 弹全部失效或者被击中：结束
- 双方弹都打完都失效：平局
- 双方都击中：平局2
- 一个击中一个没击中：看还有没有弹在飞




https://bmb.hust.edu.cn/info/1033/1739.htm


# train self play
## 环境
创建时要接收敌方策略，PPO.load好的对象  
敌方是predict产生动作  
在adapter上修改吧











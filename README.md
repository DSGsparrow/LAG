# 公开空战平台LAG上的实验
用于代码同步开发

## documents
> 文件说明

ppt 组会汇报使用材料

##  note
> 代码阅读使用记录
### missile
missile: LAGmaster/envs/JSBSim/core/simulatior.py

AIM-9L

好像没法更改，导弹参数固定，不是调用了jsbsim里的仿真

### plane
官方提供的接口都是xml文件 JSBSim/data/aircraft/f16/f16.xml

具体的动力学代码：
JSBSim/data/src/models/FGAerodynamics.cpp

调用飞机主文件：
JSBSim/data/src/FGFDMExec.cpp

对应python文件：
E:\anaconda\envs\sb3\lib\site-packages\jsbsim\_jsbsim.cp38-win_amd64.pyd

python代码中没有代码实现，只说明了接口

> 对于输入：是把值传给了catalog里的变量，然后c调用？

### 训练
参数：config.py
自博弈：
- 采用elo机制
  - 每个智能体有个elo分数
  - 越高说明越强
  - 如果elo高，那就得赢得多
  - 每次eval时更新elo和oppo，选择elo最高的
  - 双方都直接采取动作进行对抗，环境给奖励
- 似乎每次只能从头开始训练
- 自己使用：修改load
- 敌方是否训练？不
- policy_pool里面都有谁？存的是过去的自己

自博弈：
1. 第一回合只save，save是把当前episode的自己存起来，
2. policy-pool是{episode:elo};
3. 然后下个回合进eval
4. 从policy里选出来？到底是怎么choose的？随机选的
5. 打一场，得到敌我奖励
6. 更新policy-pool的elo，把这场测试中使用的opponent的elo更新为和当前self对战的
7. 更新敌方智能体为最强的


policy-pool:
先save再eval

self.policy_pool[str(episode)] = self.latest_elo

### 环境
single_combat_env：主要还是task

#### single combat task.py
分层：hierarchy 不直接到杆量，到高度、速度、航向角的变化量delta

然后把delta给训练好的模型baseline_model.pt（相当于pid）
得到杆量动作

此外还有几个准备好的Agent：直飞、追逐、固定航道、躲弹

躲弹：dodge_missile_model.pt
额外输入了导弹的状态信息：位置速度转化之后的相对信息？

### 训练 Hierarchical SingleCombat ShootTask
奖励：
- PostureReward 视线角越小，奖励越大，敌方视线角小于pi/2，给负奖励，距离超过7000就急剧下降
- 甚至用了势能（potential）函数：尽可能接近视线角对准等这样的目标，避免奖励累积
- AltitudeReward 低于安全高度给负奖励，还向下飞就给速度的负奖励
- EventDrivenReward 终端奖励：被击中或坠毁给-200，导弹获胜（击中）给200
- ShootPenaltyReward 发射一颗导弹就给-10

终止条件：
* LowAltitude 高度低于2500，坠毁
* ExtremeState 异常状态速度异常（极端高速） 旋转异常（极端角速度）
* 高度异常（极端高空） 加速度异常（超过 10G）
* Overload 过载超过10个G
* SafeReturn 正常结束：自己被击中，自己坠毁，所以敌机被击毁或者所以导弹失效
* Timeout 超时：1000个仿真步，200秒

训练成果：03-04 11：00 
好像也有一定的效果：
+ 出现了盘旋、摇摆S的飞行路径来躲弹
+ 但攻击打弹的效果一般，前期训练的智能体经常不打弹
+ 看来要修改一下打弹惩罚
+ 然后是两机距离很近时无法转进攻？
+ 出现了！转进攻出现了，400
+ 但打弹重合

> 张洪图 02-14-2025
# LAG

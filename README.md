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
readme/readme-train-shoot-missile-hierarchy-selfplay.md  

### 训练dodge missile
readme/readme-train-dodge-missile.md  
## **注：修改了singlecombat_env.py中reset_simulator函数**


### 初始化、reset 飞机状态：
env_base.py中load_simulator函数调用yaml中init_state

### record state
single_combat_with_missile_task.py get_states  
从仿真器获取双方飞机 经纬度、位置、速度

env_base.py get_states  
整合task传回来的数据，字典传回去，用_pack合成array(1,1,18)  

弹的数据暂时先不加了，多个也不好处理


> 张洪图 02-14-2025
# LAG

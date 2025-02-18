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


> 张洪图 02-14-2025

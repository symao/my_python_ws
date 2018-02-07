##1. Python 环境配置
1. 安装python, 最好安装python版本3以上, 安装参考链接:https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/0014316090478912dab2a3a9e8f4ed49d28854b292f85bb000
2. 安装依赖的python包: numpy,cv2

##2. How to Run?
1. 运行map_play.py脚本即可. 
2. 需要做的: 在map_play.py中, 参见strategy_random_walk()自定义一个决策函数, 输入是当前的地图,位置等信息, 输出是一个上下左右的方向.
3. 如何评价效果? 替换为你自己的策略, play_episode可以跑一局查看当前策略的运行效果, 方便调试. 调试完成后, 可以用play_rounds跑很多局查看平均分数,方差等.
4. map_env.py中是地图仿真程序, 可以不需要修改, pathplan.py中是A*路径规划的程序, 可以不需要修改,直接调用, 但也可以重新编写新的路径规划算法, 函数输入输出参见pathplan_astar



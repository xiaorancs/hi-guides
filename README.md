# 皇包车（HI GUIDES）
皇包车（HI GUIDES）是一个为中国出境游用户提供全球中文包车游服务的平台。拥有境外10万名华人司机兼导游（司导），覆盖全球90多个国家，1600多个城市，300多个国际机场。截止2017年6月，已累计服务400万中国出境游用户。

由于消费者消费能力逐渐增强、 旅游信息不透明程度的下降，游客的行为逐渐变得难以预测，传统旅行社的旅游路线模式已经不能满足游客需求。如何为用户提供更受欢迎、更合适的包车游路线，就需要借助大数据的力量。结合用户个人喜好、景点受欢迎度、天气交通等维度，制定多套旅游信息化解决方案和产品。

赛后总结：     

精品旅行服务成单预测

精品旅行服务成单预测

数据量：5W用户 = train 4W + test 1W
数据介绍：
1. 用户个人信息
2. 用户的行为信息
3. 用户的历史订单信息
4. 待预测订单信息，label
5. 用户评论数据


特征工程
用户个人信息特征
数据清洗:
1. 缺失的性别用，其他表示，使用one-hot
2. 用户信息中缺失的省份用-1表示，进行label编码，使用one-hot
3. 年龄是否缺失，使用同省份的平均年龄取整填充
4. 该比赛中，可以使用是否缺失值作为一个特征，在进行填充

用户评论数据特征
1. 用户订单评分的统计特征，man、min、sum、avg等
2. 用户的评分占该用户所有的订单的比例，和最后一个评分的值，以及时间，计算与预测时间的差值
3. 对评论数据进行简单的按"|"分开，计算有多少个不同的词语
4. 将用户评论数据进行进行情感分析正面或者负面，使用简单的统计手动统计一部分正面次和负面词，进行匹配

用户历史订单信息
1. 最近一个交易的月，日，周，周几，小时，分钟
2. 根据用户去过的城市和国家，统计改城市和国家的精品率，作为这个用户的富裕成都
3. 滑动窗口特诊，窗口大小 k = [3m, 6m, 12m], 订单的数据量，去过的城市数量等
4. 用户总的订单数，精品订单数和其所占的比例，去过的城市的个数，以及占总分比例等
5. 是否是有过两次以上的精品订单的老用户
6. 交易时间差的统计量

用户的APP的行为特征
1. 所有操作的时间差的统计量
2. 每一个操作的最后一次操作时间，以及与当前时间的差值
3. 每一个操作的时间差的统计量，max，min，std，avg等
4. 每个操作的统计量，max，min，avg，已经操作最多的和最少的是哪个操作
5. actionType组合操作统计，1-3,4-6,7-9，1-6，4-9，以及对应点击次数占所有点击次数的比例
6. 最后三个的操作行为
7. 使用滑动窗口，k = [3d, 7d, 15d, 30d, 3m, 6m]
8. 根据所有的9的次数进行排序，将对应的排序rank作为特征
9. 用户使用最
交叉特征:
1. 使用stacking特征
2. 使用one-hot之后的交叉特征，与其他数值特征相乘
3. 结合order和action数据，直接统计，将精品服务和费精品服务作为10和11操作
4. action 1-9 到 10-11时间差的统计特征，最后一次时间差
5. actionType序列转移时间间隔，用户action 1 3 5 2 7 4， 计算转移的时间特征，


Model：
xgb， lgb， rf， mlp进行模型的融合，catboost，等模型进行模型stacking和模型的融合。


评价指标：
AUC + F1

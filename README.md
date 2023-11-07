# Processing order

1. read data，查阅格式/数据/特征列
2. 回顾assignment 1的问题——取消的因素，y_label = is_canceled，去找特征间的相似度
3. 画图，柱图/折线，单一线形关系
4. 特征间相似度 apriori/minhash/LSH（仅针对离散型数据）先用one-hot编码再求相似（把文字型分类特征做one-hot encode）
5. 算相似度有很多方式，画图（直观）；apriori（代码少），常用于推荐算法；协方差矩阵（两两之间）

# Next step
1. 是否可以对连续型数据做pca降维

# How to run the code

`run dm.py` 

# Result
![Frequent_itemsets](https://mp-987a18d7-91e9-45e2-9479-258d4157cb76.cdn.bspapp.com/dm_assignment/frequent_itemset.png)
![Association_rules](https://mp-987a18d7-91e9-45e2-9479-258d4157cb76.cdn.bspapp.com/dm_assignment/rules.png)

# Reference
[Article](https://blog.csdn.net/m0_64336780/article/details/125355963)

# Features

hotel：表明客人入住的酒店类型，包含“resort hotel”和“city hotel’

is_cancelled：表明订单是否取消，‘0’代表未取消，‘1’代表取消订单

lead_time:提前预订时长

arrival_date_year：预订到店年份

arrival_date_month：预订到店月份

arrival_date_week_number：预订到店周数

arrival_date_day_of_month:预订到店日期

stays_in_weekends_nights:预订入住间夜（周末）

stays_in_week_nights:预订入住间夜（周中）

adults:入住成人数

children：入住儿童数

babies：入住婴儿数

meal：预订餐食类型

country:客人来源

market_segment：市场细分

distribution_channel:订单来源

is_repeated_guest：是否为老顾客，‘0’新客，‘1’老客

previous_cancelltions:顾客取消次数

previous_bookings_not_canceled：顾客先前离店订单数

reserved_room_type：客人预订房型

assigned_room_type：客人实际入住房型

booking_changes：预订更改次数

deposit_type：押金类型

agent;提交预定的旅行社id

company:提交预订公司的id

days_in_waiting_list：客户预订确认前的等待天数

customer_type：客人类型

adr:单房收入

required_car_parking_spaces：客人需要车位数，0~8不等

total_of_special_requests：其他特殊需求数

reservation_status：预订最终状态，分为check-out，cancelled，no-show

reservation_status_date：预订最终状态更新最新时间

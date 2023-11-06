import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

data = pd.read_csv('../data/hotel_bookings.csv')

'''step 1
    observe the data.shape'''

# print(data.head())
# print(data.shape)  # 119390， 32
# print(data.columns)  # 32 features
# print(data.isnull().mean())  # find the proportion of value NULL,then decide to fill NULL or drop NULL
# print(data.describe().T)  # calculate count, mean, std eta.
# print(data.loc[data['hotel'] == 'City Hotel'])  # City Hotel start from row 40060

'''step 2
    observe the general data in figure'''

'''data.hist(figsize=(20, 10))
plt.show()  # show the general distribution
plt.figure(figsize=(15,8))'''

'''step 3
    observe the relationship between single feature with the is_canceled label'''

'''
sns.countplot(x='hotel'
             ,data=data
             ,hue='is_canceled'
             ,palette=sns.color_palette('Set2',2)
            )
plt.show()

plt.figure(figsize=(15,8))
sns.countplot(x='deposit_type'
             ,data=data
             ,hue='is_canceled'
             ,palette=sns.color_palette('Set2', 2)
            )
plt.show()'''

'''step 4
    To ensure the unique value in a certain col,then decide whether to drop this feature'''

# unique_values = data['total_of_special_requests'].unique()
# to get the unique value in a certain col,then decide whether to drop this feature
# print(unique_values)
# 'reservation_status' : ['Check-Out' 'Canceled' 'No-Show']
# 'total_of_special_requests' : [0 1 3 2 4 5]
# ......


# new = data[(data['deposit_type'] == 'Refundable') & (data['is_canceled'] == 1)]
# print(new)
# is_canceled = 0, 126 rows / is_canceled = 1,36 rows
# Thus,too small to see in the figure

'''step 5
    Preprocessing'''

# reconstruct features
new_data = data.drop(columns=['reservation_status_date'])

# cate consists of classification features
cate = new_data.columns[new_data.dtypes == "object"].tolist()
num_cate = ['agent', 'company', 'is_repeated_guest']
cate = cate + num_cate

'''results = {}
for i in ['agent', 'company']:
    result = np.sort(new_data[i].unique())
    results[i] = result'''

new_data[['agent', 'company']] = new_data[['agent', 'company']].fillna(0, axis=0)

# create new feature in_company and in_agent
new_data.loc[new_data['company'] == 0, 'in_company'] = 'NO'
new_data.loc[new_data['company'] != 0, 'in_company'] = 'YES'
new_data.loc[new_data['agent'] == 0, 'in_agent'] = 'NO'
new_data.loc[new_data['agent'] != 0, 'in_agent'] = 'YES'

# create new feature same_assignment, if the reserved room is the same as the assigned room,value is Yes, else No.
new_data.loc[new_data['reserved_room_type'] == new_data['assigned_room_type'], 'same_assignment'] = 'Yes'
new_data.loc[new_data['reserved_room_type'] != new_data['assigned_room_type'], 'same_assignment'] = 'No'

new_data = new_data.drop(labels=['reserved_room_type', 'assigned_room_type', 'agent', 'company'], axis=1)

# reset 'is_repeated_guest'
new_data['is_repeated_guest'][new_data['is_repeated_guest'] == 0] = 'NO'
new_data['is_repeated_guest'][new_data['is_repeated_guest'] == 1] = 'YES'

# find a mode to fill the NAN
new_data['country'] = new_data['country'].fillna(new_data['country'].mode()[0])
new_data['children'] = new_data['children'].fillna(0, axis=0)

for i in ['in_company', 'in_agent', 'same_assignment']:
    cate.append(i)

for i in ['reserved_room_type', 'assigned_room_type', 'agent', 'company']:
    cate.remove(i)

# !!! If I encoder the months, the results will not match the actual months. (5.0000 -> July)
oe = OrdinalEncoder()
oe = oe.fit(new_data.loc[:, cate])
new_data.loc[:, cate] = oe.transform(new_data.loc[:, cate])

'''for i in new_data.columns:
    print(i)
    print(new_data[i].unique())'''
print(new_data['arrival_date_month'].unique())

'''step 6
    Create a new df to do Apriori'''

# previous_cancellations/deposit type/customer type/in_company/in_agent
new_df = new_data.drop(columns=['days_in_waiting_list', 'adr', 'country', 'lead_time',
                                'adults', 'arrival_date_week_number', 'arrival_date_day_of_month',
                                'stays_in_weekend_nights', 'stays_in_week_nights', 'market_segment',
                                'distribution_channel'
                                ])

new_df.loc[new_df['babies'] == 0, 'new_babies'] = 0
new_df.loc[new_df['babies'] > 0, 'new_babies'] = 1

new_df.loc[new_df['children'] == 0, 'new_children'] = 0
new_df.loc[new_df['children'] > 0, 'new_children'] = 1

new_df.loc[new_df['meal'] == 0, 'new_meal'] = 0
new_df.loc[new_df['meal'] > 0, 'new_meal'] = 1

new_df.loc[new_df['arrival_date_year'] == 2015, 'new_year_2015'] = 1
new_df.loc[new_df['arrival_date_year'] == 2016, 'new_year_2016'] = 1
new_df.loc[new_df['arrival_date_year'] == 2017, 'new_year_2017'] = 1

# nd stands for no deposit, nr stands for non refund, rf stands for refundable
new_df.loc[new_df['deposit_type'] == 0, 'new_nd'] = 1
new_df.loc[new_df['deposit_type'] == 1, 'new_nr'] = 1
new_df.loc[new_df['deposit_type'] == 2, 'new_rf'] = 1


new_df = new_df.drop(columns=['children', 'babies', 'meal', 'hotel', 'arrival_date_month', 'required_car_parking_spaces',
                              'total_of_special_requests', 'booking_changes', 'customer_type', 'reservation_status',
                              'previous_cancellations', 'previous_bookings_not_canceled', 'arrival_date_year',
                              'deposit_type'])
new_df = new_df.fillna(0, axis=0)

'''for i in new_df.columns:
    print(i)
    print(new_df[i].unique())'''

frequent_itemsets = apriori(new_df, min_support=0.4, use_colnames=True)
print(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric='lift')
print(rules)

# covariance_matrix = new_df.cov()
# print(covariance_matrix)
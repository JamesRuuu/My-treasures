import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import warnings
warnings.filterwarnings('ignore')

order = {'001': '面包，黄油，尿布，啤酒',
        '002': '咖啡，糖，小甜饼，鲑鱼，啤酒',
        '003': '面包，黄油，咖啡，尿布，啤酒，鸡蛋',
        '004': '面包，黄油，鲑鱼，鸡',
        '005': '鸡蛋，面包，黄油',
        '006': '鲑鱼，尿布，啤酒',
        '007': '面包，茶，糖鸡蛋',
        '008': '咖啡，糖，鸡，鸡蛋',
        '009': '面包，尿布，啤酒，盐',
        '010': '茶，鸡蛋，小甜饼，尿布，啤酒'}
data_set = []
id_set= []
shopping_basket = {}
for key in order:
    item = order[key].split('，')
    id_set.append(key)
    data_set.append(item)

shopping_basket['ID'] = id_set
shopping_basket['Basket'] = data_set
print(shopping_basket)
data = pd.DataFrame(shopping_basket)
print(data)

data_id = data.drop('Basket',1)
data_basket = data['Basket'].str.join(',')
data_basket = data_basket.str.get_dummies(',')
new_data = data_id.join(data_basket)
print(new_data)

frequent_itemsets = apriori(new_data.drop('ID', 1),min_support=0.5,use_colnames=True)
print(frequent_itemsets)
r = association_rules(frequent_itemsets,metric='lift')
print(r)

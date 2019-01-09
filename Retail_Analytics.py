
# coding: utf-8

# # Library includes

# In[1]:


import numpy as np
import pandas as pd
import sys
import seaborn as sns
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt


# In[2]:


path1 = '/home/partha/Desktop/MTP/Data/RecommenderSystem/'


# # Read data

# In[3]:


category_tree = pd.read_csv(path1 + 'category_tree.csv')
events = pd.read_csv(path1 + 'events.csv')
item_properties_part1 = pd.read_csv(path1 + 'item_properties_part1.csv')
item_properties_part2 = pd.read_csv(path1 + 'item_properties_part2.csv')


# In[4]:


item_properties_part1.info()


# In[6]:


item_properties_part1[["timestamp","itemid"]] = item_properties_part1[["timestamp","itemid"]].apply(pd.to_numeric, downcast="integer")
item_properties_part2[["timestamp","itemid"]] = item_properties_part2[["timestamp","itemid"]].apply(pd.to_numeric, downcast="integer")


# # Data Wrangling

# In[7]:


view_events = events[events["event"] == "view"]
addtocart_events = events[events["event"] == "addtocart"]
transaction_events = events[events["event"] == "transaction"]


# In[8]:


item_properties = pd.concat([item_properties_part1, item_properties_part2], ignore_index=True)


# In[9]:


pd.concat([events[15:20], transaction_events[:2]], axis = 0).reset_index(drop=True)


# # Exploratory analysis

# In[10]:


df = category_tree.groupby(["parentid"], as_index=False)
#df.groups


# In[7]:


root_nodes = category_tree[np.isnan(category_tree["parentid"])].categoryid.values
root_nodes


# In[8]:


category_tree.categoryid.nunique()


# In[9]:


category_tree.categoryid.count()


# In[ ]:


events.columns


# In[ ]:


#events.describe()


# In[ ]:


events.visitorid.nunique()


# In[ ]:


#item_properties.describe()


# In[ ]:


item_properties.head()


# In[ ]:


item_properties_part1.columns


# In[ ]:


item_properties.itemid.nunique()


# In[ ]:


item_properties.itemid.max()


# In[ ]:


item_ids = np.sort(item_properties.itemid.unique())


# In[ ]:


item_properties.timestamp.nunique()


# In[ ]:


item_properties.property.nunique()


# In[ ]:


item_properties[(item_properties.property == "888")]#.timestamp.nunique()#timestamp == 1431226800000)]'''(item_properties.itemid == 1)]# &'''


# In[ ]:


#test_df = item_properties.sort_values(["timestamp", "itemid"])
#test_df_item1[test_df_item1["timestamp"] == 1431226800000]
#test_df


# In[ ]:


#item_properties_part1.sort_values(["timestamp", "itemid"])


# In[ ]:


#item_properties_part2.sort_values(["timestamp", "itemid"])


# In[ ]:


events[events.visitorid == 257597]


# # Descriptive statistics

# ### Events distribution

# In[8]:


event_frequency = events.groupby("event", as_index=False).size().reset_index(name="Count")
event_frequency.plot.bar(x="event", y="Count");
plt.title("Event Classification");


# Serves as validation of data relevant to field. Confirms with genral observation.

# In[9]:


event_percent = event_frequency
event_percent["count%"] = event_frequency.Count*100/event_frequency.Count.sum()
event_percent = event_percent[["event", "count%"]]
event_percent


# Clearly, majority of the visitors view products online, very few add items to cart for buying and even fewer customers actually do a transaction.

# ### Customers who do transaction

# In[10]:


transaction_per_visitor = transaction_events.groupby("visitorid", as_index=False).size().reset_index(name="Count")
transaction_per_visitor.describe(percentiles=[0.75, 0.80, 0.85, 0.9, 0.95, 0.99])


# In[11]:


transaction_groups = {}
transaction_groups["first timers"] = transaction_per_visitor[transaction_per_visitor.Count == 1].Count.count()
transaction_groups["medium frequency"] = transaction_per_visitor[(transaction_per_visitor.Count < 13) & (transaction_per_visitor.Count > 1)].Count.count()
transaction_groups["high frequency"] = transaction_per_visitor[transaction_per_visitor.Count >= 13].Count.count();

#transaction_groups.values()


# In[12]:


sns.barplot(transaction_groups.keys(), transaction_groups.values());
plt.title("Customer classification");


# Outlier removal

# In[13]:


high_frequency_group = transaction_per_visitor[transaction_per_visitor.Count >= 13]


# In[14]:


transaction_events[transaction_events.visitorid == 42552]


# A basket can be identified with a unique transactionid per user in the same time stamp.

# In[13]:


#a= transaction_events[transaction_events.visitorid == 170470].itemid.values
#transaction_events[transaction_events.visitorid == 170470]


# In[14]:


#b = addtocart_events[addtocart_events.visitorid == 170470].itemid.values
#addtocart_events[addtocart_events.visitorid == 170470]


# In[15]:


#item_properties[(item_properties.itemid == 76831)]# & (item_properties.property == "available")]


# ### Histogram of basket size per transaction

# In[11]:


basket_itemCount = transaction_events.groupby(["transactionid"], as_index=False)["itemid"].size().reset_index(name="itemCount")


# In[28]:


basket_itemCount.describe(percentiles=[0.80, 0.9, 0.95, 0.99])


# Most of the baskets contain only single item and hence donot give much insight.

# #### Baskets can be divided into different sizes

# In[16]:


basket_itemCount.itemCount.hist(bins=np.arange(32)-0.5, xrot=90);
plt.xlabel("Number of items bought per basket");
plt.ylabel("Frequency in baskets");
plt.xticks(range(32));


# In[17]:


multiple_basket_itemCount = basket_itemCount[basket_itemCount["itemCount"] > 2]
multiple_basket_itemCount["itemCount"].hist(bins=np.arange(32)-0.5, xrot=90)
plt.xlabel("Number of items bought per basket")
plt.ylabel("Frequency in baskets")
plt.xticks(range(32));
plt.title("Medium and large size baskets");


# In[12]:


def basketClassification(item):
    ''' 
    Baskets are classified by size as:
    0. Very big baskets: more than 9
    1. Big baskets: more than 4 and less than equal to 9
    2. Medium baskets: Greater than 2 and less than equal to 4
    3. Small baskets: Less than equal to 2
    '''
    if item > 9:
        return "VeryBig"
    elif item > 4 and item <= 9:
        return "Big"
    elif item > 2 and item <= 4:
        return "Medium"
    elif item <= 2:
        return "Small"


# In[13]:


basket_itemCount["basketType"] = basket_itemCount.itemCount.apply(basketClassification)


# In[20]:


basketTypes = basket_itemCount.groupby(["basketType"], as_index=False).size().reset_index(name="Count")
sns.barplot(x=basketTypes.basketType,y=basketTypes.Count);
plt.title("Basket Classification");


# In[21]:


basketTypes


# #### Mapping baskets to customers

# In[38]:


transaction_events.columns


# In[14]:


customer_basket = transaction_events[["visitorid", "transactionid"]].drop_duplicates().reset_index(drop=True)
customer_basket = pd.merge(customer_basket, basket_itemCount, on="transactionid")


# In[40]:


customer_basket.visitorid.nunique()


# #### Exploring taxonomy

# In[15]:


levels = {}
i = 0


# In[16]:


relationship_tree = category_tree.copy()


# In[17]:


while len(relationship_tree.categoryid.values) != 0:
    levels[i] = relationship_tree[relationship_tree.parentid.isnull()].categoryid.values
    relationship_tree = relationship_tree[relationship_tree.parentid.notnull()]
    parents = relationship_tree.parentid.values
    for x in xrange(len(parents)):
        if parents[x] in levels[i]:
            parents[x] = np.nan
    relationship_tree["parentid"] = parents
    i = i+1;    


# In[18]:


# Number of nodes at each level
num_levels = i
#sum = 0
for j in xrange(num_levels):
    print "Level "+str(j)+": "+str(len(levels[j]))
    #sum += len(levels[j])
#sum


# #### Choosing optimal taxonomy

# In[ ]:


#The root level of each item is chosen to be used for further clustering


# #### Fixing taxonomy level for each item

# In[114]:


#chosen level:0


# In[19]:


item_category = item_properties[item_properties.property == "categoryid"]#[["itemid", "value"]]


# In[20]:


item_category["categoryid"] = item_category.value
item_category = item_category.drop("value", axis=1).reset_index(drop=True)


# In[28]:


item_category[item_category.itemid == 281245]


# In[32]:


1277 in levels[2]


# In[33]:


438 in levels[2]


# In[34]:


category_tree[category_tree.categoryid == 438]


# In[35]:


category_tree[category_tree.categoryid == 1277]


# In[36]:


item_category[item_category.itemid == 76151]


# In[37]:


category_tree[category_tree.categoryid == 1085]


# In[38]:


1228 in levels[3]


# In[21]:


def map2rootcategory(id_):
    '''Function to map a categoryid at any level to its root categoryid'''
    #print id_
    parent = category_tree[category_tree.categoryid == int(id_)].parentid.values
    if len(parent) != 0:
        parent = parent[0]
    else:
        return np.nan
    while( np.isnan(parent) == False ):
        id_ = parent
        parent = category_tree[category_tree.categoryid == id_].parentid.values[0]
    return id_


# In[40]:


map2rootcategory(1228)


# In[41]:


levels[0]


# In[22]:


item_category = item_category.drop(["timestamp", "property"], axis=1).drop_duplicates()


# In[49]:


item_category.count()


# In[23]:


#Subset of item categories considering only the 1st category id as others would also have the same root
item_category = item_category.drop_duplicates(subset="itemid")


# In[24]:


item_category = item_category.set_index("itemid")
item_category = item_category.T.to_dict('list')


# In[25]:


transaction_items = transaction_events[["itemid", "transactionid"]].reset_index(drop=True)


# In[26]:


def categoryfromitem(item):
    #print item
    #print item_category[item]
    if(item_category.get(item, -1) != -1):
        return int(item_category[item][0])
    else:
        return np.nan


# In[27]:


transaction_items["categoryid"] = transaction_items.itemid.apply(categoryfromitem)


# In[28]:


ghost_items = transaction_items[transaction_items.categoryid.isnull()].itemid.values


# In[37]:


category_tree[category_tree.categoryid == ghost_items[-1]]


# In[29]:


transaction_items = transaction_items[transaction_items.categoryid.notnull()]


# In[30]:


transaction_items["rootcategoryid"] = transaction_items.categoryid.apply(map2rootcategory)


# In[39]:


transaction_items.head()


# In[54]:





# #### Creating factor table

# In[31]:


root_categories = transaction_items.rootcategoryid.values


# In[32]:


for x in root_categories:
    transaction_items[x] = (transaction_items.rootcategoryid == x)


# In[65]:


transaction_items.columns


# In[33]:


frequency_table = transaction_items.drop(["itemid", "categoryid", "rootcategoryid"], axis=1)


# In[61]:


frequency_table[frequency_table.transactionid==11117]


# In[63]:


frequency_table[frequency_table.transactionid==11117].sum()


# In[34]:


frequency_table = frequency_table.groupby("transactionid", as_index=False).sum()


# In[68]:


frequency_table[frequency_table.transactionid ==11117]


# Instead of 0-1 values as mentioned in the paper, my frequency table has the number of items(frequency) of a particular product category( taxonomy: rootid) in each basket.

# In[35]:


factor_table = frequency_table.set_index("transactionid")
factor_table = factor_table.applymap(lambda x: 1 if x > 0 else 0)


# In[71]:


factor_table.loc[[11117]]


# In[67]:


factor_table.head()


# #### Clustering

# KMeans

# In[75]:


from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# #### Determine number of clusters using Elbow method and Silhoutte score

# In[76]:


#kmeans = KMeans().fit(factor_table)
Nc = range(1,21)
kmeans = [KMeans(n_clusters=i) for i in Nc]


# In[77]:


sse = [kmeans[i].fit(factor_table).inertia_ for i in range(len(kmeans))]


# In[72]:


plt.plot(Nc,sse);
plt.xlabel("Number of clusters");
plt.ylabel("Sum squared errors");
plt.title("Elbow Curve");
plt.xticks(range(1,21));
plt.grid();


# By visual inspection, we choose n_cluster = 12

# #### Trying to visualize clusters

# In[115]:


pca = PCA(n_components=1).fit(factor_table)
pca_d = factor_table.index
pca_c = pca.transform(factor_table)


# In[117]:


plt.scatter(pca_c[:, 0], pca_d, c=kmeans[12].labels_);


# #### Inferences

# In[125]:


kmeans = np.array(KMeans(n_clusters=12).fit_predict(factor_table))


# In[140]:


cluster_freq = {}
for x in range(12):
    cluster_freq[x] = (kmeans == x).sum()


# In[148]:


sns.barplot(x=cluster_freq.keys(),y=cluster_freq.values());
plt.xlabel("Cluster number");
plt.ylabel("Number of baskets in the cluster");


# Non uniform distribution in each of the clusters. Possiblity of deriving advantage from such a situation

# Alternate clustering

# In[51]:


factor_table.sum().sum()*100.0/factor_table.size


# ### K-Means using Hamming distance

# In[73]:


from kmeans import kmeans_hamming


# In[ ]:


#kmeans = KMeans().fit(factor_table)
Nc = range(1,21)
kmeans = [KMeans(n_clusters=i) for i in Nc]


# In[70]:


import sys
sys.path.append("home/partha/Desktop/MTP")
print sys.path


# ### Rank order CLustering

# In[40]:


from ROC import cluster


# In[49]:


descriptor_matrix = factor_table.astype("float")


# In[57]:


clusters = cluster(descriptor_matrix.values, n_neighbors=2)


# In[50]:





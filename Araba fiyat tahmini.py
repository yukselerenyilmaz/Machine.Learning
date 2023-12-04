#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ## x araçların km lerini y de araçların fiyatlarını göstersin

# In[2]:


x=np.array([[6000],[8200],[9000],[14200],[16200]]).reshape(-1,1)
y=[86000,82000,78000,75000,70000]


# In[3]:


plt.figure()
plt.title("otomobil fiyat-km dağılım grafiği")
plt.xlabel("km")
plt.ylabel("fiyat")
plt.plot(x,y,"k.")
plt.axis([3000, 20000, 60000 , 95000])
plt.grid(True)
plt.show()


# In[4]:


## şimdi regresyon modelini oluşturalım


# In[5]:


from sklearn.linear_model import LinearRegression


# In[6]:


model=LinearRegression()


# In[7]:


model.fit(x,y)


# In[8]:


## algoritmanın model parametrelerine bakalım


# In[9]:


model.intercept_


# In[10]:


model.coef_


# In[11]:


## modelin tahmin doğrusunu oluşturalım


# In[12]:


plt.figure()
plt.title("otomobil fiyat-km dağılım grafiği")
plt.xlabel("km")
plt.ylabel("fiyat")
plt.plot(x,model.predict(x),color= "blue")
plt.plot(x,y,"k.")
plt.grid(True)
plt.show()


# In[13]:


## test yapalım


# In[14]:


test_araba=np.array([[12000]])
predicted_price=model.predict(test_araba)[0]
print("12000 km'deki aracın tahmini fiyatı : %.4f" % predicted_price)


# In[15]:


test_araba=np.array([[9000]])
predicted_price=model.predict(test_araba)[0]
print("9000 km'deki aracın tahmini fiyatı : %.4f" % predicted_price)


# In[16]:


## modelimiz ne kadar iyi çalışıyor bakalım


# In[17]:


y_predictions=model.predict(x)
for i , prediction in enumerate (y_predictions):
    print("tahmin edilen fiyat :%.4f , gerçek fiyat : %s"% (prediction,y[i]))


# In[18]:


## standart yanılgılarını bulalım 


# In[19]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[20]:


mean_absolute_error(y,y_predictions)


# In[21]:


mean_squared_error(y,y_predictions)


# In[22]:


r2_score(y,y_predictions)


# In[23]:


## r^2 miz yüzde 92 gibi bir değer çıktı bu regresyon eğrimizin verinin yüzde 92'sini açıklayabildiği anlamına gelmektedir


# In[24]:


## artık kareler ortalamasını hesaplayalım


# In[25]:


from math import sqrt


# In[26]:


AKO=sqrt(mean_squared_error(y,y_predictions))


# In[27]:


print(AKO)


# In[28]:


## Bulduğumuz yüzde 92 lik açıklanma oranının makine öğrenmesi nezlinde hiçbir anlamı yoktur çünkü modelimiz var olan verilerde açıklama yapıyor model daha önce görmediği veride aynı açıklama oranına sahip olacak mı bakalım


# In[29]:


x_test=np.array([[1700],[2600],[11000],[14000],[17500]]).reshape(-1,1)
y_test=[94000,944000,73000,83000,75000]


# In[30]:


## eğitim veri setininin grafiği
plt.figure()
plt.title("otomobil fiyat-km serpilme grafiği")
plt.xlabel("km")
plt.ylabel("fiyat")
plt.plot(x,y,"k.")
plt.axis([0,20000,60000,95000])
plt.grid(True)
plt.show()


# In[31]:


## eğitim veri seti ve tahmin doğrusu
plt.figure()
plt.title("otomobil fiyat-km dağılım grafiği")
plt.xlabel("km")
plt.ylabel("fiyat")
plt.plot(x,model.predict(x))
plt.plot(x,y,"k.")
plt.axis([0,20000,60000,100000])
plt.grid(True)
plt.show()


# In[32]:


plt.figure()
plt.title("otomobil fiyat-km dağılım grafiği")
plt.xlabel("km")
plt.ylabel("fiyat")
plt.plot(x,y,"k.")
plt.plot(x_test,y_test,"x")
plt.axis([0,20000,60000,100000])
plt.grid(True)
plt.show()


# In[33]:


plt.figure()
plt.title("otomobil fiyat-km dağılım grafiği")
plt.xlabel("km")
plt.ylabel("fiyat")
plt.plot(x,model.predict(x))
plt.plot(x,y,"k.")
plt.plot(x_test,y_test,"x")
plt.axis([0,20000,60000,100000])
plt.grid(True)
plt.show()


# In[34]:


plt.figure()
plt.title("otomobil fiyat-km dağılım grafiği")
plt.xlabel("km")
plt.ylabel("fiyat")
plt.plot(x,model.predict(x),color="red")
plt.plot(x_test,model.predict(x_test),"--",color="red")
plt.plot(x,y,"k.")
plt.plot(x_test,y_test,"*")
plt.axis([0,20000,60000,100000])
plt.grid(True)
plt.show()


# In[6]:


araba=pd.read_csv("C:\\Users\\aracl\\Desktop\\car_price_prediction.csv")


# In[7]:


araba.head()


# In[8]:


araba.shape


# In[9]:


araba[araba.duplicated()]


# In[10]:


## sadece kolon isimlerini getirmek için 


# In[11]:


araba.drop_duplicates(inplace=True)


# In[12]:


araba[araba.duplicated()]


# In[13]:


araba.shape


# In[14]:


## ıd sayısal olmasına rağmen bir işimize yaramayacağı için onu çıkarmalıyız


# In[15]:


araba.drop("ID",axis=1,inplace=True)


# In[16]:


araba.head()


# In[17]:


araba[araba["Levy"]== "-"]


# In[18]:


araba["Levy"].replace({"-":np.nan},inplace=True)


# In[19]:


araba["Levy"]


# In[20]:


## olmayan verileri Nan şekline getirdik az önceki işlem olmasaydı olmaz mıydı ?


# In[21]:


araba.isnull().sum()


# In[22]:


##araba.isnull().sum().sum() => Toplam 'NaN' değeri sayısını verir.


# In[23]:


## veriden çıkartılsa daha iyi olmaz mı ?


# Yıllara Göre Fiyat Grafiği

# In[24]:


fig, ax  = plt.subplots(figsize=(14,7))
sns.lineplot(x= araba["Prod. year"], y= araba["Price"], ax =ax)


# In[25]:


## Prod. year yapmayınca neden çalışmadı


# In[26]:


## kategori kolunundaki değerlerden kaç adet olduğunu göstermek için


araba["Category"].value_counts()


# In[27]:


plt.figure(figsize=(14,7))
sns.countplot(araba["Category"])


# In[28]:


## deri koltuk olup olmayan araçlar


# In[29]:


sns.countplot(araba['Leather interior'],color="r")


# In[30]:


## yakıt türü


# In[31]:


araba["Fuel type"].value_counts()


# In[32]:


sns.countplot(araba['Fuel type'])


# In[33]:


## silindir sayılarının grafiği


# In[34]:


sns.kdeplot(araba["Cylinders"])


# In[35]:


sns.countplot(araba['Gear box type'])


# In[36]:


sns.countplot(araba['Drive wheels'])


# In[37]:


## arabaların renklerinin dağılışı


# In[38]:


araba["Color"].value_counts()


# In[39]:


plt.figure(figsize=(15,10))
sns.countplot(araba["Color"])


# In[40]:


sns.kdeplot(araba['Airbags'])


# In[41]:


## özellikleri değiştirme


# In[42]:


araba['Levy'] = araba['Levy'].astype(float) ## burada levy değerlerini float cinsinden yazdık


# In[43]:


sns.kdeplot(araba["Levy"])


# In[44]:


## burada vergilerin üstel bir dağılıma benzediğini düşünebiliriz 


# In[45]:


## vergilerde çok fazla eksik değer olduğunu görmüştük onları fit.transform ile dolduralım


# In[46]:


from sklearn.impute import KNNImputer


# In[47]:


impute= KNNImputer(n_neighbors=5) ## bu verinin ne demek olduğunu bilmiyorum


# In[48]:


araba["Levy"]= impute.fit_transform(araba["Levy"].values.reshape(-1,1))
## reshape kısmına bak                                                       


# In[49]:


sns.kdeplot(araba["Levy"])


# In[50]:


## kategorik verilere tam sayı değerleri aktarmalıyız ki makine bunları işleyebilsin


# In[51]:


pip install category_encoders


# In[52]:


import category_encoders as ce


# In[53]:


ge = ce.LeaveOneOutEncoder()


# In[54]:


## https://analyticsindiamag.com/a-complete-guide-to-categorical-data-encoding/  
## kategorik kodlamaya buradan çalıştım
## https://contrib.scikit-learn.org/category_encoders/
## buradan indirdim


# In[55]:


araba['Manufacturer'] = ge.fit_transform(araba['Manufacturer'],araba['Price'])


# In[56]:


target= ce.TargetEncoder()


# In[57]:


araba['Model'] = target.fit_transform(araba['Model'],araba['Price'])


# In[58]:


araba["Category"]= target.fit_transform(araba['Category'],araba['Price'])


# In[59]:


araba["Leather interior"]= araba["Leather interior"].map ({"Yes":1, "No":0})


# In[60]:


## şimdi bir korelasyon ölçümü yapacağız


# In[61]:


araba.corr()


# In[62]:


## sonuçlardan anlaşıldığı kadarıyla fiyatla en yüksek korelasyon kategori arasındadır


# In[63]:


## yakıt türüne sıralı kodlama uyguluyoruz


# In[64]:


mean_encoding = araba.groupby(["Fuel type"])["Price"].mean()


# In[65]:


araba["Fuel type"] = araba["Fuel type"].map(mean_encoding)


# In[66]:


araba['Turbo'] = araba['Engine volume'].astype(str).str.contains("Turbo")


# In[67]:


## arabada turba olup olmamasını 0 1 olarak ikili kodlarız


# In[68]:


araba["Turbo"]= araba["Turbo"].map({False:0 , True : 1})


# In[69]:


araba['Engine volume'] = araba["Engine volume"].astype(str).str.split().str[0]


# In[70]:


araba["Engine volume"] = araba["Engine volume"].astype(float)


# In[71]:


araba["Mileage"] = araba["Mileage"].astype(str).str.split().str[0]


# In[72]:


araba["Mileage"]=araba["Mileage"]. astype(str).str.split().str[0]


# In[73]:


araba['Mileage'] = araba['Mileage'].astype(float)


# In[74]:


plt.figure(figsize=(8,5))
sns.kdeplot(araba['Mileage'])


# In[75]:


araba["Gear box type"]=ge.fit_transform(araba["Gear box type"],araba["Price"])


# In[76]:


araba["Drive wheels"]=ge.fit_transform(araba["Drive wheels"],araba["Price"])


# In[77]:


araba["Doors"]=araba["Doors"].map({"04-May": 4, "02-Mar":2 , ">5":5 })


# In[79]:


##


# In[80]:


mean_encoding_1 = araba.groupby(['Wheel'])['Price'].mean().sort_values(ascending = False)


# In[81]:


##Burada bir çeşit ortalama sıralaması yapılmıştır


# In[82]:


araba['Wheel'] = araba['Wheel'].map(mean_encoding_1)


# In[83]:


araba['Color'] = ge.fit_transform(araba['Color'],araba['Price'])


# In[84]:


araba.head()


# In[85]:


## görüldüğü gibi bütün str değerler encoding ile sayısal bir ifadeye atandı 


# In[86]:


## artık bir korelasyon ısı tablosu oluşturabiliriz


# In[87]:


plt.figure(figsize=(15,10))
sns.heatmap(araba.corr(),annot=True, fmt= ".2g")
## annot = ifade doğruysa her hücreyi bölecek ve veri değerini girip ona göre renklendirecek


# ## Aykırı Değer Tespiti
# Python'da bir işlev, def anahtar kelimesi kullanılarak tanımlanır burada amacımız bir class oluşturup aykırı değer sayısını bulmaktır
# In[88]:


Outliers = []
def data_outliers(data):
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)
    
    for i in data:
        Z_score = (i-mean)/std
        if abs(Z_score)>threshold:
            Outliers.append(i)
    return Outliers


# In[89]:


data_outliers(araba['Price'])


# In[90]:


## çeyrekliklere ayırıyoruz

Orada z puanı kullanmak, aykırı değerlerin çok fazla olmamasını sağlar, ancak en göze çarpan yöntem, aykırı değerleri tedavi etmek için IQR yöntemini kullanmaktır.
# In[91]:


Q1 = np.percentile(araba['Price'],25)


# In[92]:


Q1


# In[93]:


Q3=np.percentile(araba["Price"],75)


# In[94]:


Q3


# In[95]:


IQR= Q3- Q1


# In[96]:


## ortalama aykırı değerleri bulmak için


# In[97]:


IQR


# In[98]:


lower_bond = Q1 - (1.5*IQR)
upper_bond = Q3 + (1.5*IQR)


# In[99]:


lower_bond


# In[100]:


upper_bond


# In[101]:


## burada bir güven aralığı bulduk


# In[102]:


araba[(araba['Price'] < lower_bond) | (araba['Price'] > upper_bond)]


# In[103]:


def outliers_detection(araba,f):
    Q1 = np.percentile(araba[f],25)
    Q3 = np.percentile(araba[f],75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - (1.5*IQR)
    upper_bound = Q3 + (1.5*IQR)
    
    ls = araba.index[(araba[f] < lower_bound ) | (araba[f] > upper_bound) ]
    
    return ls


# In[104]:


araba.columns


# In[105]:


index_list = []
## using for loop to extract all the outliers
for feature in ['Price', 'Levy', 'Mileage']:
    index_list.extend(outliers_detection(araba,feature))


# In[106]:


print(index_list)


# In[107]:


len(index_list)


# In[108]:


def remove(df,ls):
    ls = sorted(set(ls))
    araba = df.drop(ls)
    return araba


# In[109]:


df_cleaned = remove(araba,index_list)


# In[110]:


df_cleaned


# In[111]:


sns.boxplot(araba['Price'])


# In[112]:


sns.boxplot(df_cleaned['Price'])


# In[113]:


df_cleaned['Price'].plot()


# In[114]:


sns.kdeplot(df_cleaned['Price'])


# In[115]:


x = df_cleaned.drop('Price',axis = 1)


# In[116]:


x


# In[117]:


y = df_cleaned['Price']


# In[ ]:





# In[118]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[119]:


x_train


# In[120]:


y_train


# In[121]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[122]:


x_train = sc.fit_transform(x_train)


# In[123]:


x_test = sc.transform(x_test)


# In[124]:


from sklearn.tree import DecisionTreeRegressor


# In[125]:


dt = DecisionTreeRegressor()


# In[126]:


dt.fit(x_train,y_train)


# In[127]:


dt.score(x_train,y_train)


# In[128]:


dt.score(x_test,y_test)


# In[129]:


pred = dt.predict(x_test)


# In[130]:


pred


# In[131]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error


# In[132]:


mean_absolute_error(pred,y_test)


# In[133]:


np.sqrt(mean_squared_error(pred,y_test))


# In[134]:


mean_absolute_percentage_error(pred,y_test)


# In[135]:


from sklearn.metrics import r2_score


# In[136]:


r2_score(pred,y_test)


# In[137]:


from sklearn.model_selection import cross_val_score


# In[138]:


scores = cross_val_score(dt, x_train, y_train, scoring='r2', cv=5)


# In[139]:


scores


# In[140]:


scores = cross_val_score(dt, x_train, y_train, scoring='neg_mean_absolute_error', cv=5)


# In[141]:


(-scores)


# In[142]:


plt.figure(figsize=(12,8))
plt.scatter(y_test,pred)


# In[ ]:





# In[ ]:





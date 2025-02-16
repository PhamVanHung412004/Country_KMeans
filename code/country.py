import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from sklearn.cluster import KMeans, DBSCAN
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px

def show_boxplot(datas : pandas.core.frame.DataFrame, 
                 name_rows : list[str], 
                 ax : np.ndarray,
                 colors : list[str]) -> None:
    for i in range(len(name_rows)):
        # seaborn.boxplot(x=data[name_rows[i]],ax=ax[i],color=colors[2])
        seaborn.histplot(data=datas[name_rows[i]], ax=ax[i], kde=True, color=colors[2])

def show_histplot(data : pandas.core.frame.DataFrame, 
                  name_rows : list[str],
                  ax : np.ndarray,
                  colors : list[str]):
    
    for i in range(len(name_rows)):
        seaborn.histplot(x=data[name_rows[i]],ax=ax[i],color=colors[2])

class Read_dataset:
    def __init__(self,file_path_data : str) ->None:
        self.file_path_data = file_path_data
    def read_dataset(self) -> pandas.core.frame.DataFrame:
        data = pandas.read_csv(self.file_path_data)
        return data
    
class imshow_plt:
    def __init__(self, data : pandas.core.frame.DataFrame,
                 name_rows : list[str],
                 ax : np.ndarray, 
                 value_bool : int,
                 colors : list[str]) -> None:

        self.data = data
        self.name_rows = name_rows
        self.ax = ax
        self.value_bool = value_bool
        self.colors = colors

    def check(self) -> None:
        if (self.value_bool == 0):
            show_boxplot(self.data, self.name_rows, self.ax, self.colors)            
        else:   
            show_histplot(self.data, self.name_rows, self.ax, self.colors)            

def get_title(file_path : str) -> list:
    with open(file_path, mode="r") as file:
        data = file.readline()
        data = data.replace("\n","")
        data = data.split(",")
        data.pop(0)
        return data
    
def tranform_data(data : pandas.core.frame.DataFrame, name_cows : list[str]) -> dict:
    return data[name_cows].copy()

def show_heatmap(data : pandas.core.frame.DataFrame) -> None:
    fig=plt.figure(figsize=(15,8))
    seaborn.heatmap(data=abs(data.drop(["country"], axis=1).corr()),annot=True, square=True)

def check_result_very_good(data : pandas.core.frame.DataFrame, labels : np.ndarray) -> int:
    return silhouette_score(data,labels)

def check_model_clusters(data_frame : pandas.core.frame.DataFrame) -> None:
    fig,ax=plt.subplots(nrows=1,ncols=3,figsize=(18,8))
    print(ax)
    seaborn.scatterplot(data=data_frame, x='exports', y='income', hue='KMeans_labels', ax=ax[0])
    seaborn.scatterplot(data=data_frame, x='exports', y='gdpp', hue='KMeans_labels', ax=ax[1])
    seaborn.scatterplot(data=data_frame, x='child_mort', y='health', hue='KMeans_labels', ax=ax[2])

def check_country(data_frame : pandas.core.frame.DataFrame) -> None:
    labels_and_country = {0:'Cần giúp',
           1:'Có thể cần giúp',
           2:'Không cần giúp'}
    labels = data_frame["KMeans_labels"]
    labels_new = np.array([labels_and_country[labels[i]] for i in range(len(labels))])
    data_frame["KMeans"] = labels_new
    countrys = data_frame["country"]
    print("Các quốc gia cần giúp đỡ là: ")    
    for i in range(len(countrys)):
        if (labels_new[i] == "Cần giúp"):
            print(countrys[i])

def using_KMeans(
        data : pandas.core.frame.DataFrame, 
        name_rows : list[str],
        colors : list[str]) -> None:

    # Heamap
    show_heatmap(data)
    fig,ax = plt.subplots(
        nrows = 3,
        ncols = 3,
        figsize=(15,8),
        constrained_layout=True
    )

    #check noise(outlier)
    ax = ax.flatten()
    imshow_plt(data,name_rows,ax,0,colors).check()
    imshow_plt(data,name_rows,ax,1,colors).check()

    data_format = tranform_data(data,name_rows)
    data_format = pandas.DataFrame(data_format)

    # format data StandardScaler
    data_format = StandardScaler().fit_transform(data_format)

    # check clusters verry good using KElbowVisualizer 
    model = KMeans()
    show_screen = KElbowVisualizer(model, k = (1,10))
    fig = plt.figure(figsize=(10,8))
    show_screen.fit(data_format)
    show_screen.poof()

    # Train model with number clusters search 
    model_train = KMeans(n_clusters=3, random_state=42)
    model_train.fit(data_format)
    labels = model_train.labels_
    data_check = data.drop(["country"], axis=1)
    data["KMeans_labels"] = labels

    # Using Seaborn show cluster feature
    check_model_clusters(data)

    check_country(data) 

    # comment result score verry good using kmeans training model
    print("score max using kmeans:",check_result_very_good(data_check,labels))
    
def using_DBSCAN(data : pandas.core.frame.DataFrame) -> None:
    data_new = data[['child_mort', 'imports', 'gdpp']]
    data_new = StandardScaler().fit_transform(data_new)
    for eps in [i / 10 for i in range(2,7)]:
        for min_samples in range(7,10):
            print(f'\neps : {eps}')
            print(f'min samples: {min_samples}')
            dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data_new)
            labels = dbscan.labels_
            if (len(set(labels)) == 1):
                continue
            score = silhouette_score(data_new, labels)
            print(f'clusters present: {np.unique(labels)}')
            print(f'clusters sizes: {np.bincount(labels + 1)}')
            print(f'Silhouette Score: {score}')
    
def main():
    colors = ['#DB1C18','#DBDB3B','#51A2DB']
    
    #read and get data
    file_path_data = r"D:\D\project_github\Country_KMeans\dataset\Country-data.csv"         
    file_path_data_dict = r"D:\D\project_github\Country_KMeans\dataset\data-dictionary.csv"
    
    #check error read file csv
    try: 
        # Using KMeans
        data = Read_dataset(file_path_data).read_dataset()
        data_dict = Read_dataset(file_path_data_dict).read_dataset()
        name_rows = get_title(file_path_data)
        using_KMeans(data, name_rows,colors)
        
        # Bounos using DBSCAN check outlier 
        using_DBSCAN(data)

    except:
        print("Error read file data")
main()
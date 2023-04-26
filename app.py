from flask import Flask,render_template,request
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import warnings

details = pickle.load(open('detailed.pkl','rb'))
pt = pickle.load(open('pt.pkl','rb'))
similarity_scores = pickle.load(open('similarity_score.pkl','rb'))
top20 = pickle.load(open('top20streamed.pkl','rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html',song_name=list(details['title'].values),
                          artist=list(details['artist'].values),
                          year=list(details['year'].values),
                          chart=list(details['chart'].values),
                          )

@app.route('/home')
def home():

    return render_template('home.html',titled=list(top20['title'].values),
                           yeard = list(top20['year'].values),
                           regiond=list(top20['region'].values)
                           )

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_songs',methods=['POST'])
def recommend():
    user_input=request.form.get('user_input')
    user_input=user_input.title()
    index=np.where(pt.index==user_input)[0][0]
    similar_items=sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:8]#return list of index
    
    data=[]
    for i in similar_items:
        item=[]
        temp_df=details[details['title']==pt.index[i[0]]]
        
        item.extend(list(temp_df.drop_duplicates('title')['title']))
        item.extend(list(temp_df.drop_duplicates('artist')['artist']))
        
        data.append(item) 
        print(data)
    return render_template('recommend.html',data=data)

@app.route('/crmdf',methods=['POST','DELETE','GET'])

def crmdf():
    regions=str(request.form.get('region_input'))
    year=str(request.form.get('year_input'))
    song=request.form.get('song_input')
    # path = 'datasets'+'//'+regions+'//'+year+'//'+'.csv'
    strr = 'C:\\Users\\ADMIN\\Desktop\\Project\\datasets\\regions'
    sl = '\\'
    path = str(strr + sl + str(regions) + sl + str(year) + ".csv")
    
    df = pd.read_csv(path)
    df1 = df.drop(['Unnamed: 0.1','Unnamed: 0','region','rank','year'],axis=1)#feature removal(here region is removed because we have load data of indian region)
    df2=df1[['artist','chart','trend']]
    for columns in df2.columns:
        ordinal_label3={k:i for i, k in enumerate(df2[columns].unique(),0)}
        df2[columns] = df2[columns].map(ordinal_label3)#labelencoder
    df1=df1.drop(['artist','chart','trend'],axis=1)
    df4 = pd.concat([df1,df2],axis=1)
    for column in ['streams','artist','chart','trend']:
        df4[column] = (df4[column] - df4[column].min()) / (df4[column].max() - df4[column].min())
    pt=df4.pivot_table(index=['title'],values=['streams','artist','chart','trend'])#pivoting the table and setting index and values
    similarity_scores=cosine_similarity(pt)
    
    index=np.where(pt.index==song)[0][0]
    similar_items=sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:8]
    data=[]
    for i in similar_items:
        item=[]
        temp_df=df[df['title']==pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('title')['title']))
        item.extend(list(temp_df.drop_duplicates('artist')['artist']))
        data.append(item)
    return render_template('index.html',data=data)


if __name__=='__main__':
    app.run(debug=True)
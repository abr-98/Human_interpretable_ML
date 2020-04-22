import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report,f1_score,precision_score,recall_score,accuracy_score,confusion_matrix

################## to get datasets###########################################
def get_datasets(target_column):
    df=pd.read_csv("features_embedded.csv")
    df['norm_road']=df[['Signal', 'bus_stop', 'Turn','Congestion']].apply(lambda e: 1 if e[0]==e[1]==e[2]==e[3]==0 else 0,axis=1)

    labels=df[[target_column]].values
    features=df.drop(columns=['norm_road','Signal', 'bus_stop', 'Turn','Congestion']).values

    oversample = SMOTE()
    features,labels = oversample.fit_resample(features,labels)
    labels=labels.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=42)

    s=MinMaxScaler()
    X_train=s.fit_transform(X_train)
    X_test=s.transform(X_test)

    return X_train, X_test, y_train, y_test,s

#################training column selection model ##########################
def get_custom_model(columns):
    input_layer=tf.keras.layers.Input((17,))
    net=tf.keras.layers.Lambda(lambda e:tf.transpose(tf.nn.embedding_lookup(tf.transpose(e),columns)))(input_layer)
    net=tf.keras.layers.Dense(128,'relu')(net)
    net=tf.keras.layers.Dense(128,'relu')(net)
    net=tf.keras.layers.Dense(128,'relu')(net)
    net=tf.keras.layers.Dense(128,'relu')(net)
    out=tf.keras.layers.Dense(1,'sigmoid')(net)
    model=tf.keras.Model(inputs=[input_layer],outputs=[out],name='Custom_model')
    return model

################# to get models###############################################
def Get_trained_model_on_data(training_columns,X_train,y_train,X_test,y_test,data_class=None):
    with tf.name_scope(data_class):
        model=get_custom_model(training_columns)#[0, 2, 10, 9, 14, 11, 13, 12, 6, 16, 15, 5, 7]) #other columns are deleted runtime
        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

        callback=\
        tf.keras.callbacks.ModelCheckpoint(filepath='./saved/{}_model.h5'.format(data_class),
                                           monitor='val_acc',
                                           verbose=0,
                                           save_best_only=True,
                                           save_weights_only=False,
                                           mode='auto',
                                           save_freq='epoch')
    history=model.fit(X_train,
                      y_train,
                      batch_size=32,
                      epochs=1000,
                      validation_data=(X_test,y_test),
                      callbacks=[callback])

    model=tf.keras.models.load_model('./saved/{}_model.h5'.format(data_class))
    
    return model,history

################ to get numeric reports###########################################
def get_numeric_reports(model,X_train,y_train,X_test,y_test):
    training_acc=model.evaluate(X_train,y_train)[1]
    testing_acc=model.evaluate(X_test,y_test)[1]
    pred=np.round(model.predict(X_test)) #this model doesnot have model.predict_classes
    classify_report=classification_report(y_test.flatten(),pred.flatten())
    return training_acc,testing_acc,classify_report

###############to get all metrics ################################################
def get_metrics(model,target_column,prop_column,testing_df):
    test_df=testing_df[testing_df[prop_column].round()==1]
    feat=test_df.drop(columns=[target_column]).values
    lab=test_df[[target_column]].values
    pred=np.round(model.predict(feat)) #this model doesnot have model.predict_classes
    metrics=[accuracy_score(lab,pred),precision_score(lab,pred),recall_score(lab,pred),f1_score(lab,pred)]
    return metrics

############## to plot a columns metrics(Bar plots)###########################
def plot_metrics(model,target_column,check_props,testing_df):
    fig=plt.figure(figsize=(15,10))
    for i,prop in enumerate(check_props):
        ax1=fig.add_subplot(3,4,i+1)
        ax1.set_xlabel(prop)
        ax1.set_ylabel('Score')
        ax1.bar(['acc','precision','recall','f1-score'],get_metrics(model,target_column,prop,testing_df),color=sns.color_palette(),width=0.4)
        #ax1.tick_params(axis='x', labelrotation=45)
        ax1.set_ylim(0,1)
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')
    fig.suptitle(target_column,fontsize=16)
    #fig.savefig('./picture/{}_metric_barplot.png'.format(target_column))
    plt.close()
    return fig
############## to plot line plots ##############################################
def line_plot_metrics(model,target_column,check_props,testing_df):
    l=[]
    for prop in check_props:
        l.append(get_metrics(model,target_column,prop,testing_df))
    metric_arr=np.array(l)
    fig=plt.figure(figsize=(7,5))
    fig.suptitle(target_column,fontsize=16)
    ax=fig.add_subplot(111)
    ax.plot(check_props,metric_arr[:,0],c='blue',label='accuracy',linewidth=2,alpha=0.7,marker='^',linestyle='--')
    ax.plot(check_props,metric_arr[:,1],c='orange',label='precision',linewidth=2,alpha=0.7,marker='v',linestyle='--')
    ax.plot(check_props,metric_arr[:,2],c='green',label='recall',linewidth=2,alpha=0.7,marker='<',linestyle='--')
    ax.plot(check_props,metric_arr[:,3],c='red',label='f1-score',linewidth=2,alpha=0.7,marker='>',linestyle='--')
    ax.tick_params(axis='x', labelrotation=60)
    ax.set_ylim(0.5,1)
    ax.legend()
    #fig.savefig('./picture/{}_metric_lineplot.png'.format(target_column))
    plt.close()
    return fig

############### to plot all informations ########################################
def plot_all_information(model,target_column,X_test,y_test):
    f_cols=\
    ['next_stop_distance', 'total_waiting_time', 'wifi_count', 'honks',
           'rsi', 'zone_highway', 'zone_market_place', 'zone_normal_city',
           'time_level_1', 'time_level_2', 'time_level_3', 'time_level_4',
           'Population_density_dense', 'Population_density_medium',
           'Population_density_sparse', 'Weekend/day_Week-day',
           'Weekend/day_Week-end']

    testing_df=pd.DataFrame(np.concatenate([X_test,y_test],axis=1),columns=f_cols+[target_column])

    check_props=[
    'time_level_1',
    'time_level_2',
    'time_level_3',
    'time_level_4',
    'zone_highway',
    'zone_market_place',
    'zone_normal_city',
    'Population_density_dense',
    'Population_density_medium',
    'Population_density_sparse',
    'Weekend/day_Week-day',
    'Weekend/day_Week-end']

    fig1=plot_metrics(model,target_column,check_props,testing_df)
    fig2=line_plot_metrics(model,target_column,check_props,testing_df)
    return fig1,fig2
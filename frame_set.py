
# Name :Roshan Mishra
# Git_id : Roshanmishra150

from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
from time import *
import csv
screen = Tk()

##############################################################  =>  Importing Libraries for Projects
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


##############################################################  =>  Screen View
screen.geometry('900x600+390+100')
screen.title("Major Crime Detection")
screen.minsize(800,600)
screen.maxsize(800,600)
screen.configure(bg="#c2c4c4")

##############################################################  =>  Dataset Import
messagebox.showwarning(title="Enter Dataset",message=" Please Insert Your Dataset In CSV File Format Only .. ",parent=screen)
sleep(0.2)
filename = filedialog.askopenfilename(title = "Select file",filetypes = (("CSV Files","*.csv"),))
df = pd.read_csv(filename,sep=',')


##############################################################  =>  Heading 
heading = Label(screen,text=" ******** Major Crime Detection *********",font=("arial",25,"italic"),fg="brown")
heading.grid(row=0,column=1,columnspan=4,padx=(3,3),pady=30)


##############################################################  =>  Buttons

btn1 = Button(screen,text="View Dataset",padx=45,bg="#99ccff",bd=10,font=('arial',20,'italic'),command=lambda:Dataset(" Your Dataset", df))
btn1.grid(row=1,column=1,padx=(50,30),pady=30)

btn2 = Button(screen,text="Total Crime 2014-2019",bg="#66ff99",bd=10,font=('arial',20,'italic'),command=lambda:linebar())
btn2.grid(row=1,column=2,padx=(15,30),pady=30)

btn3 = Button(screen,text="Premise Type Crime",bg="#ff99cc",bd=10,font=('arial',20,'italic'),command=lambda:pichart())
btn3.grid(row=2,column=1,padx=(50,30),pady=30)

btn4 = Button(screen,text="Top Major crime ..",padx=(1),bg="#d9b3ff",bd=10,font=('arial',20,'italic'),command=lambda:major())
btn4.grid(row=2,column=2,padx=(20,30),pady=30)

btn5 = Button(screen,text="Top 20 Crimes",padx=45,bg="#03fcbe",bd=10,font=('arial',20,'italic'),command=lambda:Top20())
btn5.grid(row=3,column=1,padx=(50,30),pady=30)

btn6 = Button(screen,text="Predictions",padx=75,bg="red",bd=10,font=('arial',20,'italic'),fg="white",command=lambda:prediction())
btn6.grid(row=3,column=2,padx=(20,30),pady=30)


##############################################################  =>  Functions

def Dataset(winTitel,mesShow):
    global df
    newWindow = Toplevel(screen)
    newWindow.title(winTitel)
    newWindow.geometry("800x800")
    Label(newWindow,text=mesShow,font=("arial",20,"bold")).pack(pady=20)
    


def linebar():
    # Creating a Countplot
    df2 = df[df['occurrenceyear'] > 2013]
    yearwise_total_crime = df2.groupby('occurrenceyear').size()
    newWindow = Toplevel(screen)
    newWindow.title("Total Number of Criminal Cases throughout 2014 to 2019")
    newWindow.geometry("400x400")
    Label(newWindow,text=yearwise_total_crime,font=("arial",20,"bold")).pack(pady=20)

    plt.figure(figsize=(13,6))
    ct = yearwise_total_crime.sort_values(ascending=True)
    ax = ct.plot.line()
    ax.set_xlabel('Year')
    ax.set_ylabel('Total Number of Criminal Cases throughout 2014 to 2019')
    ax.set_title('Yearwise total Criminal Cases throughout 2014 to 2019',color = 'red',fontsize=25)
    ax.grid(linestyle='-')
    plt.show()


def pichart():
    # Proportion of crime according to premisetype
    premise_type = df.groupby('premisetype').size()
    premise_type.head()
    labels = ['Outside','Apartment','Commercial','House','Other']
    count = [54253,49996,41081,37927,23178]
    explode = (0, 0, 0, 0, 0) 

    fig, ax = plt.subplots(figsize = (9,6))
    ax.pie(count, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title("Proportion of Crime according to Premise Type", color='red', fontsize=25)
    plt.show()


def major():
    major_crime_indicator = df.groupby('MCI',as_index=False).size()
    # print(major_crime_indicator)
    newWindow = Toplevel(screen)
    newWindow.title("Top Major crime ..")
    newWindow.geometry("400x400")
    Label(newWindow,text=major_crime_indicator,font=("arial",20,"bold")).pack(pady=40)

    plt.subplots(figsize = (9, 5))
    ct = major_crime_indicator.sort_values(ascending = False)
    ax = ct.plot.bar()
    ax.set_xlabel('Offence')
    ax.set_ylabel('Total Number of Criminal Cases from 2014 to 2019')
    ax.set_title('Major Crime Indicator',color = 'red',fontsize=25)
    plt.show()


def Top20():
    location_group = df.groupby('Neighbourhood',as_index=False).size().sort_values(ascending = False).head(20)
    newWindow = Toplevel(screen)
    newWindow.title("Top 20 most crime ..")
    newWindow.geometry("800x800")
    Label(newWindow,text=location_group,font=("arial",20,"bold")).pack(pady=40)

    plt.subplots(figsize = (15, 8))
    ct = location_group.sort_values(ascending = False)
    ax = ct.plot.bar()
    ax.set_xlabel('Neighbourhoods')
    ax.set_ylabel('Number of occurences')
    ax.set_title('Neighbourhoods with Most Crimes',color = 'red',fontsize=25)
    plt.show()



##############################################################  =>  Data Preprocessing
# Columns for the models
col_list = ['occurrenceyear',	'occurrencemonth','occurrenceday','occurrencedayofyear','occurrencedayofweek','occurrencehour','MCI',	'Division',	'Hood_ID','premisetype']

# New dataframe from columns
df2 = df[col_list]
df2 = df2[df2['occurrenceyear'] > 2013]

#Factorize dependent variable column:
crime_var = pd.factorize(df2['MCI'])
df2['MCI'] = crime_var[0]
definition_list_MCI = crime_var[1]

#factorize independent variables:
premise_var = pd.factorize(df2['premisetype'])
df2['premisetype'] = premise_var[0]
definition_list_premise = premise_var[1] 

#factorize occurenceyear:
year_var = pd.factorize(df2['occurrenceyear'])
df2['occurrenceyear'] = year_var[0]
definition_list_year = year_var[1] 

#factorize occurencemonth:
month_var = pd.factorize(df2['occurrencemonth'])
df2['occurrencemonth'] = month_var[0]
definition_list_month = month_var[1] 

#factorize occurenceday:
day_var = pd.factorize(df2['occurrenceday'])
df2['occurenceday'] = day_var[0]
definition_list_day = day_var[1] 

#factorize occurencedayofweek:
dayweek_var = pd.factorize(df2['occurrencedayofweek'])
df2['occurrencedayofweek'] = dayweek_var[0]
definition_list_day = dayweek_var[1] 

#factorize division:
division_var = pd.factorize(df2['Division'])
df2['Division'] = division_var[0]
definition_list_division = division_var[1] 

#factorize HOOD_ID:
hood_var = pd.factorize(df2['Hood_ID'])
df2['Hood_ID'] = hood_var[0]
definition_list_hood = hood_var[1] 

#factorize occurencehour:
hour_var = pd.factorize(df2['occurrencehour'])
df2['occurrencehour'] = hour_var[0]
definition_list_hour = hour_var[1] 

#factorize occurencedayofyear:
dayyear_var = pd.factorize(df2['occurrencedayofyear'])
df2['occurrencedayofyear'] = dayyear_var[0]
definition_list_dayyear = dayyear_var[1] 


##############################################################  =>  Testing And Training

#set X and Y:
X = df2.drop(['MCI'],axis=1).values
y = df2['MCI'].values

#split the data into train and test sets for numeric encoded dataset:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)

#need to OneHotEncode all the X variables for input into the classification model:
# binary_encoder = OneHotEncoder(sparse=False,categories='auto')
# encoded_X = binary_encoder.fit_transform(X)
# X_train_OH, X_test_OH, y_train_OH, y_test_OH = train_test_split(encoded_X, y, test_size = 0.25, random_state = 21)


def prediction():
    messagebox.showwarning(screen,message=" ***** Prediction may take 2 to 3 minutes ,Please Wait.... and Don't Click Any Were else.  Just Press OK . ",parent=screen)

    # Numeric Encoded Model
    classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    p1 = ("Accuracy of Random Forest : ",accuracy_score(y_test, y_pred))
    p2 = (confusion_matrix(y_test, y_pred)) 
    p3 = (classification_report(y_test,y_pred, target_names=definition_list_MCI)) 
    newWindow = Toplevel(screen)
    newWindow.title(" Predictions of our Model")
    newWindow.geometry("700x700")
    Label(newWindow,text=p1,font=("arial",20,"bold")).pack(pady=40)
    Label(newWindow,text=p2,font=("arial",20,"bold")).pack(pady=40)
    Label(newWindow,text=p3,font=("arial",20,"bold")).pack(pady=40)

    #One Hot Encoded Model

    # classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)
    # classifier.fit(X_train_OH, y_train_OH)
    # y_pred_OH = classifier.predict(X_test_OH)

    # p3=("Accuracy of Random Forest with OneHotEncoder : ",accuracy_score(y_test, y_pred))
    # p4=(confusion_matrix(y_test_OH, y_pred_OH)) 
    # p5=(classification_report(y_test_OH,y_pred_OH, target_names=definition_list_MCI))  
    # newWindow = Toplevel(screen)
    # newWindow.title(" Predictions of our Model")
    # newWindow.geometry("400x600")
    # Label(newWindow,text=p3,font=("arial",20,"bold")).pack(pady=40)
    # Label(newWindow,text=p4,font=("arial",20,"bold")).pack(pady=40)
    # Label(newWindow,text=p5,font=("arial",20,"bold")).pack(pady=40)



##############################################################  =>  End Screen
screen.mainloop()



##############################################################  =>  Dataset required things
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
from keras.utils import dataset_creator
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from scipy import stats
plt.style.use('fivethirtyeight')
# libraries used for Chatbot

def first_string(Word):
  """ In a multi word string, uses first word as reference """
  Cut= Word.split()
  return Cut

#gets the most recent closing price
def curr_clos_price(company, df):
    #Time when you want to START the data from 
    # convert to unix timestamp.
    start = pd.to_datetime([df['Date'].iat[-2]]).astype(int)[0]//10**9 

    #Time when you want to END the data from
    # convert to unix timestamp.
    end = pd.to_datetime([df['Date'].iat[-1]]).astype(int)[0]//10**9 

    url = 'https://query1.finance.yahoo.com/v7/finance/download/' + company + \
    '?period1=' + str(start) + '&period2=' + str(end) + '&interval=1d&events=history'
    quote2 = pd.read_csv(url)
    curr_quote = quote2['Close']
    return curr_quote

#Sets dataframe
def set_df(company, start_date, end_date):
    #Time when you want to START the data from 
    # convert to unix timestamp.
    start = pd.to_datetime(['2012-01-01']).astype(int)[0]//10**9 

    #Time when you want to END the data from
    # convert to unix timestamp.
    end = pd.to_datetime(['2019-01-01']).astype(int)[0]//10**9 

    url = 'https://query1.finance.yahoo.com/v7/finance/download/' + company +\
    '?period1=' + str(start) + '&period2=' + str(end) + '&interval=1d&events=history'
    df = pd.read_csv(url)

    #creates a new data frame with just the closing prices
    data = df.filter(['Close'])

    dataset = data.values

    training_data_len = math.ceil(len(dataset)* 0.8)

    training_data_len

    #scales the data
    scaler = MinMaxScaler(feature_range = (0,1))
    scaled_data = scaler.fit_transform(dataset)

    return scaled_data, training_data_len, data, scaler, df, dataset

#Gets the predicted price for next closing price
def pred_clos_price(company, start_date, end_date):
    #Time when you want to START the data from 
    # convert to unix timestamp.
    start = pd.to_datetime(['2012-01-01']).astype(int)[0]//10**9 

    #Time when you want to END the data from
    # convert to unix timestamp.
    end = pd.to_datetime(['2022-11-30']).astype(int)[0]//10**9 

    url = 'https://query1.finance.yahoo.com/v7/finance/download/' + company +\
     '?period1=' + str(start) + '&period2=' + str(end) + '&interval=1d&events=history'
    quote = pd.read_csv(url)

    #creates scaled data frame for the last 60 days
    new_df = quote.filter(['Close'])
    last_60_days = new_df[-60:].values
    last_60_days_scaled = scaler.transform(last_60_days)

    X_test = []

    X_test.append(last_60_days_scaled)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    #Calculates the predicted price
    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    return pred_price


Intro = input(
    """To get started we would like to ask what is your name?
    """)
User_name = first_string(Intro)
print(f'Hello {User_name[0]} and welcome to our Chatbot!')

# Introduction to Chatbot, creates a name to refer to the User as
base = True
while base == True:
    company = input("""
        What is the ticker of the company you should like to predict their closing
        price? A couple reccomendations are TSLA, AAPL, GOOGL, etc.
        """)  
    start_date = input("""
        What is the starting date you would like to take into consideration? Please
        format the date like this year-month-day
        """)
    end_date = input("""
        What is the last date you would like to take into consideration? Please
        format the date like this year-month-day
        """)
    try:
        scaled_data, training_data_len, data, scaler, df, dataset = set_df(company, start_date, end_date)
        break
    except:
        print("""
        Sorry one of your inputs is not valid.
        Please try again
        """)
        continue


#Sets the amount of days it will predict with the training data
prediction_days = 60
#Train the data
train_data = scaled_data[0:training_data_len, :]
x_train = []
y_train = []

for x in range(prediction_days, len(train_data)):
    x_train.append(train_data[x - prediction_days:x, 0])
    y_train.append(train_data[x,0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


again = True
while again == True:
# To continue the commands menu until the user decides to end chatbot
  Commands = input("""
  Below are the options at hand, please specify which command you would like \
  chatbot to execute:
  A. Descriptive Statistics
  B. T-test and P-Value
  C. Graph Function
  D. Stock Predictor
  E. End
  """)
  # list of commands chat bot executes for user to decide and resets after 
  # each use to the command menu

  while Commands not in ['a', 'b', 'c', 'd', 'e', 'A','B','C','D','E']: 
    print('Error')
    Commands = input("""
  Below are the options at hand, please specify which command you would like \
  chatbot to execute:
  A. Descriptive Statistics
  B. T-test and P-Value
  C. Graph Function
  D. Stock Predictor
  E. End
  """)
  if Commands.lower() == 'a':
    descriptive = input("""
    What would you like to know about the dataset? 
    A. Minimum Values
    B. Maximum Values
    C. Mean Values
    D. Variance
    E. Standard Deviatione
    """)
    while descriptive not in ['a', 'b', 'c', 'd', 'e', 'A','B','C','D', 'E']: 
      # error message will be produced when one of the above selections of 
      # answers is not chosesn and the user will have to choose again
      # from the menu above
    
      print('Error')
      descriptive = input("""
    What would you like to know about the dataset? 
    A. Minimum Values
    B. Maximum Values
    C. Mean Values
    D. Variance
    E. Standard Deviatione
    """)
      
      # descriptive statistics for the user to view, minimum values, maximum
      # variance, standard deviation, and mean
    if descriptive.lower() == 'a':
      print(df.min())
    elif descriptive.lower() == 'b':
      print(df.max())
    elif descriptive.lower() == 'c':
      print(df.mean())
    elif descriptive.lower() == 'd':
      print(df.var())
    elif descriptive.lower() == 'e':
      print(df.std())
  if Commands.lower() == 'b':
    t_test = input("""
    The follwing are options available for this chatbot:
    A: T-test and p-value between open and close
    B: T-test and p-value between open and high
    C: T-test and p-value between open and low
    D: T-test and p-value between close and high
    E: T-test and p-value between close and low
    F: T-test and p-value between high and low
    """)
    while Commands not in ['a', 'b', 'c', 'd','e', 'f', 'A', 'B' , 'C' ,'D', 'E','F']: 
      print('Error')
      t_test = input("""
    The follwing are options available for this chatbot:
    A: Correlation and p-value between open and close
    B: Correlation and p-value between open and high
    C: Correlation and p-value between open and low
    D: Correlation and p-value between close and high
    E: Correlation and p-value between close and low
    F: Correlation and p-value between high and low
    """) 
      # t-test used on data to determine how big the difference is between the 
      # values, higher p-values represent higher significance
    
    if t_test.lower() == 'a':
      print(stats.ttest_ind(df['Open'],df['Close']))
    elif t_test.lower() == 'b':
        print(stats.ttest_ind(df['Open'],df['High']))
    elif t_test.lower() == 'c':
        print(stats.ttest_ind(df['Open'],df['Low']))
    elif t_test.lower() == 'd':
        print(stats.ttest_ind(df['Close'],df['High']))
    elif t_test.lower() == 'e':
        print(stats.ttest_ind(df['Close'],df['Low']))
    elif t_test.lower() == 'f':
        print(stats.ttest_ind(df['High'],df['Low']))

  if Commands.lower() == 'c':
    graph = input("""
    Which of the follwing options would you like to see?
    A: Histoplot of Open
    B: Pointplot of Open and Close
    C: Lineplot of All
    D: Histoplot of Close
    E: Pointplot of High and Low
    """)
    # graphs of the data where histoplots are used to view the count of certain
    # values within the data set. Pointplot to view comparison of two values
    # Pointplot to see all values graph alongside one another
    
    while graph not in ['a', 'b', 'c', 'd', 'e', 'A', 'B', 'C' ,'D', 'E']: 
      print('Error')
      graph = input("""
    Which of the follwing options would you like to see?
    A: Histoplot of Open
    B: Pointplot of Open and Close
    C: Lineplot of All
    D: Histoplot of Close
    E: Pointplot of High and Low
    """)
    if graph.lower() == 'a':
        histO = sns.histplot( x = 'Open', data = df, bins = 100)
        plt.show(histO)  
    elif graph.lower() == 'b':
        pointOC = sns.pointplot(x= 'Open', y = 'Close', data=df)
        plt.show(pointOC)
    elif graph.lower() == 'c':
        line = sns.lineplot(data=df, linewidth = 10)
        plt.show(line)
    elif graph.lower() == 'd':
        histC = sns.histplot( x = 'Close', data = df, bins = 100)
        plt.show(histC)
    elif graph.lower() == 'e':
        pointHL = sns.pointplot(x= 'High', y = 'Low', data=df)
        plt.show(pointHL)
      
  if Commands.lower() == 'd':
    
      print(f'Please give us a second to produce the prediction...')

      #Build Model
      model = Sequential()

      #More units the longer you will have to train 
      model.add(LSTM(50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
      model.add(LSTM(50, return_sequences = False))
      model.add(Dense(25))
      model.add(Dense(1))


      model.compile(optimizer = 'adam', loss = 'mean_squared_error')

      #Model will see 32 units at once
      model.fit(x_train, y_train, batch_size = 32, epochs = 1)

      pred_price = pred_clos_price(company, start_date, end_date)
      curr_quote = curr_clos_price(company, df)

      print(f'\033[1;32mThe most recent closing price is {curr_quote}.')
      print(f'The predicted closing price for {company} tomorrow is {pred_price}. ')
      response = input(f'Would you like to see the Closing Price Stock Price Graph of {company}? ')
      if response.lower() == 'yes':
          
          #Graph with prediction
          test_data = scaled_data[training_data_len - prediction_days: , :]
          x_test = []
          y_test = dataset[training_data_len:, :]

          for x in range(prediction_days, len(test_data)):
              x_test.append(test_data[x - prediction_days: x, 0 ])

          x_test = np.array(x_test)
          x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

          predictions = model.predict(x_test)
          predictions = scaler.inverse_transform(predictions)

          train = data[:training_data_len]
          valid = data[training_data_len:]
          valid['Predictions'] = predictions

          plt.figure(figsize = (16,8))
          plt.title('Close Price History')
          plt.xlabel('Date', fontsize = 18)
          plt.ylabel('Close Price USD ($)', fontsize = 18)
          plt.plot(train['Close'])
          plt.plot(valid[['Close', 'Predictions']])
          plt.legend(['Train', 'Val', 'Predictions'], loc = 'lower right')
          plt.show()
  if Commands.lower() == 'e':
    again = False
    # ends chat bot and then prints message saying thanks for chatting
   
print('Thanks for using the Chatbot!')
import requests
import json
from bs4 import BeautifulSoup
import datetime
import matplotlib.pyplot as plt
import numpy as np
import time
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
np.random.seed(2018)

def get_data(s_code,t_type,total):
    
    res = requests.get("https://tw.quote.finance.yahoo.net/quote/q?type=ta&perd="+t_type+"&sym="+s_code)
    soup = BeautifulSoup(res.text,"html.parser")
    temp = str(soup)[5:-2]
    test = json.loads(temp)
    
    for index in test["ta"]:
        #range of date
        #if index["t"] > 20180601:
            #print("date = "+str(index["t"]))
            #print("price = "+str(index["c"]))
            total.append(index["c"])
            
    return total

def prepare_data(stock, seq_len, ratio):

    total = []
    
    total = get_data(stock,"m",total)
    total = get_data(stock,"w",total)
    total = get_data(stock,"d",total)
    
    pred_test = []
    result = []
    for index in range(len(total)-seq_len+1):
        result.append(total[index:index+seq_len])
        if index == len(total)-seq_len:
            pred_test = total[index:index+seq_len]
    two_mouth_ago = total[-40]
    result = np.array(result).astype('float64')
    
    result_mean = result.mean()
    result -= result_mean
    print("Shift : ", result_mean)
    print("Data : ", result.shape)
    
    row = int(round(ratio * result.shape[0]))
    train = result[:row, :]
    np.random.shuffle(train)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = result[row:, :-1]
    y_test = result[row:, -1]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    return X_train, y_train, X_test, y_test, result_mean, pred_test, two_mouth_ago

def build_model():
    
    model = Sequential()
    layers = [1, 100, 200, 1]
    model.add(LSTM(
            layers[1],
            input_shape=(None, 1),
            return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(
            layers[2],
            return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(
            layers[3]))
    model.add(Activation("linear"))
    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    print("Model Compilation Time : ", time.time() - start)
    
    return model

def run_network(model, X_train, y_train, X_test, y_test, result_mean):
    epochs = 80
    t_start = time.time()
    model.fit(
        X_train, y_train,
        batch_size=256, epochs=epochs, verbose=0, validation_split=0.05)
    
    predicted = model.predict(X_test)
    predicted = np.reshape(predicted, (predicted.size,))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    y_test += result_mean
    predicted += result_mean
    ax.plot(y_test,label='real')
    plt.plot(predicted,label='pred')
    plt.legend(loc='upper left')
    plt.show()
    print('Training duration (s) : ', time.time() - t_start)
    return model, y_test, predicted

def pred_next(model,pred_test,two_mouth_ago):
    count=0
    for index in range(10):
        pred_test = np.array(pred_test).astype('float64')
        pred_test_mean = np.mean(pred_test)
        pred_test -= pred_test_mean
        pred_test = np.reshape(pred_test, (1, pred_test.shape[0], 1))
        pred_next_d = model.predict(pred_test)
        pred_next_d = np.reshape(pred_next_d, (pred_next_d.size,))
        pred_test = pred_test[:,1:,:]
        pred_test += pred_test_mean
        pred_next_d += pred_test_mean
        temp = pred_next_d[0]/pred_test[0,-1,0]
        if temp > 1:
            count += 1
        if temp > 1.1:
            pred_next_d[0] = pred_test[0,-1,0]*1.1
        elif temp <0.9:
            pred_next_d[0] = pred_test[0,-1,0]*0.9
        pred_test = np.append(pred_test, pred_next_d)
    range_rate = pred_test[-1]/pred_test[-11]
    t_m_rate = pred_test[-11]/two_mouth_ago
    plt.plot(pred_test[-11:],label='pred_next_10day')
    plt.legend(loc='upper left')
    plt.show()
    return count, range_rate, t_m_rate

def find_industry(final):
    indus_list = []
    print("Recommend and trend")
    for index in range(len(final)):
        #公司名稱
        print(final[index].text[5:])
        #股票編號
        print(final[index].text[0:4])    
        #產業類別
        res = requests.get("https://tw.stock.yahoo.com/d/s/company_"+final[index].text[0:4]+".html")
        soup = BeautifulSoup(res.text,"html.parser")
        tempClass = soup.find('td',width='84',align='left')
        #print(tempClass.get_text())
        print("產業類別: "+tempClass.get_text())
        #營利比重->最高占比
        tempRp = soup.find_all('td','yui-td-left',colspan='3')
        #print(tempRp[4].get_text())
        bestRp = tempRp[4].get_text().split('、')
        #print(bestRp[0])
        #print(bestRp[0].strip('1234567890.%'))
        print("最高營收占比項目: "+bestRp[0].strip('1234567890.%'))
        if tempClass.get_text() != "其他":
            indus_list.append(tempClass.get_text())
        else:
            indus_list.append(bestRp[0].strip('1234567890.%'))
    return indus_list
        
def user_ask(indus_list, model):
    while 1:
        ask_stock = str(input("input 'stock no.' to analysis,input 'exit' to leave: "))
        if ask_stock == "exit":
            break
        try:
            stock_no = int(ask_stock)
        except ValueError:
            print("Not a stock no. which you input")
            continue
        res = requests.get("https://tw.stock.yahoo.com/d/s/company_"+str(stock_no)+".html")
        soup = BeautifulSoup(res.text,"html.parser")
        tempName = soup.find('td',width='630',align='left')
        try:
            print("股票名稱: "+tempName.get_text().split('\n')[0])
        except AttributeError:
            print("stock not found")
            continue
        tempClass = soup.find('td',width='84',align='left')
        #print(tempClass.get_text())
        print("產業類別: "+tempClass.get_text())
        #營利比重->最高占比
        tempRp = soup.find_all('td','yui-td-left',colspan='3')
        #print(tempRp[4].get_text())
        bestRp = tempRp[4].get_text().split('、')
        #print(bestRp[0])
        #print(bestRp[0].strip('1234567890.%'))
        print("最高營收占比項目: "+bestRp[0].strip('1234567890.%'))
        X_train, y_train, X_test, y_test, result_mean, pred_test, two_mouth_ago = prepare_data(ask_stock, seq_len=20, ratio=0.9)
        model, y_test, predicted = run_network(model, X_train, y_train, X_test, y_test, result_mean)
        count, range_rate,t_m_rate = pred_next(model,pred_test,two_mouth_ago)
        for index in indus_list:
            if tempClass.get_text() == "其他":
                if index == bestRp[0].strip('1234567890.%'):
                    print("This industry may be trend")
                    break
            else:
                if index == tempClass.get_text():
                    print("This industry may be trend")
                    break
                    
        if count >= 5 and range_rate > 1.05 and t_m_rate > 1.1:
            print("Can buy")
        elif range_rate > 1.05:
            print("Can consider")
        else:
            print("No recommend")
        
def main():
    global_start_time = time.time()
    print("Today : ",datetime.date.today())
    url = "https://tw.stock.yahoo.com/d/i/rank.php?t=up&e=tse#&n=100"
    res = requests.get(url)
    soup = BeautifulSoup(res.text,"html.parser")
    stock_name = soup.find_all('td','name')
    
    model = build_model()
    
    final = []
    for index in stock_name:
        print('.........')
        print(index.text[5:])
        print(index.text[0:4])
        X_train, y_train, X_test, y_test, result_mean, pred_test, two_mouth_ago = prepare_data(index.text[0:4], seq_len=20, ratio=0.9)
        model, y_test, predicted = run_network(model,X_train, y_train, X_test, y_test, result_mean)
        count, range_rate, t_m_rate = pred_next(model,pred_test,two_mouth_ago)
        if count >= 5 and range_rate > 1.05 and t_m_rate > 1.1:
            print("can buy")
            final.append(index)
        print('.........')
    indus_list = find_industry(final)
    print('Total duration (s) : ', time.time() - global_start_time)
    user_ask(indus_list, model)
    
main()
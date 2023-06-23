import csv
import random
import numpy as np

def load_data(data_path):
    raw_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count <= 5:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                raw_data.append(row)
                line_count += 1
        print(f'Processed {line_count} lines.')
    return raw_data

def polish_data(raw_data):
    polished_data = []
    c=0
    for r in raw_data:
        newrow = []
        if r[0] == 'Male':
            newrow.append(1.0)
        else:
            newrow.append(2.0)

        newrow.append(float(r[1]))
        newrow.append(float(r[2]))
        newrow.append(float(r[3]))

        if r[4] == "No Info":
            newrow.append(1.0)
        elif r[4] == "never":
            newrow.append(2.0)
        elif r[4] == "former":
            newrow.append(3.0)
        elif r[4] == "current":
            newrow.append(4.0)
        elif r[4] == "not current":
            newrow.append(5.0)
        elif r[4] == "ever":
            newrow.append(6.0)
        #print(c, " ", r[4])
            
        
        newrow.append(float(r[5]))
        newrow.append(float(r[6]))
        newrow.append(float(r[7]))
        newrow.append(float(r[8]))
        polished_data.append(newrow)
    X = [[r[i] for i in range(len(r)-1)] for r in polished_data]
    y = [[r[len(r)-1]] for r in polished_data]
    return X,y

def balance_data(X,y):
    print("[DATASET BALANCING] starting...")
    minority = []
    count = 0
    for i in range(len(y)):
        if float(y[i][0]) >= 0.5:
            minority.append([X[i],y[i]])
            count += 1
    print(100 * count/len(y), "%")
    for i in range(int(1/(count/len(y)))):
        for m in minority:
            X.append(m[0])
            y.append(m[1])
    count = 0
    for i in range(len(y)):
        if float(y[i][0]) >= 0.5:
            count += 1

    print(100 * count/len(y), "%")
    print("[DATASET BALANCING] finished.")
    return X,y

X,y=polish_data(load_data("dataset/diabetes_prediction_dataset.csv"))
balance_data(X,y)

def split_train_test(X,y,test_size=0.15,validation=False, val_size=0):
    trainX = []
    testX = []
    valX=[]
    trainY = []
    testY = []
    valY = []
    indexes = [i for i in range(len(X))]
    random.shuffle(indexes)
    ccc=0
    for i in range(int(len(X)-(len(X)*test_size + len(X)*val_size))): 
        trainX.append(X[indexes[i]])
        trainY.append(y[indexes[i]])

    for i in range(int(len(X)*test_size)):
        testX.append(X[indexes[round(len(X)-(len(X)*test_size+len(X)*val_size))+i]])
        testY.append(y[indexes[round(len(X)-(len(X)*test_size+len(X)*val_size))+i]])

    for i in range(int(len(X)*val_size)):
        valX.append(X[indexes[round(len(X)-len(X)*val_size)+i]])
        valY.append(y[indexes[round(len(X)-len(X)*val_size)+i]])

    if validation:
        return (np.asarray(trainX, dtype=np.float32),
                np.asarray(testX, dtype=np.float32),
                np.asarray(valX, dtype=np.float32),
                np.asarray(trainY, dtype=np.float32),
                np.asarray(testY, dtype=np.float32),
                np.asarray(valY, dtype=np.float32))
    else:
        return (np.asarray(trainX, dtype=np.float32),
                np.asarray(testX, dtype=np.float32),
                np.asarray(trainY, dtype=np.float32),
                np.asarray(testY, dtype=np.float32))

# X,y=polish_data(load_data("dataset/diabetes_prediction_dataset.csv"))
# trainX, testX, valX, trainY, testY, valY = split_train_test(X,y,test_size=0.15,validation=True, val_size=0.10)

# # generate a 3-class classification problem with 1000 data points,
# # where each data point is a 4D feature vector
# print("[INFO] preparing data...")
# (X, y) = make_blobs(n_samples=1000, n_features=4, centers=3,
# 	cluster_std=2.5, random_state=95)



# # create training and testing splits, and convert them to PyTorch
# # tensors
# (trainX, testX, trainY, testY) = train_test_split(X, y,
# 	test_size=0.15, random_state=95)
# trainX = torch.from_numpy(trainX).float()
# testX = torch.from_numpy(testX).float()
# trainY = torch.from_numpy(trainY).float()
# testY = torch.from_numpy(testY).float()
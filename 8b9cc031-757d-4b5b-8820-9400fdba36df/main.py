import numpy as np
import pandas as pd
import random
import time
import os
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import optimizers
import tensorflow as tf

# Set seeds for reproducibility
SEED = 9
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Ensure the GPU is detected (if available)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=1024)]
        )
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found. Running on CPU.")

# Load data
SP500_df = pd.read_csv('data/SPXconst.csv')
all_companies = list(set(SP500_df.values.flatten()))
all_companies.remove(np.nan)

constituents = {'-'.join(col.split('/')[::-1]): set(SP500_df[col].dropna()) for col in SP500_df.columns}

constituents_train = {}
for test_year in range(1993, 2016):
    months = [f"{t}-{'0' if m < 10 else ''}{m}" for t in range(test_year-3, test_year) for m in range(1, 13)]
    constituents_train[test_year] = set(company for m in months for company in constituents.get(m, []))

def makeLSTM():
    inputs = Input(shape=(240, 1))  # Expecting 240 time steps with 1 feature
    x = LSTM(25, return_sequences=False)(inputs)
    x = Dropout(0.1)(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(), metrics=['accuracy'])
    model.summary()
    return model

def callbacks_req(model_type='LSTM'):
    csv_logger = CSVLogger(f"{model_folder}/training-log-{model_type}-{test_year}.csv")
    filepath = f"{model_folder}/model-{model_type}-{test_year}-E{{epoch:02d}}.keras"  # Updated file extension
    model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=False, save_freq='epoch')
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
    return [csv_logger, early_stopping, model_checkpoint]

def reshaper(arr):
    arr = np.array(np.split(arr, 3, axis=1))
    arr = np.swapaxes(arr, 0, 1)
    arr = np.swapaxes(arr, 1, 2)
    return arr

def trainer(train_data, test_data, model_type='LSTM'):
    np.random.shuffle(train_data)
    # Make sure to select exactly 240 columns for train_x
    train_x, train_y, train_ret = train_data[:, 2:242], train_data[:, -1], train_data[:, -2]

    # Calculate the number of features for reshaping
    num_features = train_x.shape[1]

    # Reshape with calculated features
    train_x = np.reshape(train_x, (len(train_x), num_features, 1)).astype(np.float32)
    train_y = np.reshape(train_y, (-1, 1)).astype(np.float32)
    train_ret = np.reshape(train_ret, (-1, 1)).astype(np.float32)
    
    # Encoding the labels
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(train_y)
    enc_y = enc.transform(train_y).toarray()
    train_ret = np.hstack((np.zeros((len(train_data), 1)), train_ret))

    if model_type == 'LSTM':
        model = makeLSTM()
    else:
        return

    callbacks = callbacks_req(model_type)
    
    model.fit(
        train_x,
        enc_y,
        epochs=1000,
        validation_split=0.2,
        callbacks=callbacks,
        batch_size=512
    )

    dates = list(set(test_data[:, 0]))
    predictions = {}
    for day in dates:
        test_d = test_data[test_data[:, 0] == day]
        test_d = np.reshape(test_d[:, 2:242], (len(test_d), num_features, 1)).astype(np.float32)
        predictions[day] = model.predict(test_d)[:, 1]
    return model, predictions

def trained(filename, train_data, test_data):
    model = load_model(filename)

    dates = list(set(test_data[:, 0]))
    predictions = {}
    for day in dates:
        test_d = test_data[test_data[:, 0] == day]
        test_d = np.reshape(test_d[:, 2:242], (len(test_d), test_d.shape[1] - 4, 1)).astype(np.float32)
        predictions[day] = model.predict(test_d)[:, 1]
    return model, predictions

def simulate(test_data, predictions):
    rets = pd.DataFrame([], columns=['Long', 'Short'])
    k = 10
    for day in sorted(predictions.keys()):
        preds = predictions[day]
        test_returns = test_data[test_data[:, 0] == day][:, -2]
        top_preds = predictions[day].argsort()[-k:][::-1]
        trans_long = test_returns[top_preds]
        worst_preds = predictions[day].argsort()[:k][::-1]
        trans_short = -test_returns[worst_preds]
        rets.loc[day] = [np.mean(trans_long), np.mean(trans_short)]
    print('Result : ', rets.mean())
    return rets

def create_label(df_open, df_close, perc=[0.5, 0.5]):
    if not np.all(df_close.iloc[:, 0] == df_open.iloc[:, 0]):
        print('Date Index issue')
        return
    perc = [0.] + list(np.cumsum(perc))
    label = (df_close.iloc[:, 1:] / df_open.iloc[:, 1:] - 1).apply(
        lambda x: pd.qcut(x.rank(method='first'), perc, labels=False), axis=1)
    return label[1:]

def create_stock_data(df_open, df_close, st, m=240):
    """
    Create stock data for a given stock `st`, including intraday returns and future intraday return.

    Args:
    df_open (DataFrame): DataFrame with opening prices.
    df_close (DataFrame): DataFrame with closing prices.
    st (str): Stock ticker symbol.
    m (int): Number of previous days to include for intraday returns.

    Returns:
    tuple: Tuple containing numpy arrays for training and testing data.
    """
    # Initialize a list to collect the data
    data = {
        'Date': df_close['Date'],
        'Name': [st] * len(df_close),
    }

    # Calculate daily change
    daily_change = df_close[st] / df_open[st] - 1

    # Collect shifted intraday return columns
    for k in range(m):
        data[f'IntraR{k}'] = daily_change.shift(k)

    # Add future return and labels
    data['IntraR-future'] = daily_change.shift(-1)
    data['label'] = list(label[st]) + [np.nan]
    data['Month'] = df_close['Date'].str[:-3]

    # Convert dictionary to DataFrame
    st_data = pd.DataFrame(data).dropna()

    # Split data into training and testing sets
    trade_year = st_data['Month'].str[:4]
    st_data = st_data.drop(columns=['Month'])
    st_train_data = st_data[trade_year < str(test_year)]
    st_test_data = st_data[trade_year == str(test_year)]

    # Convert to numpy array excluding non-numeric columns
    train_numeric = st_train_data.drop(columns=['Date', 'Name']).to_numpy().astype(np.float32)
    test_numeric = st_test_data.drop(columns=['Date', 'Name']).to_numpy().astype(np.float32)

    return train_numeric, test_numeric

def scalar_normalize(train_data, test_data):
    scaler = RobustScaler()
    scaler.fit(train_data[:, :-2])
    train_data[:, :-2] = scaler.transform(train_data[:, :-2])
    test_data[:, :-2] = scaler.transform(test_data[:, :-2])

# Set directories for models and results
model_folder = 'models-Intraday-240-1-LSTM'
result_folder = 'results-Intraday-240-1-LSTM'
for directory in [model_folder, result_folder]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Run training and testing for each year
for test_year in range(1993, 2020):
    print('-' * 40)
    print(test_year)
    print('-' * 40)
    
    # Load opening and closing prices
    filename_open = f'data/Open-{test_year-3}.csv'
    filename_close = f'data/Close-{test_year-3}.csv'
    
    df_open = pd.read_csv(filename_open)
    df_close = pd.read_csv(filename_close)
    
    # Create labels for training data
    label = create_label(df_open, df_close)
    
    # Filter stocks by available data
    stock_names = sorted(list(constituents[f'{test_year-1}-12']))
    
    train_data, test_data = [], []
    start = time.time()
    
    # Prepare training and testing data for each stock
    for st in stock_names:
        st_train_data, st_test_data = create_stock_data(df_open, df_close, st)
        train_data.append(st_train_data)
        test_data.append(st_test_data)
        
    # Concatenate data arrays for model input
    train_data = np.concatenate(train_data).astype(np.float32)
    test_data = np.concatenate(test_data).astype(np.float32)
    
    # Normalize the data
    scalar_normalize(train_data, test_data)
    
    print(f"Training data shape: {train_data.shape}, Testing data shape: {test_data.shape}, Time taken: {time.time() - start:.2f}s")
    
    # Train the model and make predictions
    model, predictions = trainer(train_data, test_data)
    
    # Simulate results and save returns
    returns = simulate(test_data, predictions)
    returns.to_csv(f'{result_folder}/avg_daily_rets-{test_year}.csv')
    
    # Generate statistics and save results
    # Assuming Statistics is a custom module, replace with actual statistics computation if needed
    # result = Statistics(returns.sum(axis=1))
    # print('\nAverage returns prior to transaction charges')
    # result.shortreport() 
    
    # Append results to a summary file
    with open(f'{result_folder}/avg_returns.txt', "a") as myfile:
        res = '-' * 30 + '\n'
        res += str(test_year) + '\n'
        # Replace result.mean() and result.sharpe() with actual mean and sharpe computation if needed
        res += 'Mean = ' + str(returns.mean().mean()) + '\n'
        # res += 'Sharpe = ' + str(result.sharpe()) + '\n'
        res += '-' * 30 + '\n'
        myfile.write(res)

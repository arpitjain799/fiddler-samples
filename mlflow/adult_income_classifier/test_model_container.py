import pandas as pd
import requests
import argparse

def test_endpoint(url, input_df, target_col):
    try:    
        headers = {'Content-Type': 'application/json; format=pandas-split'}
        if target_col:
            input_df.drop(target_col, axis=1, inplace=True)
        data = input_df.to_json(orient='split', index=False)
        print(data)
        res = requests.post(url, headers=headers, data=data)
    except Exception as e:
        error_info = res.json()
        raise RuntimeError(
            f'Model execution failed with exception message:\n {error_info}'
        )
    try:
        print(res.content)
        df = pd.read_json(res.content, orient='values')
        print('Model execution was successful')
        print(df.head())
        return df
    except Exception as e:
        raise RuntimeError(
            f'Model execution call was successful, but '
            f'failed to deserialize prediction. {e} {res.content[0:1000]}'
        )

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', help='Model endpoint URL', type=str)
    parser.add_argument('--csv', help='CSV file with test data', type=str)
    parser.add_argument('--target', help='Name of the target column in the CSV (will be dropped)', type=str, default=None)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = get_args()
    print(args.target)
    df = pd.read_csv(args.csv)
    test_endpoint(args.url, df.sample(100), args.target)


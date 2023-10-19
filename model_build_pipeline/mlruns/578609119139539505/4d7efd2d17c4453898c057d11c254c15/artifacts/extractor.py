import pandas as pd

def feature_filter(df):
    df_f = df[['checking_status',
              'age',
              'installment_commitment',
              'credit_history',
              'credit_amount',
              'other_parties',
              'other_payment_plans',
              'purpose',
              'employment',
              'savings_status',
              'property_magnitude',
              'personal_status',
              'class']].copy() # add the target variable
    return df_f

def change_data(df):

    df_c = df.copy()
    # purpose
    df_c['purpose'].replace(
    {
        'buy_radio_tv': 'low-risk',
        'education': 'normal',
        'buy_furniture_equipment': 'low-risk',
        'buy_new_car': 'low-risk',
        'buy_used_car': 'low-risk',
        'business': 'normal',
        'buy_domestic_appliance': 'normal',
        'repairs': 'normal',
        'other': 'normal',
        'retraining': 'normal'
    }, inplace=True)

    # credit_history
    df_c['credit_history'].replace(
    {
        '\'critical/other existing credit\'': 'good',
        '\'existing paid\'': 'good',
        '\'delayed previously\'': 'good',
        '\'no credits/all paid\'': 'normal',
        '\'all paid\'': 'normal'
    }, inplace=True)

     # credit_history
    df_c['class'].replace(
    {
        'good':0,
        'bad':1
    }, inplace=True)

    return df_c

def ohc_data(df):
    data_fs = df.copy()
    cats = ['credit_history',
            'other_parties',
            'other_payment_plans',
            'purpose',
            'employment',
            'savings_status',
            'checking_status',
            'property_magnitude',
            'personal_status']
    
    # onehot encode
    def onehotencode(df, feat):
        new_df = df.copy()
        ohc = pd.get_dummies(df[feat], prefix=feat)
        new_df.drop(columns=feat, inplace=True)
        return pd.concat([new_df, ohc], axis=1)

    for v in cats:
        data_fs = onehotencode(data_fs, v)

    return data_fs

def extract_data(df):
    extract = ["checking_status_'no checking'",
                'age',
                'credit_history_normal',
                'other_parties_guarantor',
                "checking_status_'>=200'",
                "savings_status_'<100'",
                'installment_commitment',
                'credit_amount',
                "personal_status_'male single'",
                'other_payment_plans_none',
                'other_parties_none',
                'purpose_normal',
                "employment_'4<=X<7'",
                "savings_status_'>=1000'",
                "savings_status_'no known savings'",
                "checking_status_'<0'",
                "employment_'<1'",
                'employment_unemployed',
                "property_magnitude_'no known property'",
                "class"]
    
    df_ext = df[extract].copy()

    return df_ext

def fe_pipeline(df):
    """
    Pipeline for extracting features of a pandas dataframe
    """
    df_fe = df.copy()
    df_fe = feature_filter(df_fe)
    df_fe = change_data(df_fe)
    df_fe = ohc_data(df_fe)
    df_fe = extract_data(df_fe)

    return df_fe

# df = pd.read_csv('test.csv')
# print(df.columns)
# df_f = fe_pipeline(df)
# print(df_f.columns)
# print(df_f.shape)
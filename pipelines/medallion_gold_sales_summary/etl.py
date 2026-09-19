def process(df):
    return df.filter(df['status'] == 'COMPLETED').dropDuplicates(['transaction_id'])

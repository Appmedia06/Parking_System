import pandas as pd


# new a dataframe
parking_df = pd.DataFrame(columns=['Car_Id','Number_Plate','Entry_Year', 'Entry_Month', 'Entry_Day','Entry_Hour','Entry_Minute'
                           ,'Entry_Second','Leave_Year','Leave_Month','Leave_Day', 'Leave_Hour', 'Leave_Minute','Leave_Second'
                           , 'Amounts Payable','is_Paid'])
# save
parking_df.to_csv('parking_df',index=False)


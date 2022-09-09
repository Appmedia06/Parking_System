import parking_sys as ps
import pandas as pd


parking_df = pd.read_csv('parking_df')

possible_Id = ps.payment_machine1(parking_df)

final_Id = ps.payment_machine2(possible_Id)

parking_df = ps.check_out(parking_df,final_Id)

parking_df.to_csv('parking_df', index=False)


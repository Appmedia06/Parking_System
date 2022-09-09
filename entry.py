import parking_sys as ps
import pandas as pd


parking_df = pd.read_csv('parking_df')

car_index = len(parking_df) + 1

car = ps.Car_Object(car_index)

ps.entry_cam(car_index)

car.process_image()

car_number_plate, parking_df = car.identify_number_plate(parking_df)

print(car_number_plate)

ps.open_gate()

parking_df.to_csv('parking_df', index=False)
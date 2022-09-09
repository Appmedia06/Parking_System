import parking_sys as ps
import pandas as pd


parking_df = pd.read_csv('parking_df')

with open('leave_index.txt', 'r') as f:
    leave_index = int(f.read())
f.close()


leave_car = ps.Leave_sys(leave_index)

leave_car.leave_cam()

leave_car.process_image()

leave_number_plate = leave_car.identify_number_plate()

print(leave_number_plate)

leave_car.check_database(parking_df)


with open('leave_index.txt', 'w') as f:
    f.write(str(leave_index + 1))
f.close()
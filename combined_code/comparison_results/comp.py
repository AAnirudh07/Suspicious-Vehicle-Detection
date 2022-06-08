# import pandas as pd

# input_time = 30317
# output_time = 32449

# df = pd.read_csv("event.csv",names=["v_id","f_id","vehicle","empty"])
# df['time'] = input_time + df['f_id']//25
# df.to_csv('input.csv')

import pandas as pd
df1 = pd.read_csv("input.csv",names=["v_id","f_id","vehicle","empty","time"])

df2 = pd.read_csv("output.csv",names=["v_id","f_id","vehicle","empty","time"])
df2["marked"] = "not found"
df1["marked"] = "-1"
for index1, row1 in df1.iterrows():
#   print(index1)
  for index2, row2 in df2.iterrows():
      if(row2["marked"] !="found" and row1["vehicle"] == row2["vehicle"] and row2["time"]>row1["time"]):
          row1["marked"] = str(int(row2["time"])-int(row1["time"]))
          row2["marked"] ="found"
        #   print(row1)
        #   print(row2)
          break
df1.to_csv('video1.csv')
df2.to_csv('video2.csv')

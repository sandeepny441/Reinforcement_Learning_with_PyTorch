import streamlit as st
from datetime import datetime, timedelta

# Example data fetch function
def fetch_amplitude_data_mock(start_date, end_date):
    # Replace this with the actual fetch function
    return [
        {"event": "login", "count": 120},
        {"event": "purchase", "count": 45},
        {"event": "logout", "count": 20}
    ]

# Streamlit Dashboard
st.title("Amplitude Dashboard")

# Input for date range
start_date = st.date_input("Start Date", datetime.now() - timedelta(days=7))
end_date = st.date_input("End Date", datetime.now())

if st.button("Fetch Data"):
    data = fetch_amplitude_data_mock(start_date, end_date)
    if data:
        st.subheader("Aggregated Data")
        for record in data:
            st.write(f"Event: {record['event']}, Count: {record['count']}")
    else:
        st.error("Failed to fetch data from Amplitude.")

import pandas as pd
import numpy as npimport pandas as pd
import numpy as npimport pandas as pd
import numpy as npimport pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



player_win=tk.Label(gui,text='')
for i in ('row1','row2','row3'):
    for j in range(3):
        
        vars()[i+str(j+1)]=tk.Button(vars()[i], text=f'       ',bd='1',command=partial(clicked,i+' '+str(j+1)))
        vars()[i+str(j+1)].pack(side='left')


gui.mainloop()
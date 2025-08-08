import sqlite3
import pandas as pd

conn = sqlite3.connect(r"J:\Gold_FX\test_trading.db")

df = pd.read_sql("SELECT * FROM signals", conn)
print(df.head())

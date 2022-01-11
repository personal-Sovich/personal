# pip install mysql-connector (command prompt)
# pip install mysql-connector-python
# WINDOWS > SERVICES > MYSQL80 > START THE SERVICE
# Connect server to python (host='localhost', user='root', passwd='password')
import mysql.connector # make a connection
import pandas as pd

# Connecting to database
mydb = mysql.connector.connect(
    host='localhost',
    user='root', 
    passwd='password',
    auth_plugin='mysql_native_password',
    database='sakila')
mycursor = mydb.cursor()

# Show available databases
mycursor.execute("SHOW DATABASES") # can also log in to mySQL workbench to view databases
for i in mycursor:
    print(i)

# Pulling data from database
mycursor.execute("""
    SELECT first_name, last_name 
    FROM actor 
    WHERE first_name = 'James'
    LIMIT 10
    """)
pandas_df = pd.DataFrame(mycursor.fetchall())

# Creating a new datababse
database = "main"
mycursor.execute(f"CREATE DATABASE {database}")

# Dropping a database
mycursor.execute(f"DROP DATABASE {database}")

# Close connections
mycursor.close()
mydb.close()

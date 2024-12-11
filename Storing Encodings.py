import sqlite3

#connect ot database
conn = sqlite3.connect('Face Encodings.sql')

#close connection
conn.close()
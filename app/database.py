import base64
import sqlite3
import os
import time

db_name = './defected_images.db'

if not os.path.exists(db_name):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''CREATE TABLE defected_images
                 (id integer primary key, image_name text, image_data LONGBLOB,
                 datetime text)''')
    conn.commit()
    conn.close()


def insert_image(image_path):
    id = int(time.time())
    image_name = image_path
    datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read())

    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    c.execute("INSERT INTO defected_images VALUES (?, ?, ?, ?)",
              (id, image_name, image_data, datetime))

    conn.commit()
    conn.close()


def get_all_images():
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    c.execute("SELECT * FROM defected_images")
    rows = c.fetchall()

    conn.close()
    return [(row[0], row[1], base64.b64decode(row[2]), row[3]) for row in rows]

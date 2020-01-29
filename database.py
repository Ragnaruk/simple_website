import sqlite3
import datetime
import requests
import json
import time
from pathlib import Path


def prepare_db_file():
    with open(Path(__file__).parent / "data" / "price.csv", "r") as f:
        price = f.read().split('\n')

    def generate_price(list):
        for el in list:
            try:
                yield [el.split(",")[0], el.split(",")[4]]
            except Exception:
                pass

    def generate_normalize(list):
        for el in list:
            date, time = el[0].split(" ")
            date = date.split(".")
            time = time.split(":")
            dt = datetime.datetime(
                int(date[2]),
                int(date[1]),
                int(date[0]),
                int(time[0]),
                int(time[1])
            )

            yield [str(dt), float(el[1])]

    # # Accessing database
    conn = sqlite3.connect("mydatabase.db")
    cursor = conn.cursor()

    # # Inserting values
    cursor.execute("CREATE TABLE bitcoin (btc_date text, btc_price real)")
    conn.commit()

    for entry in generate_normalize(generate_price(price)):
        print(entry)
        cursor.execute("INSERT INTO bitcoin VALUES (?,?)", entry)
    conn.commit()


def prepare_db_api():
    """
    https://www.coingecko.com/ru/api#
    """
    conn = sqlite3.connect("mydatabase.db")
    cursor = conn.cursor()

    for coin in ["bitcoin", "ethereum", "litecoin", "eos", "ripple"]:
        response = requests.get(
            "https://api.coingecko.com/api/v3/coins/{0}/market_chart?vs_currency=usd&days=30"
                .format(coin)
        )

        prices = json.loads(response.text)["prices"]

        # For db creation
        cursor.execute(f"CREATE TABLE {coin} (btc_date text, btc_price real)")
        conn.commit()

        for price in prices:
            cursor.execute(f"INSERT INTO {coin} VALUES ({price[0]},{price[1]})")
        conn.commit()

        print("Finished getting {0} prices for the last 30d.".format(coin))
        print("Currect time: {}".format(int(time.time())))


def get_price(coin, date_from, date_until):
    conn = sqlite3.connect("mydatabase.db")
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM \"{0}\""
        " WHERE btc_date >= \"{1}\""
        " AND btc_date <= \"{2}\"".format(coin, date_from, date_until)
    )

    return cursor.fetchall()


if __name__ == '__main__':
    # get_price("bitcoin", "2019-11-10 00:00:00", "2019-11-30 00:00:00")
    prepare_db_api()

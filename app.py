"""
https://coder-coder.com/build-flexbox-website-layout/
"""
from flask import Flask, render_template, url_for
from pathlib import Path

app = Flask(__name__)


@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/graph')
def graph():
    with open(Path(__file__).parent / "data" / "price.csv", "r") as f:
        price = f.read()

    price_labels = []
    price_data = []
    for pr in price.split('\n'):
        try:
            d = pr.split(',')

            price_labels.append(
                d[0]
            )

            price_data.append(
                d[4]
            )
        except Exception:
            pass

    return render_template(
        'graph.html',
        price_labels=price_labels,
        price_data=price_data
    )


@app.route('/news')
def news():
    return render_template('news.html')


@app.route('/stats')
def stats():
    return render_template('stats.html')


if __name__ == '__main__':
    app.run(debug=True)

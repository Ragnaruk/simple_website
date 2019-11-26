"""
https://coder-coder.com/build-flexbox-website-layout/
"""
from flask import Flask, render_template, url_for

app = Flask(__name__)


@app.route('/')
def landing():
    return render_template('landing.html')


@app.route('/graph')
def graph():
    return render_template('graph.html')


@app.route('/news')
def news():
    return render_template('news.html')


@app.route('/stats')
def stats():
    return render_template('stats.html')


if __name__ == '__main__':
    app.run(debug=True)

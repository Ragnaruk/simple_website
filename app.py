"""
https://coder-coder.com/build-flexbox-website-layout/
"""
from flask import Flask, render_template, url_for
import json

from graph import get_graph_data

app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/blog')
def blog():
    return render_template('blog.html')


@app.route('/blog-post1')
def blog_post1():
    return render_template('blog-post1.html')


@app.route('/blog-post2')
def blog_post2():
    return render_template('blog-post2.html')


@app.route('/blog-post3')
def blog_post3():
    return render_template('blog-post3.html')


@app.route('/blog-post4')
def blog_post4():
    return render_template('blog-post4.html')


@app.route('/graph')
def graph():
    labels, data, prediction_data = get_graph_data()

    return render_template(
        'graph.html',
        price_labels=labels,
        price_data=data,
        prediction_data=prediction_data
    )


@app.route('/api/graph')
def graph_update():
    return json.dumps(get_graph_data())


if __name__ == '__main__':
    app.run(debug=True, threaded=False)

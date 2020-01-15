"""
https://coder-coder.com/build-flexbox-website-layout/
"""
from flask import Flask, render_template, url_for
from database import get_price
from datetime import datetime

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
    # 1578996628988 to 1579083004000
    price = get_price("bitcoin", "1579007000000", "1579083004000")
    labels = [datetime.utcfromtimestamp(int(p[0]) / 1000) for p in price]
    data = [p[1] for p in price]

    print(labels)
    print(data)

    return render_template(
        'graph.html',
        price_labels=labels,
        price_data=data
    )


if __name__ == '__main__':
    app.run(debug=True)

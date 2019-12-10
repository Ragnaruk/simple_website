"""
https://coder-coder.com/build-flexbox-website-layout/
"""
from flask import Flask, render_template, url_for
from pathlib import Path

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


if __name__ == '__main__':
    app.run(debug=True)

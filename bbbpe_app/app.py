from flask import Flask, render_template
app = Flask(__name__)

# two decorators, same function
@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html', the_title='BBBPE Home')

@app.route('/myth.html')
def myth():
    return render_template('myth.html', the_title='BBBPE Database')

if __name__ == '__main__':
    app.run(debug=True)

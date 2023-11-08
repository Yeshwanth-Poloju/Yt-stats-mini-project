from flask import Flask, render_template, request
from transcript_summary import get_video_summary
from comments_classifier import get_classified_comments

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        video_url = request.form['url']
        if '=' in video_url:
            video_id = video_url.split('=')[1]
        else:
            video_id = video_url[-11:]
        video_summary = get_video_summary(video_id)
        get_classified_comments(video_id)
    return render_template("analytics.html",summary = video_summary)

if __name__ == '__main__':
    app.run(debug=True)
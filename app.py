from flask import Flask , url_for, render_template, request
from flask_bootstrap import Bootstrap 

#NLP webpackages
from spacy_summarization import text_summarizer
from nltk_summarization import nltk_summarizer
from gensim.summarization import summarize 
#other packages
import spacy
nlp = spacy.load('en')
import time

#Web Scraping Packages
from bs4 import BeautifulSoup
from urllib import urlopen
# from urllib.request import urlopen

#sumy packages
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


#sentiment analysis
from textblob import TextBlob,Word 
import random 


app = Flask(__name__)

Bootstrap(app)


# Sumy 
def sumy_summary(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result






def readingTime(mytext):
	# total_words = len([token.text for token in nlp(mytext)])
	
	total_words = len([ token.text for token in nlp(mytext)])
	estimated_read_time = total_words/200.0
	#200 -- average reading time
	return estimated_read_time

#fetch data from url
def get_text(url):
	page = urlopen(url)
	soup = BeautifulSoup(page,"lxml")
	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))        #all the paragraph text , and then joining them
	
	return fetched_text




#home
@app.route('/')
def index():
	return render_template('index.html')




@app.route('/compare_summary')
def compare_summary():
	return render_template('compare_summary.html')



@app.route('/analyze', methods=['GET','POST'])
def analyze():

	start = time.time()

	if request.method == 'POST':
		rawtext = request.form['rawtext']

		final_read_time  = readingTime(rawtext)
		#now we need to do

		#Summarization
		final_summary = text_summarizer(rawtext)

		#Reading Time

		summary_read_time = readingTime(final_summary)

		end = time.time()

		final_time = end -start

	return render_template('index.html', ctext=rawtext,final_summary=final_summary, final_time=final_time, final_read_time=final_read_time,summary_read_time=summary_read_time)


@app.route('/analyze_url', methods=['GET','POST'])
def analyze_url():
	start = time.time()

	if request.method == 'POST':

		raw_url = request.form['raw_url']

		rawtext = get_text(raw_url)

		final_read_time  = readingTime(rawtext)
		#now we need to do

		#Summarization
		final_summary = text_summarizer(rawtext)

		#Reading Time

		summary_read_time = readingTime(final_summary)

		end = time.time()

		final_time = end -start

	return render_template('index.html',ctext=rawtext, final_summary=final_summary, final_time=final_time, final_read_time=final_read_time,summary_read_time=summary_read_time)





@app.route('/comparer', methods=['GET','POST'])
def comparer():

	
	start = time.time()

	if request.method == 'POST':

		rawtext = request.form['rawtext']
		final_reading_time = readingTime(rawtext)
		final_summary_spacy = text_summarizer(rawtext)
		summary_reading_time = readingTime(final_summary_spacy)
		# Gensim Summarizer
		final_summary_gensim = summarize(rawtext)
		summary_reading_time_gensim = readingTime(final_summary_gensim)
		# NLTK
		final_summary_nltk = nltk_summarizer(rawtext)
		summary_reading_time_nltk = readingTime(final_summary_nltk)
		# Sumy
		final_summary_sumy = sumy_summary(rawtext)
		summary_reading_time_sumy = readingTime(final_summary_sumy) 

		end = time.time()
		final_time = end-start

	return render_template('compare_summary.html',ctext=rawtext,final_summary_spacy=final_summary_spacy,final_summary_gensim=final_summary_gensim,final_summary_nltk=final_summary_nltk,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time,summary_reading_time_gensim=summary_reading_time_gensim,final_summary_sumy=final_summary_sumy,summary_reading_time_sumy=summary_reading_time_sumy,summary_reading_time_nltk=summary_reading_time_nltk)

@app.route('/sentiment',methods=['GET','POST'])
def sentiment():
	start = time.time()


	if request.method == 'POST':
		rawtext = request.form['rawtext']

		#NLP
		blob = TextBlob(rawtext)

		recieved_text = blob

		blob_sentiment, blob_subjectivity = blob.sentiment.polarity, blob.sentiment.subjectivity
		num_of_tokens = len(list(blob.words))


		# Extracting Main Points
		nouns = list()
		summary = list()
		final_time=0
		for word, tag in blob.tags:

		    if tag == 'NN':

		        nouns.append(word.lemmatize())
		        len_of_words = len(nouns)
		        rand_words = random.sample(nouns,len(nouns))
		        final_word = list()
		        for item in rand_words:
		        	word = Word(item).pluralize()
		        	final_word.append(word)
		        	summary = final_word
		        	end = time.time()
		        	final_time = end-start


	return render_template('sentiment.html',recieved_text=recieved_text, num_of_tokens=num_of_tokens,blob_sentiment=blob_sentiment, blob_subjectivity=blob_subjectivity,summary=summary,final_time=final_time)


@app.route('/sentiment_index')
def sentiment_index():
	return render_template('sentiment.html')




if __name__ == '__main__':
	app.run(debug=True)
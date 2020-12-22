# Sentiment Analysis with Count-based Methods, LSTM, and Transformer-based Method


**Author** : Mucong (Collin) Li

**Time Completed**: December 2020 

## Technical
**Method**:        
- Scikit-Learn: CountVectorizer; TfidfVectorizer
- NLTK (Natural Language Toolkit)
- Keras: LSTM
- TensorFlow
- Pipelines from the Hugginface Library for Transformer-based methods

**Language**: Python (notebook)   

**Data**: 
- Twitter_Data: The File Contains 163k tweets along with its Sentimental Labelling      
- Reddit_Data: The File Contains 37k comments along with its Sentimental Labelling
- All the Comments in the dataset are cleaned and assigned with a Sentimet L Label Using Textblob            

## Takeaway

### Approach 1: Count-based Methods
#### CountVectorizer vs. TFIDFVectorizer:
**Difference between the two models? Which one is better and why?**
- The CountVectorizer converts the text documents to a matrix of token counts, which record the number of time that certain word appears
- The TfidfVectorizer converts the text documents to a matrix of TF-IDF features, which record the number of time that certain word appears, but normalized by total times that it appears in the document, and this makes word such as ‘the’, ‘a’, ‘and’ less important for the model
- For Reddit, TFIDF is better
- For Twitter, Count is better
- This is actually NOT what I expected as I thought TFIDF would be better in all scenarios. 
- After thinking about it, I think this happens because:
- On Twitter, many people post similar contents on the same topic. If a certain topic has a strong emotion and sentiment associated with it, more people are going to discuss it. Thus, the count of appearance of words can be a strong indicator or the sentiment.
- On the other hand, Reddit is a forum. People are more likely to discuss certain issue under one post/thread. If someone has said something, other people are not very likely to repeat similar thing. TFIDF can effectively eliminate the common words and emphasize the valuable words, and it is more effective here.

### Approach 2: LSTM
#### To improve modeling result:
- dropout
- batch normalization
- changing the activation functions.
- More complex:
    - different optmizers 
    - learning rate schedulers 
- Embedding size and batch size


#### Embedding Size:
- The first thing that I notice when the embedding size is increased was that the model ran slower.
- Then the result metrics (on test data) get better result. 
- The embedding size is the output dimension of the Keras Embedding Layer, which is the size of the vector space in which words will be embedded. It defines the size of the output vectors from this layer for each word.
- The embedding dimensions will determine how much you will compress / intentionally bottleneck the lexical information; larger dimensionality will allow your model to distinguish more lexical detail which is good if and only if your supervised data has enough information to use that lexical detail properly, but if it's not there, then the extra lexical information will overfit and a smaller embedding dimensionality will generalize better.

### Approach 3: Transformer-based Method
#### Fine-tune the transformer model
Steps taken:
- Specify the optimizer to Adam and learning rate to 5e-5
- Specify the batch size to 16
- Specify the epochs to 2

Result:
- The fine-tuned method perform much better with an accuracy of 0.95

## Further:   
#### Which situations to use, the count-based, the LSTM, the out-of-the-box, and the fine-tuned model?
- **Count-based**: for situation where the count of appearance of words can be a strong indicator of sentiment. For example, on Twitter, many people post similar contents on the same topic. If a certain topic has a strong emotion and sentiment associated with it, more people are going to discuss it. Thus, the count of appearance of words can be a strong indicator or the sentiment.
- **LSTM**: the typical use cases for LSTM are music generation, text generation, and stock prediction
- **Out-of-the-box**: when the application of the analysis is similar to the one that is used to train the pre-trained model
- **Fine-tuned**: when the out-of-the-box model gets a decent baseline score, and the analytical situation is similar so that fine-tuning would result in a great increase of performance





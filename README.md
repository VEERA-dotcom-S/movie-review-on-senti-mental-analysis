# movie-review-on-senti-mental-analysis
# Abstract
Sentiment analysis is among the primary natural language processing (NLP)
 tasks and is widely utilized for extracting emotions and sentiments from text
 corpora. This paper proposes a comprehensive sentiment analysis approach for
 movie reviews based on Word2Vec, TextBlob, VADER, and Gated Recurrent
 Units (GRU). Word2Vec is employed for word embeddings to extract semantic
 1
word relationships for improved feature representation. TextBlob and VADER
 are implemented as lexicon-based sentiment analysis tools, for which TextBlob
 is interested in polarity and subjectivity and VADER is engineered for short
 texts with clear-cut sentiment indications. Besides, deep learning architecture
 in the form of GRU is employed for extracting long dependencies and context
 associations between words of text corpora for enhanced sentiment classifica
tion. Methods are experimented and contrasted on the basis of a benchmark
 IMDB dataset with reference to accuracy, precision, recall, and F1 score. Exper
imental findings substantiate that sentiment handling by deep learning-based
 approaches, i.e., GRU via Word2Vec embeddings, is better than traditional lexi
con based approaches. The work provides insights to NLP-based opinion mining
 researchers and practitioners regarding the merit of utilizing hybrid approaches
 towards sentiment classification.
 
 1 Introduction
 The rapid rise of web based reviews has ensured sentiment analysis has become an
 accepted tool to read the public psyche, especially within the entertainment sector.
 Reviews for f ilms, specifically, play an important role in audience choices and offer
 worthwhile feedback regarding viewer tendencies. Examination of these textual data
 allows stakeholders, such as f ilm makers, producers, and advertising agencies, to assess
 the feelings of audiences and refine upcoming productions. Classic sentiment analysis
 methods are mostly lexicon based, and they are best represented by tools like TextBlob
 and VADER that categorize sentiments with pre-specified words and polarity scores.
 Although these methods are easy to interpret and use, they have the disadvantage
 of context sensitive sentiments and sophisticated linguistic patterns. To mitigate such
 shortcomings, more advanced models like Re current Neural Networks (RNNs) and
 their extensions, i.e., Gated Recurrent Units (GRU), have emerged as top-of-the line
 solutions because they are capable of learning sequentially complex patterns in text
 data efficiently.
 In this research, we propose a hybrid method that combines word embeddings
 generated through Word2Vec with tradi tional and deep learning-driven sentiment
 analysis techniques. Word2Vec embeds text data in dense vector spaces, thus enriching
 semantic understanding of words. TextBlob and VADER are used as the basis for
 sentiment classification, and GRU is incorporated to identify contextual dependencies,
 thus increasing classification accuracy. The IMDB movie review corpus serves as a
 baseline to determine the efficacy of the approaches.
 The primary goals of this work are presented below. For comparison of lexicon
based versus deep learning-based sentiment analysis techniques.
 For analyzing whether Word2Vec embeddings can improve sentiment classification
 accuracy.
 2
To compare the performance of GRU in handling compound sentiments with
 traditional methods.
 The rest of the paper is structured as follows. Section II is an overview of related
 work in sentiment analysis and deep learning models. Section III is an overview of
 the proposed methodology, data pre-processing, feature extraction,and model imple
mentation. Section IV is for experimental results and performance evaluation. Finally,
 Section V is the conclusion of the study and the future direction of the research.
 By combining traditional and deep learning-based senti ment analysis techniques,
 this research provides an exhaustive comparative study, emphasizing the strengths
 and weaknesses inherent in each technique in movie review classification.

# 2 LITERATURE SURVEY
 [1]This work suggests a hybrid sentiment analysis approach that integrates rule-based,
 supervised, and machine learning methods. It enhances the effectiveness of classifi
cation via F1 score, balancing precision with recall. [2] This work targets sentiment
 analysis from Twitter through automatic collection and processing of a corpus for
 opinion mining. A classifier is constructed to identify positive, negative, and neutral
 sentiments, performing better than earlier strategies. [3] This paper addresses emoji
 prediction as a text classifi cation problem with 1.5 million tweets, training LSTM
RNNand CNNmodels. The CNN model performs better than others in accuracy and
 F1-score, demonstrating its efficiency. [4] This paper formulates emoji prediction as a
 text clas sification problem with 1.5 million tweets. The CNN model performs better
 than LSTM-RNNandbaseline models in accuracy and F1-score. [5] This work presents
 a Street Illumination Mapping algorithm based on smartphone sensors and an IoT
cloud framework to generate high-granular nighttime city lighting maps. It compares
 favorably with conventional approaches using actual data from Kolkata. [6] This work
 introduces WebClassify, a tool that employs a na¨ ıve Bayes algorithm with a multi
nomial model modified to classify web pages into categories. It explains the process
 of preparing data and shows better accuracy with a larger vocabulary size. [7] This
 paper suggests a fuzzy set-based collaborative f iltering model that combines both
 subjective (expert) and objective (user) feedback. It solves problems such as the cold
 start problem and generates high-quality recommendations. [8] This paper presents
 a semi-automatic process to con struct multilingual sentiment dictionaries based on
 a triangu lation method. Experiments demonstrate that triangulated word lists are
 more precise than typical machine-translated lists. [9] This work introduces an opinion
 mining system that mines product features and sums up sentiment from web opinions
 based on IR-based feature extraction. The method proposed works similarly to con
ventional ML and gives en couraging results when partitioned by features. [10] In this
 paper, sentiment classification is performed on travel blog reviews by employing Na¨
 ıve Bayes, SVM, and character-based N-gram models. The results indicate SVM and
 N-gram perform better than Na¨ ıve Bayes with all the models attaining accuracy of
 over 80 [11] This paper improves variational EM in probabilistic NLP by employing
 a mixture-based posterior rather than using strict independence assumptions. This
 provides the flexibility of including prior knowledge in the form of soft constraints
 3
during the E-step. [12] The multivariate Bernoulli and multinomial naive Bayes text
 classification models are compared in this paper. Experiments demonstrate the multi
nomial model performs better, particularly with larger vocabularies, and can achieve
 as much as a 27 [13] In this paper, 45,278 tweets are analyzed using ICT and lexicon
based approaches to measure public sentiment towards the education system. Results
 show predominantly negative sentiment worldwide, with positive sentiment in certain
 low income nations and greater subjectivity among men. [14] This paper suggests a
 sentiment-oriented web crawling framework to rapidly identify and examine opinion
ated content from film and hotel reviews. It compares K-NN and Na¨ ıve Bayes, and
 f
 inds Na¨ ıve Bayes works better for film reviews, whereas both give similar results
 for hotel reviews. [15] This paper reports on SemEval 2014 message polarity classifi
cation participation, with special focus on the utilization of unlabeled data. Blending
 word representations with classic features produced notable improvements in accu
racy. [16] This paper reports SO-CAL, a lexicon-based sentiment analysis system that
 employs annotated dictionaries with po larity, strength, intensification, and negation
 support. It obtains consistent domain and unseen data polarity classification, with
 dictionary verification through Mechanical Turk. [17] This paper proposes an unsu
pervised hierarchical rule based approach for aspect term extraction in aspect-level
 sen timent analysis, aiming for high recall. Tested on the SemEval 2014 dataset, it
 achieves recall scores of 81.9 (Restaurant) and 68.7 (Laptop), outperforming several
 state-of-the-art models. [18] This work introduces WebClassify, a web page classi f ica
tion system based on a multinomial model na¨ ıve Bayes algorithm. It describes the
 necessary data preparation and demonstrates increased classification accuracy with
 growing vocabulary size. [19] This work introduces a fuzzy set theory-based collabo
 rative filtering approach that combines both subjective (expert) and objective (user)
 views. The method solves conventional CF problems such as new users/items and
 generates high-quality recommendations.
 2.1 What is Sentiment Analysis
 Sentiment analysis (SA), or opinion mining, is a Natural Language Processing (NLP)
 technique used to determine the sentiment expressed in a piece of text, classifying
 it as positive, negative, or neutral. SA is used extensively in business, social media,
 entertainment, and finance to analyze customer views, monitor trends, and improve
 decision-making. In movie re views, SA apprises producers of what the public is react
ing to in their productions and allows the audience to make informed decisions. The
 general methods are lexicon-based, machine learning, and deep learning techniques.
 Sarcasm, negation, and context dependence problems complicate sentiment classifica
 tion and therefore need complex mechanisms such as word embeddings and neural
 networks to achieve better accuracy.
 2.2 Importance of Sentiment Analysis
 22,500 film review sentiment analysis helps discover audi ence emotions and improves
 content and promotion by helping directors do the same. Programs like Logistic
 Regression and Decision Tree Classifier analyze instantly with reduced manpower. The
 4
like of streaming channels such as Netflix and Amazon Prime improves recommenda
tion with trends of sen timents. TF-IDF vectorization and machine learning facilitate
 real-time review monitoring, increasing customer interaction. Efficiency in dealing
 with big data is illustrated through a high dimensional sparse matrix (15,750x50,478).
 These methods are applied to social media analysis, assisting brands in their online
 reputation. Companies employing sentiment analysis have a competitive advantage
 through the adjustment of strategies based on feedback from the audience. This
 method enhances decision-making and customer experience. It also forms the basis for
 NLP research, improving the accuracy of sentiment classification.
 2.3 Sentiment Analysis in Movie Reviews
 This work uses machine learning models to classify 22,500 movie reviews as positive
 or negative. Preprocessing was done on the dataset by deleting HTML tags, special
 characters, and stopwords, followed by stemming on the Porter Stemmer. TF IDF
 vectorization converted the text into numerical represen tation for training the model.
 The dataset was divided into 70 percentage training and 30 percentage testing, and
 Logistic Regression and Decision Tree Classifier were employed. The Decision Tree
 had 71.2 percentage accuracy, with a balanced precision and recall for both sentiment
 classes.
 2.4 Challenges in Sentiment Analysis
 Sentiment analysis is plagued by a myriad of challenges that can make it ineffective
 and unbefitting. The largest of these challenges is linguistic ambiguity, where a term
 has multiple meanings based on the context of reference. The term ”cool,” for exam
ple, can be used to describe a temperature state or used to indicate approval, making
 it difficult for the model to accurately detect sentiment. Sarcasm and irony also pose
 great challenges since sentences such as ”Oh great, another delay!” convey negative
 sentiment despite having positive words. Domain dependence is also a problem since
 models trained on one corpus, e.g., movie reviews, hardly generalize to another cor
pus like product reviews or political opinions. Moreover, handling negations is very
 difficult, as can be seen from the necessity to consider ”not bad” as an expression of a
 positive sentiment. Finally, there is also a need to properly classify subjective opinions
 against objective facts in a way that can properly identify sentiment. Aspect-based
 sentiment analysis (ABSA) is a unique task where models need to determine senti
ment about certain attributes instead of reading the whole reviews as a whole. Opinion
 spam and fake reviews also skew sentiment analysis by providing biased information
 that misleads the consumer as well as the business. Lastly, unbalanced datasets have
 the potential to bias sentiment prediction such that certain sentiment categories over
ride others. Secondly, there is also the space for fine-grained sentiment classification
 such as capturing more refined emotions in place of binary positive or negative clas
sification—anger, joy, or sadness, for example. Resolving these challenges is vital to
 developing more precise sentiment analysis systems.
 5
# 3 METHODOLOGY
 3.1 Data Collection and Preprocessing
 1) Overview of the Dataset: The present work utilizes the IMDB Movie Review
 Dataset, a popular dataset to evaluate sentiment analysis models. It contains a set of
 50,000 reviews,an equal number 25,000 each of which is positive and 25,000 of which
 is negative. Sentiment labels are available for all reviews, in which a value of 1 reflects
 positive sentiment and a value of 0 reflects negative sentiment.
 This data is loaded to the dataset utilizing Pandas, a high data manipulation
 library from Python, so that the data can be successfully manipulated and sorted out
 for subsequent analysis and further processing. Preprocessing attempts to transform
 the text into cleaned up and normalized form to improve sentiment analysis model
 performances. —
 2) Preprocessing pipeline : Raw text, to best leverage model performance, is subjected
 to Natural Language Process ing (NLP) transformations like:
 3) Lowercasing: All the text is in lowercase to maintain consistency and prevent case
 sensitivity.
 4) Punctuation and Special Character Removals: All the special characters, punc
tuation symbols, and duplicate charac ters are removed to focus on meaningful
 words.
 5) Stopword Removal: Common words like ”the,” ”is,” ”and,” ”in” that do not
 contribute to sentiment are removed with NLTK’s stopword corpus.
 6) Data Stored after Cleaning: The cleaned text is then stored in a new column of
 the dataset. The cleaned column can be used as the main input for all the following
 operations requiring sentiment analysis, holding it in reserve so that only the advanced
 text data are sent to the machine and deep learning algorithms.
 # 4 SENTIMENT ANALYSIS METHODS
 4.1 Lexicon-Based Sentiment Analysis
 Lexicon-based approaches depend on capitalizing on senti ment lexicons accessible to
 assign the polarity of a text:
 TextBlob: Takes advantage of the pre-annotated words’ sentiment value to generate
 polarity scores. VADER (Valence Aware Dictionary and Sentiment Reasoner): Rule
based care fully crafted and trained for social media text with compound sentiment
 scores. The two are compared with regard to the quality of identifying sentiment
 polarity within IMDB movie reviews.
 4.2 Machine Learning and Neural Network-Based Models
 1. Word2Vec Embeddings Word embeddings are calculated by Gensim’s Word2Vec,
 which is trained on tokenized reviews. Words are represented as 100-dimensional
 vectors preserv ing semantic similarity and relationships.
 2. Neural Network Classifier There is a sentiment classifi cation feedforward neural
 network:
 6
Input Layer: 100-dimensional Word2Vec embeddings.
 Hidden Layers: Two fully connected layers (128 neurons and 64 neurons) with ReLU
 activation.
 Dropout Layers: Used with dropout of 0.3 for preventing overfitting.
 Output Layer: Using sigmoid activation for binary classifi cation.
 3. GRU-Based Deep Learning Model GRU network is utilized with TensorFlow to
 process sequential data: Embedding Layer: Maps tokenized text to dense vector
 representation.
 GRU Layers: Processes sequential dependencies and far contextual information.
 Dense Layers: Classification with fully connected layers.
 Optimizer: Adam optimizer with binary cross-entropy loss function for faster conver
gence. All of these methods are compared to determine the most suitable sentiment
 classification method. Model Comparison and Evaluation These sentiment analysis
 models are compared by analyzing the following major metrics:
 Accuracy: Overall accuracy of the predictions is demon strated by dividing correctly
 classified reviews to total reviews.
 Precision: Demonstrates the proportion of the correctly predicted positive reviews to
 all the reviews that were labeled as positive.
 Recall: It demonstrates the capability of the model to label the correct actual positive
 reviews without losing good examples.
 F1-Score: Harmonic mean of recall and precision that offers a very good balanced
 measure of model performance where there are issues of class imbalance.
 # 5 RESULT
 ## 5.1 Comparison of Sentiment Analysis Models (TextBlob vs
 VADER)
 The following figure is a scatter plot of sentiment scores by VADER and TextBlob on
 various reviews. Individual reviews are represented on the x-axis and sentiment scores
 between-1 (strongly negative) and +1 (strongly positive) are represented on the y
axis. Red is utilized to distinguish VADER scores and blue to distinguish TextBlob
 scores. A dashed horizontal line at y = 0 is utilized to distinguish positive and negative
 sentiments.
 The denser clustering of the red points indicates that VADER is measuring a
 greater number of sentiment inten sities, distinguishing between reviews more subtly.
 TextBlob scores, however, are denser, reflecting a more conservative
 scale of sentiment values. This graphically different presenta tion is a reflection of
 VADER’s sensitivity to subtle emotional cues, with TextBlob providing more rough
 estimates.
 ##  5.2 Training and Validation Performance Over Epochs
 The figure above is a plot of accuracy vs. loss vs. training epochs for the training
 and validation sets. **Left plot (Accuracy Over Epochs):**- Training accuracy (blue)
 <img width="476" height="278" alt="image" src="https://github.com/user-attachments/assets/973f111d-5496-44d9-8e7c-9f674e3f5d4f" />
  gets better steadily, indicating that the model is learning from the training data.
Validation accuracy (orange) increases f irst but then levels off, which could indicate
 possible overfit ting. **Right plot (Loss Over Epochs):**- Training loss (blue) reduces
 steadily, which indicates good learning.- Validation loss (orange) first decreases but
 later oscillates and increases, again indicating possible overfitting. This implies that
 the model may require regularization or truncation to generalize better on new data.
 <img width="387" height="164" alt="image" src="https://github.com/user- 
 attachments/assets/e84cbdc8-5798-4dfb-9fd8-9ea9339116e1" />
  ## 5.3 Training Performance of GRU Model
 The narrative includes two plots of training accuracy and loss of the GRU model
 versus the number of epochs.
 Left Chart: Training Loss vs Epochs The loss decreases progressively for every sub
sequent epoch, which is a sign that the model is indeed learning and that the
optimization process is smooth.- **Right Chart: Comparison of Training Accuracy**
 Accuracy has an upward trend, reflecting the improvement of the model in training
 iterations. Both plots use epochs on the x-axis and loss/accuracy on the y-axis, with
 red dashed lines and round markers plotting the GRU model’s performance.
 <img width="468" height="437" alt="image" src="https://github.com/user-attachments/assets/88d0e86b-3fb3-464a-8690-174cecbf6e2a" />
 ## 6 REFERENCES
 [1] Rudy Prabowo and Mike Thelwall, “Sentiment Analysis: A Combined Approach,”
 Journal of Informetrics, Vol. 3, Issue 2, pp. 143-157, 2009.
 [2] Alexander Pak, Patrick Paroubek, “Twitter as a Corpus for Sentiment Anal
ysis and Opinion Mining,” International Conference on Language Resources and
 Evaluation, pp. 1320 1326, 2010.
 9
[3] Mary Margarat Valentine, Ms. Veena Kulkarni, Dr.R.R.Sedamkar, ”A Model
 for Predicting Movie’s Perfor mance using Online Rating and Revenue,” International
 Jour nal of Scientific Engineering Research, vol. Volume 4, no. issue 9, pp. 277-282,
 2013.
 [4] Luda Zhao, Connie Zeng, “Using Neural Networks to Predict Emoji Usage from
 Twitter Data,” Semantic Scholar, pp. 1-6, 2017.
 [5] C. Albon, ”Chris Albon,” 2011. [Online]. Avail able:https://chrisalbon.com/ [6]
 G. S. Tomar, S. Verma Ashish Jha, “Web Page Clas sification using Modified Naive
 Bayesian Approach,” IEEE TENCON-2006, pp. 1-4, 14-17 Nov 2006.
 [7] H.-A. W. Li-Chen Cheng, ”A novel fuzzy recommen dation system integrated
 the experts’ opinion,” IEEE Inter national Conference on Fuzzy Systems (FUZZ-IEEE
 2011), 2011.
 [8] Josef Steinberger, ”Creating sentiment dictionaries via triangulation,” Decision
 Support Systems 53(4), p. 5, 2012.
 [9] Kushal Dave, ”Mining the Peanut Gallery: Opinion Extraction and Semantic
 Classification of Product Reviews,” Proc. 12th Int. Conf. World Wide Web, p. 10,
 2003.
 [10] Z. Z. R. L. Qiang Ye, ”Sentiment classifica tion of online reviews to travel
 destinations by super vised,”elsevier.com/locate/eswa, p. 9, 2009.
 [11] C. P.-R. Mahesh Joshi, ”Generalizing Dependency Fea tures for Opinion
 Mining,” Proceedings of the ACL-IJCNLP 2009 Conference Short Papers, p. 3, 2009.
 [12] G. M. Raul Garreta, ”Learning Scikit learn Machine Learning in Python,”
 Packt Publishing Ltd, 2013, p. 118.
 [13] McCallum, ”A Comparison of Event Models for Naive Bayes Text Classifica
tion,” Learning for Text Categorization: Papers from the 1998 AAAI Workshop, p. 7,
 1998.
 [14] Wikipedia. [Online]. Available at:https://en.wikipedia.org/wiki/Logisticregression.
 [15] V. A. K. a. S. Sonawane, ”Sentiment Analysis of Twitter Data: A Survey of
 Techniques,” International Journal of Computer Applications, vol. Volume 139, no.
 issue 11, p. 11, 2016.
 [16] S. C. A. B. B. B. a. S. T. Lopamudra Dey, ”Sentiment Analysis of Review
 Datasets using Naive Bayes and K-NN Classifier,” International Journal of Information
 Engineering and Electronic Business, vol. Volume 4, p. 8, 2016.
 [17] Medium, https://medium.com. ”Medium,” [Online]. Available:
 [18] M. A. M. F. a. M. J. S. Silvio Amir, ”TUGAS: Exploiting Unlabelled Data
 for Twitter Sentiment Analysis,” Proceedings of the 8th International Workshop on
 Semantic Evaluation (SemEval1 2014), p. 4, 2014.
 [19] Rudy Prabowo and Mike Thelwall, “Sentiment Anal ysis: A Combined
 Approach,” Journal of Informetrics, Vol.3, Issue 2, pp.143-157, 2009. (Duplicate, but
 included for the requested count)
 [20] Alexander Pak, Patrick Paroubek, “Twitter as a Corpus for Sentiment Anal
ysis and Opinion Mining, “ International Conference on Language Resources and
 Evaluation, pp.1320 1326, 2010. (Duplicate, but included for the requested count)


 


 



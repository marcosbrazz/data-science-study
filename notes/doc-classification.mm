<map version="1.0.1">
<!-- To view this file, download free mind mapping software FreeMind from http://freemind.sourceforge.net -->
<node CREATED="1554658753003" ID="ID_895310138" MODIFIED="1554665255217" STYLE="bubble" TEXT="PythonAlgs">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
<node CREATED="1554658837295" ID="ID_1685191302" MODIFIED="1554665611983" POSITION="right" STYLE="bubble" TEXT="Feature Extraction">
<node CREATED="1554665613865" ID="ID_650122442" MODIFIED="1554666540168" TEXT="Text feature extraction">
<node CREATED="1554665666723" ID="ID_1957233411" MODIFIED="1554666765375" TEXT="Extract numerical feature from text">
<node CREATED="1554666765363" ID="ID_730095626" MODIFIED="1554666773359" TEXT="Bag of Words">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
<node CREATED="1554665718315" ID="ID_1855032986" MODIFIED="1554665725951" TEXT="Tokenizing">
<node COLOR="#999999" CREATED="1554665824396" ID="ID_1096219645" MODIFIED="1554666227225" TEXT="tokenizing strings and giving an integer id for each possible token">
<font NAME="SansSerif" SIZE="10"/>
</node>
</node>
<node CREATED="1554665733611" ID="ID_662621033" MODIFIED="1554665736039" TEXT="Counting">
<node COLOR="#999999" CREATED="1554665843300" ID="ID_1879344601" MODIFIED="1554666226497" TEXT="counting the occurrences of tokens in each documen">
<font NAME="SansSerif" SIZE="10"/>
</node>
</node>
<node CREATED="1554665738451" ID="ID_753856199" MODIFIED="1554665742311" TEXT="Normalizing">
<node COLOR="#999999" CREATED="1554665953557" ID="ID_1003065093" MODIFIED="1554665959874" TEXT="normalizing and weighting with diminishing importance tokens that occur in the majority of samples / documents.">
<font NAME="SansSerif" SIZE="10"/>
</node>
<node CREATED="1554665962405" ID="ID_501807079" MODIFIED="1554665970033" TEXT="TFIDF"/>
</node>
<node CREATED="1554666808138" ID="ID_1509796955" MODIFIED="1554666809166" TEXT="Bag of n-grams"/>
</node>
</node>
<node CREATED="1554666246448" HGAP="26" ID="ID_70462042" MODIFIED="1554666854230" TEXT="Terminology" VSHIFT="16">
<node CREATED="1554666251496" ID="ID_1000841837" MODIFIED="1554666256956" TEXT="Feature">
<node CREATED="1554666269184" ID="ID_26606705" MODIFIED="1554666284764" TEXT="Individual token ocurrence frequency (normalized or not)"/>
</node>
<node CREATED="1554666262904" ID="ID_1767330809" MODIFIED="1554666264396" TEXT="Sample">
<node CREATED="1554666334209" ID="ID_1817180731" MODIFIED="1554666345965" TEXT="Vector of all token frequencies of a document"/>
</node>
</node>
<node CREATED="1554666540162" ID="ID_191615788" MODIFIED="1554666586370" TEXT="Vectorization" VSHIFT="24">
<node CREATED="1554666577447" ID="ID_225019695" MODIFIED="1554666583994" TEXT="Turning a collection of text documents into numerical feature vectors">
<node CREATED="1554666664408" ID="ID_891204898" MODIFIED="1554666667813" TEXT="One row per document"/>
<node CREATED="1554666678153" ID="ID_1002411233" MODIFIED="1554666686918" TEXT="One column per token"/>
</node>
<node CREATED="1554658797184" ID="ID_506207471" MODIFIED="1554665689439" STYLE="bubble" TEXT="CountVectorizer" VSHIFT="16">
<node CREATED="1554660695300" ID="ID_194508996" MODIFIED="1554660739816" TEXT="Convert a collection of text to matrix of token counts"/>
<node CREATED="1554667092618" ID="ID_364906710" MODIFIED="1554667096726" TEXT="Tokenization"/>
<node CREATED="1554667098299" ID="ID_648857915" MODIFIED="1554667100926" TEXT="Counting"/>
</node>
<node CREATED="1554658808102" HGAP="16" ID="ID_956100349" MODIFIED="1554665639327" STYLE="bubble" TEXT="TfidfVectorizer" VSHIFT="30">
<node CREATED="1554664104432" ID="ID_1946551214" MODIFIED="1554664110588" TEXT="Transform a count matrix to a normalized tf or tf-idf representation"/>
<node CREATED="1554664134754" ID="ID_1248383020" MODIFIED="1554664142705">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <p>
      <font color="rgb(29, 31, 34)" face="Helvetica, Arial, sans-serif" size="14.4px">The goal of using tf-idf instead of the raw frequencies of occurrence of a token in a given document is to scale down the impact of tokens that occur very frequently in a given corpus and that are hence empirically less informative than features that occur in a small fraction of the training corpus.</font>
    </p>
  </body>
</html></richcontent>
</node>
<node CREATED="1554664190044" ID="ID_395607866" MODIFIED="1554664191336" TEXT=" tf-idf(t, d) = tf(t, d) * idf(t)">
<node CREATED="1554664199301" ID="ID_1576910374" MODIFIED="1554664199657" TEXT="idf(t) = log [ n / df(t) ] + 1"/>
</node>
<node CREATED="1554661317928" ID="ID_1068540183" MODIFIED="1554664100397" TEXT="Document frequency">
<node CREATED="1554661312456" ID="ID_133745542" MODIFIED="1554661333916" TEXT="Is the number of documents in the document set that contain the term t"/>
</node>
</node>
</node>
<node CREATED="1554666008853" ID="ID_1976489160" MODIFIED="1554666013081" TEXT="https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction" VSHIFT="19"/>
</node>
</node>
<node CREATED="1554658923887" ID="ID_1248636847" MODIFIED="1554658933299" POSITION="right" TEXT="Linear Model" VSHIFT="28">
<node CREATED="1554658928447" ID="ID_579876937" MODIFIED="1554658929347" TEXT="LogisticRegression"/>
</node>
<node CREATED="1554659549124" HGAP="24" ID="ID_850579464" MODIFIED="1554659555463" POSITION="right" TEXT="Ensemble" VSHIFT="20">
<node CREATED="1554659560203" ID="ID_349431958" MODIFIED="1554659560679" TEXT="ExtraTreesClassifier"/>
<node CREATED="1554659565043" ID="ID_1858418062" MODIFIED="1554659565631" TEXT="RandomForestClassifier"/>
</node>
<node CREATED="1554665255209" ID="ID_1389056190" MODIFIED="1554665260817" POSITION="right" TEXT="Feature Selectors">
<node CREATED="1554659572835" HGAP="19" ID="ID_116027219" MODIFIED="1554659600751" TEXT="Naive Bayes" VSHIFT="26">
<node CREATED="1554659592636" ID="ID_1371742654" MODIFIED="1554659593183" TEXT="MultinomialNB"/>
<node CREATED="1554659597316" ID="ID_1050724031" MODIFIED="1554659598136" TEXT="BernoulliNB"/>
</node>
</node>
<node CREATED="1554659616812" ID="ID_453109372" MODIFIED="1554659617423" POSITION="right" TEXT="Pipeline"/>
<node CREATED="1554659623580" HGAP="21" ID="ID_703943441" MODIFIED="1554659634040" POSITION="right" TEXT="SVM" VSHIFT="13">
<node CREATED="1554659628212" ID="ID_1463897854" MODIFIED="1554659630128" TEXT="SVC"/>
</node>
<node CREATED="1554659641036" ID="ID_1843160223" MODIFIED="1554659643784" POSITION="left" TEXT="Metrics">
<node CREATED="1554659650036" ID="ID_1202968076" MODIFIED="1554659650480" TEXT="accuracy_score"/>
<node CREATED="1554659656396" ID="ID_1350572425" MODIFIED="1554659656888" TEXT="precision_score"/>
<node CREATED="1554659660676" ID="ID_1060125941" MODIFIED="1554659661072" TEXT="recall_score"/>
<node CREATED="1554659665020" ID="ID_1285353999" MODIFIED="1554659665448" TEXT="confusion_matrix"/>
<node CREATED="1554659669388" ID="ID_852229681" MODIFIED="1554659669840" TEXT="average_precision_score"/>
<node CREATED="1554659674508" ID="ID_1947265801" MODIFIED="1554659674817" TEXT="make_scorer"/>
<node CREATED="1554659678764" ID="ID_789043026" MODIFIED="1554659679952" TEXT="f1_score"/>
<node CREATED="1554659702252" ID="ID_667533872" MODIFIED="1554659703224" TEXT="cross_val_score"/>
</node>
<node CREATED="1554659710564" ID="ID_165777185" MODIFIED="1554659722536" POSITION="left" TEXT="Model Selection" VSHIFT="43">
<node CREATED="1554659727717" ID="ID_1215800101" MODIFIED="1554659727984" TEXT="StratifiedShuffleSplit"/>
<node CREATED="1554659732317" ID="ID_1736322036" MODIFIED="1554659732840" TEXT="learning_curve"/>
</node>
<node CREATED="1554659799053" ID="ID_1969457430" MODIFIED="1554659824009" POSITION="left" TEXT="NLTK" VSHIFT="45">
<node CREATED="1554659807797" ID="ID_315940777" MODIFIED="1554659808033" TEXT="word_tokenize"/>
<node CREATED="1554659813661" ID="ID_1466544028" MODIFIED="1554659814017" TEXT="stopwords"/>
</node>
</node>
</map>

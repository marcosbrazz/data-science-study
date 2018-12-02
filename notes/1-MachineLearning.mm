<map version="1.0.1">
<!-- To view this file, download free mind mapping software FreeMind from http://freemind.sourceforge.net -->
<node CREATED="1535308852335" ID="ID_495337093" MODIFIED="1535312370328" STYLE="bubble" TEXT="Machine Learning">
<font BOLD="true" NAME="SansSerif" SIZE="14"/>
<node COLOR="#990000" CREATED="1535313848706" FOLDED="true" ID="ID_1614390726" MODIFIED="1543785135212" POSITION="left" TEXT="Supervised Learning" VSHIFT="-83">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
<node CREATED="1535308881762" HGAP="27" ID="ID_1237674882" MODIFIED="1543702004123" STYLE="bubble" TEXT="Classification " VSHIFT="-43">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
<node CREATED="1535309190496" ID="ID_53585579" MODIFIED="1535309199101" TEXT="Predict class of a case"/>
<node CREATED="1537132513547" ID="ID_135769754" MODIFIED="1537132526095" TEXT="Algorithms" VSHIFT="8">
<node CREATED="1537132530163" ID="ID_658572627" MODIFIED="1537132538957" TEXT="Decision Trees">
<node CREATED="1537716195306" ID="ID_366174086" MODIFIED="1537716315118" TEXT="Recursive partition">
<node CREATED="1537716087689" ID="ID_1150863333" MODIFIED="1537716109865" TEXT="Choose an attribute from dataset">
<icon BUILTIN="full-1"/>
</node>
<node CREATED="1537716118241" ID="ID_1502116818" MODIFIED="1537716917781" TEXT="Calculate significance of attr ">
<icon BUILTIN="full-2"/>
<node CREATED="1537716489798" ID="ID_774932465" MODIFIED="1537716495788" TEXT="Result is more pure">
<node CREATED="1537716538550" ID="ID_1413923111" MODIFIED="1537716544348" TEXT="Lower entropy"/>
</node>
<node CREATED="1537716917772" HGAP="21" ID="ID_583418330" MODIFIED="1537718997818" TEXT="Entropy E" VSHIFT="12">
<node CREATED="1537716841508" HGAP="18" ID="ID_186992170" MODIFIED="1537717141566" TEXT="homogeneity of samples for each node" VSHIFT="-3"/>
<node CREATED="1537717092554" ID="ID_1027690720" MODIFIED="1537719049722" TEXT="E = -p(A)log(p(A)) - p(B)log(p(B))" VSHIFT="12">
<node COLOR="#999999" CREATED="1537718108851" ID="ID_1199822556" MODIFIED="1537718116828" TEXT="log base 2">
<font NAME="SansSerif" SIZE="10"/>
</node>
</node>
</node>
<node CREATED="1537718628367" HGAP="23" ID="ID_1205806451" MODIFIED="1537719012402" TEXT="Information Gain I" VSHIFT="19">
<node CREATED="1537718730014" ID="ID_1568306316" MODIFIED="1537719018970" TEXT="I = E(bp) - [p(b1)E(b1) + p(b2)E(b2)]">
<node COLOR="#999999" CREATED="1537718912949" ID="ID_574263615" MODIFIED="1537718945230" TEXT="bp = parent branch">
<font NAME="SansSerif" SIZE="10"/>
</node>
<node COLOR="#999999" CREATED="1537718959413" ID="ID_1307171507" MODIFIED="1537718968286" TEXT="b2 = branch 2">
<font NAME="SansSerif" SIZE="10"/>
</node>
<node COLOR="#999999" CREATED="1537718946125" ID="ID_533923700" MODIFIED="1537718968285" TEXT="b1 = branch 1">
<font NAME="SansSerif" SIZE="10"/>
</node>
</node>
</node>
</node>
<node CREATED="1537716142801" ID="ID_1115653749" MODIFIED="1537718651636" TEXT="Split data based on value of best attr" VSHIFT="21">
<icon BUILTIN="full-3"/>
</node>
<node CREATED="1537716755988" HGAP="19" ID="ID_467696528" MODIFIED="1537716775065" TEXT="Minimize the impurity each step" VSHIFT="11"/>
</node>
</node>
<node CREATED="1537132545283" ID="ID_1217358681" MODIFIED="1537132566328" TEXT="Na&#xef;ve Bayes"/>
<node CREATED="1537132571675" ID="ID_111007382" MODIFIED="1537132582185" TEXT="Linear Discriminant Analysis"/>
<node CREATED="1537132593395" ID="ID_904233126" MODIFIED="1537132602353" TEXT="K-Nearest Neighbor">
<node CREATED="1537133142546" ID="ID_1757697356" MODIFIED="1537133259634" TEXT="Similiar cases of same class are near each other"/>
<node CREATED="1537133263854" ID="ID_1591553730" MODIFIED="1537133326292" TEXT="The most neighbors with same class define the predicted class"/>
<node CREATED="1537133352909" ID="ID_201860464" MODIFIED="1537133374404" TEXT="Use Euclidian distance to calculate"/>
<node COLOR="#999999" CREATED="1537709163561" ID="ID_1242119184" MODIFIED="1537709178514" TEXT="sklearn.neighbors.KNeighborsClassifier">
<font NAME="SansSerif" SIZE="10"/>
</node>
</node>
<node CREATED="1537132703018" ID="ID_1558987386" MODIFIED="1543170419521" TEXT="Logistic Regression" VSHIFT="40">
<node CREATED="1543172804235" ID="ID_805097054" MODIFIED="1543172836376" TEXT="Model the prob of input X belongs to the default class">
<node CREATED="1543172863372" ID="ID_1976392057" MODIFIED="1543172872544" TEXT="P(Y=1|X)"/>
</node>
<node CREATED="1543170814393" ID="ID_188973306" MODIFIED="1543170933633">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <p>
      Predict Categorical / Descrete Variables
    </p>
  </body>
</html></richcontent>
</node>
<node CREATED="1543170681313" ID="ID_574107795" MODIFIED="1543170900733" TEXT="Analogous to Linear Regression" VSHIFT="6">
<edge WIDTH="2"/>
<node CREATED="1543170853417" ID="ID_846847610" MODIFIED="1543171016627">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <p>
      Linear predict continue variables
    </p>
  </body>
</html></richcontent>
</node>
</node>
<node CREATED="1543173287020" ID="ID_339759801" MODIFIED="1543173301024" TEXT="Training process" VSHIFT="20">
<node CREATED="1543174499143" ID="ID_684107468" MODIFIED="1543174530348" TEXT="Find best parameters &#x3b8;">
<node CREATED="1543174509439" ID="ID_1833330608" MODIFIED="1543174516731" TEXT="Minimize cost function"/>
</node>
<node CREATED="1543173312964" ID="ID_1512013837" MODIFIED="1543173448242" TEXT="Optimization of &#x3b8; is usually done by Desdendent Gradient"/>
<node CREATED="1543173375893" ID="ID_259286716" MODIFIED="1543173438170" TEXT="Optimization &#x3b8; itaration stops when accuracy is satisfactory"/>
</node>
<node CREATED="1543171216420" HGAP="19" ID="ID_302215625" MODIFIED="1543171251833" TEXT="Good candidate for:" VSHIFT="17">
<node CREATED="1543171236476" ID="ID_1401507381" MODIFIED="1543171245193" TEXT="Binary data"/>
<node CREATED="1543171265189" ID="ID_135082836" MODIFIED="1543171277593" TEXT="You need the probability of prediction">
<node CREATED="1543171331518" ID="ID_432826203" MODIFIED="1543171345706" TEXT="Prob of customer buy a product"/>
</node>
<node CREATED="1543171405342" ID="ID_1321854192" MODIFIED="1543171420795" TEXT="You need a linear decision boundary">
<node CREATED="1543171436975" ID="ID_1334552315" MODIFIED="1543171438611" TEXT="Line"/>
<node CREATED="1543171439791" ID="ID_185684739" MODIFIED="1543171448923" TEXT="Plane"/>
<node CREATED="1543171449367" ID="ID_780354948" MODIFIED="1543171453099" TEXT="Hyper-plane"/>
</node>
<node CREATED="1543171544200" ID="ID_543024647" MODIFIED="1543171555564" TEXT="You need to understand the impact of a feature"/>
</node>
</node>
<node CREATED="1537132711786" HGAP="23" ID="ID_743834340" MODIFIED="1543170417289" TEXT="Neural Networks" VSHIFT="39"/>
<node CREATED="1537132723425" ID="ID_1322366209" MODIFIED="1537132747543" TEXT="Support Vector Machines SVM">
<node CREATED="1535312813001" ID="ID_545685169" MODIFIED="1537132760251" TEXT="SVC">
<font NAME="SansSerif" SIZE="12"/>
<node COLOR="#999999" CREATED="1535312818531" HGAP="18" ID="ID_428788065" MODIFIED="1535312860821" TEXT="Support Vector Classification" VSHIFT="8">
<font NAME="SansSerif" SIZE="10"/>
</node>
<node CREATED="1535312893888" ID="ID_1742444329" MODIFIED="1535312898925" TEXT="Build classifier"/>
<node CREATED="1543778378346" ID="ID_223634890" MODIFIED="1543778382062" TEXT="Kernelling">
<node CREATED="1543778389721" ID="ID_300082501" MODIFIED="1543778404549" TEXT="Map data into higher dimensional space"/>
<node CREATED="1543778612898" ID="ID_929135174" MODIFIED="1543778663102" TEXT="Find a linear separator for data at diff classes"/>
<node CREATED="1543778416457" ID="ID_1861631392" MODIFIED="1543778418613" TEXT="Linear"/>
<node CREATED="1543778420417" ID="ID_390748348" MODIFIED="1543778423557" TEXT="Polynomial"/>
<node CREATED="1543778426497" ID="ID_419206521" MODIFIED="1543778428837" TEXT="RBF"/>
<node CREATED="1543778430457" ID="ID_1496148040" MODIFIED="1543778433285" TEXT="Sigmoid"/>
</node>
<node CREATED="1543778930044" HGAP="22" ID="ID_590343696" MODIFIED="1543778961257" TEXT="Pros" VSHIFT="23">
<node CREATED="1543778937636" ID="ID_1887186050" MODIFIED="1543778948255" TEXT="Accurate in high dimensional spaces"/>
<node CREATED="1543778951139" ID="ID_1309682302" MODIFIED="1543778956520" TEXT="Memory efficient"/>
</node>
<node CREATED="1543778983284" ID="ID_1176849092" MODIFIED="1543778987351" TEXT="Cons">
<node CREATED="1543778988411" ID="ID_1212566351" MODIFIED="1543778994064" TEXT="Prone to overfitting">
<node CREATED="1543778995059" ID="ID_1628777037" MODIFIED="1543779017512" TEXT="Number of feature is much greater than samples"/>
</node>
<node CREATED="1543779030523" ID="ID_616846968" MODIFIED="1543779036680" TEXT="No prob. estimations"/>
<node CREATED="1543779069798" ID="ID_300993660" MODIFIED="1543779074163" TEXT="Small datasets">
<node CREATED="1543779093679" ID="ID_37773301" MODIFIED="1543779100628" TEXT="&lt; 1000 rows"/>
</node>
</node>
<node CREATED="1543779126209" HGAP="22" ID="ID_1095209630" MODIFIED="1543779151999" TEXT="Good for" VSHIFT="15">
<node CREATED="1543779130242" ID="ID_616768883" MODIFIED="1543779138622" TEXT="Image analysis tasks">
<node CREATED="1543779141722" ID="ID_1830395049" MODIFIED="1543779147943" TEXT="recognition"/>
</node>
<node CREATED="1543779175964" HGAP="22" ID="ID_1908150763" MODIFIED="1543779230427" TEXT="Text-mining tasks" VSHIFT="10">
<node CREATED="1543779215278" ID="ID_224457296" MODIFIED="1543779220098" TEXT="Detect span"/>
<node CREATED="1543779222502" ID="ID_161476750" MODIFIED="1543779227459" TEXT="Sentiment analysis"/>
</node>
</node>
</node>
</node>
</node>
<node CREATED="1537134775507" ID="ID_265036794" MODIFIED="1543701950370" TEXT="Evaluation Model" VSHIFT="28">
<node CREATED="1537134815219" ID="ID_1289879733" MODIFIED="1543701874902" TEXT="Jaccard index">
<node CREATED="1537134871323" ID="ID_1275097805" MODIFIED="1537134996341">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <img src="jaccard_model.png"/>
  </body>
</html></richcontent>
</node>
<node CREATED="1537135008978" ID="ID_881991789" MODIFIED="1537135016973">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <img src="jaccard_example.png"/>
  </body>
</html></richcontent>
</node>
<node CREATED="1543701963790" ID="ID_58546206" MODIFIED="1543701976074" TEXT="Accuracy metric"/>
</node>
<node CREATED="1537134823579" HGAP="21" ID="ID_765837597" MODIFIED="1543701944297" TEXT="F1-score" VSHIFT="21">
<node CREATED="1537135067329" ID="ID_1179042029" MODIFIED="1543165942879" TEXT="Confusion Matrix">
<node CREATED="1543165702378" ID="ID_1174002244" MODIFIED="1543165715383">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <img src="confusion-matrix-example.png" />
  </body>
</html></richcontent>
</node>
</node>
<node CREATED="1537135911595" ID="ID_1334204124" MODIFIED="1537137195303" TEXT="Precision" VSHIFT="15"/>
<node CREATED="1537136036698" ID="ID_1933769478" MODIFIED="1537137193511" TEXT="Recall" VSHIFT="11"/>
<node CREATED="1543168096883" HGAP="21" ID="ID_1784970856" MODIFIED="1543168115175" TEXT="Used for multiclass classification" VSHIFT="8"/>
<node CREATED="1537137393417" HGAP="21" ID="ID_1940744365" MODIFIED="1537137439901" TEXT="F-score" VSHIFT="13">
<node CREATED="1537137402993" ID="ID_1148330274" MODIFIED="1537137433519" TEXT="2*(prec*rec)/(prec + rec)"/>
<node CREATED="1543162803060" ID="ID_448765068" MODIFIED="1543162804256" TEXT="the harmonic mean of precision and recall"/>
<node CREATED="1543703515533" ID="ID_623952038" MODIFIED="1543703516570" TEXT="good way to show that a classifer has a good value for both recall and precision"/>
<node CREATED="1543703693521" ID="ID_1571631301" MODIFIED="1543703708845" TEXT="Values from 0 - 1."/>
<node CREATED="1543703709145" ID="ID_724170091" MODIFIED="1543703719965" TEXT="1 is the best score"/>
</node>
</node>
<node CREATED="1537134831307" ID="ID_1062271131" MODIFIED="1543162811624" TEXT="Log loss" VSHIFT="25">
<node CREATED="1537137674319" ID="ID_982790660" MODIFIED="1537137685693" TEXT="Probability of a class"/>
<node CREATED="1537137588592" ID="ID_1837487638" MODIFIED="1537137658637" TEXT="Measure performance when the output is a probability between 0 - 1"/>
<node CREATED="1537137769862" ID="ID_356902506" MODIFIED="1537137781178">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <img src="log_loss_model.png"/>
  </body>
</html></richcontent>
</node>
<node CREATED="1537137797342" ID="ID_829917380" MODIFIED="1537137808276" TEXT="Lower log loss = better accuracy"/>
</node>
<node COLOR="#999999" CREATED="1537709119580" ID="ID_165197053" MODIFIED="1537709140494" TEXT="sklearn.metrics" VSHIFT="7">
<font NAME="SansSerif" SIZE="10"/>
</node>
</node>
</node>
<node CREATED="1535308869637" FOLDED="true" HGAP="16" ID="ID_1761901146" MODIFIED="1543785132287" STYLE="bubble" TEXT="Regression/Estimation">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
<node CREATED="1535309113152" ID="ID_38658008" MODIFIED="1535309133534" TEXT="Predict continuous values">
<node CREATED="1535309151760" ID="ID_1204087170" MODIFIED="1535309169821" TEXT="Case of price of a house..."/>
</node>
<node CREATED="1535315120026" ID="ID_1764385559" MODIFIED="1535315158767" TEXT="X" VSHIFT="11">
<node CREATED="1535315131722" ID="ID_1804467834" MODIFIED="1535315142151" TEXT="Independent variable"/>
<node CREATED="1535315288433" ID="ID_670040782" MODIFIED="1535315299502" TEXT="Explanatory variables"/>
<node CREATED="1535315335985" ID="ID_862885034" MODIFIED="1535315346630" TEXT="Causes of those states"/>
</node>
<node CREATED="1535315121938" HGAP="21" ID="ID_1684826600" MODIFIED="1535315157166" TEXT="Y" VSHIFT="17">
<node CREATED="1535315144450" ID="ID_401209426" MODIFIED="1535315154087" TEXT="Dependent variable"/>
<node CREATED="1535315175778" ID="ID_1847170667" MODIFIED="1535315189847" TEXT="What I want to predict"/>
<node CREATED="1535315246521" ID="ID_1864146752" MODIFIED="1535315258030" TEXT="Final goal of study"/>
<node CREATED="1535315263897" ID="ID_1672404741" MODIFIED="1535315266583" TEXT="target"/>
<node CREATED="1535315659079" ID="ID_974350419" MODIFIED="1535315686404" TEXT="Must be continuous value">
<icon BUILTIN="yes"/>
</node>
</node>
<node CREATED="1535317709890" ID="ID_569200297" MODIFIED="1535318749278">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <p>
      &#x177; = &#x3b8;<sub>0 + </sub>&#x3b8;<sub>1</sub>&#xa0;x<sub>1</sub>
    </p>
  </body>
</html></richcontent>
<node CREATED="1535318780067" ID="ID_935273943" MODIFIED="1535325487730">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <p>
      &#x3b8;<sub>0</sub>&#xa0;= BIAS / Interceptor
    </p>
  </body>
</html></richcontent>
</node>
<node CREATED="1535318864459" ID="ID_1610783869" MODIFIED="1535318891879">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <p>
      &#x3b8;<sub>1</sub>&#xa0;= Slope&#xa0;
    </p>
  </body>
</html></richcontent>
</node>
<node CREATED="1535318931770" ID="ID_502224622" MODIFIED="1535318951248" TEXT="Find optimum &#x3b8;"/>
<node CREATED="1536022556437" ID="ID_872703031" MODIFIED="1536022619554" TEXT="Many factors tend to over-fit ">
<icon BUILTIN="messagebox_warning"/>
</node>
<node CREATED="1536022720236" ID="ID_1512261261" MODIFIED="1536022726858" TEXT="Scatterplots">
<node CREATED="1536022732763" ID="ID_1974068471" MODIFIED="1536022795312">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <p>
      Check <b>linearity relationship</b>&#xa0;between
    </p>
    <p>
      dependent and <b>independent variables</b>
    </p>
  </body>
</html></richcontent>
<node CREATED="1536022809019" ID="ID_506677540" MODIFIED="1536022841580" TEXT="Use Non-linear regression for non linear relationship"/>
</node>
</node>
</node>
<node CREATED="1535315912293" HGAP="10" ID="ID_655254979" MODIFIED="1536710083419" TEXT="Algoritms" VSHIFT="80">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
<node CREATED="1535315921853" ID="ID_996962811" MODIFIED="1535315940810" TEXT="Ordinal Regression"/>
<node CREATED="1535315943005" ID="ID_1860001174" MODIFIED="1535316047587" TEXT="Poisson Regression"/>
<node CREATED="1535315963845" ID="ID_1411896454" MODIFIED="1535316036826" TEXT="Fast Forest Quantile"/>
<node CREATED="1535315999173" ID="ID_1926614781" MODIFIED="1535316032162" TEXT="Linear, Polynomial, Lasso, Stepwise, Ridge"/>
<node CREATED="1535316076556" ID="ID_1628137033" MODIFIED="1535316098257" TEXT="Bayesian Linear Regression"/>
<node CREATED="1535316104316" ID="ID_1109930666" MODIFIED="1535316120393" TEXT="Neural Network Regression"/>
<node CREATED="1535316137892" ID="ID_1849839264" MODIFIED="1535316147257" TEXT="Decision Forest Regression"/>
<node CREATED="1535316155092" ID="ID_1515487730" MODIFIED="1535316172377" TEXT="Boosted Decision Tree Regression"/>
<node CREATED="1535316180972" ID="ID_1463095856" MODIFIED="1535316208281" TEXT="KNN (K-nearest neighbors)"/>
</node>
<node CREATED="1535319669422" ID="ID_1950518613" MODIFIED="1535319672125" TEXT="FAST"/>
<node CREATED="1535319687190" ID="ID_1572365761" MODIFIED="1535319700203" TEXT="Does not require tuning the parameters"/>
<node CREATED="1536710066729" ID="ID_1214362514" MODIFIED="1543782570278" TEXT="Evaluation">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
<node CREATED="1536023301967" HGAP="22" ID="ID_1695063362" MODIFIED="1536023313500" TEXT="Model Evaluation" VSHIFT="18">
<node CREATED="1536023965747" ID="ID_1626091772" MODIFIED="1536023971552" TEXT="Err Calc">
<node CREATED="1536023972571" ID="ID_473547512" MODIFIED="1536024039142">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <img src="err_calc.png"/>
  </body>
</html></richcontent>
</node>
</node>
<node CREATED="1536024269305" ID="ID_1371405825" MODIFIED="1536024276710" TEXT="Train / test split"/>
<node CREATED="1536024501655" ID="ID_1059294481" MODIFIED="1536024521277" TEXT="K-Fold Cross-validation"/>
</node>
<node CREATED="1536710153267" ID="ID_204134866" MODIFIED="1536710315472" TEXT="Accuracy Metrics">
<node CREATED="1535316829416" HGAP="22" ID="ID_1410269793" MODIFIED="1536710131631" TEXT="MSE (mean squared error)" VSHIFT="72">
<node CREATED="1535316885703" ID="ID_530375383" MODIFIED="1535316895037" TEXT="Mean of residual errors"/>
<node CREATED="1535316905447" ID="ID_166837669" MODIFIED="1535316951612" TEXT="Shows how poorly the line fits the whole dataset"/>
<node CREATED="1535316962015" HGAP="18" ID="ID_14142957" MODIFIED="1535317090994" TEXT="The target is to minimize this" VSHIFT="20">
<node CREATED="1535317063062" ID="ID_1768274508" MODIFIED="1536021943043" TEXT="1 Mathematical approach">
<node CREATED="1536021943001" ID="ID_962422759" MODIFIED="1536022317006" TEXT="Ordinary Least Square">
<node CREATED="1535319172977" ID="ID_99708365" MODIFIED="1535319181652">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <img src="find_theta.png"/>
  </body>
</html></richcontent>
<node CREATED="1535319395791" ID="ID_1288806" MODIFIED="1535319416237" TEXT="x&#x304; = mean of all x in datase"/>
</node>
<node CREATED="1536022010345" HGAP="21" ID="ID_1419818047" MODIFIED="1536022319683" TEXT="Good for less than 10k training set" VSHIFT="24"/>
</node>
</node>
<node CREATED="1535317073558" ID="ID_349309551" MODIFIED="1536022080317" TEXT="2 Optimization approach" VSHIFT="39">
<node CREATED="1536022070792" ID="ID_801567118" MODIFIED="1536022077303" TEXT="Gradient Descent">
<node CREATED="1536022108040" ID="ID_880631433" MODIFIED="1536022112951" TEXT="Large dataset"/>
</node>
</node>
</node>
</node>
<node CREATED="1536710223906" ID="ID_1693016768" MODIFIED="1536710234177" TEXT="MAE (mean absolute error)"/>
<node CREATED="1536710242826" ID="ID_797690613" MODIFIED="1536710273230" TEXT="RMSE (root mean square error)" VSHIFT="34"/>
<node CREATED="1536710734975" ID="ID_532814293" MODIFIED="1536710746965" TEXT="RSE (relative square area">
<node CREATED="1536710748015" ID="ID_646518668" MODIFIED="1536710753133" TEXT="Most popular"/>
<node CREATED="1536710757847" ID="ID_17982276" MODIFIED="1536710766717" TEXT="R&#xb2; = 1 - RSE">
<node CREATED="1536710782719" ID="ID_1117728033" MODIFIED="1536710783485" TEXT="The higher the R-squared, the better the model fits your data"/>
</node>
</node>
</node>
</node>
<node CREATED="1536711757760" ID="ID_880934135" MODIFIED="1536711816654" TEXT="How to Model">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
<node CREATED="1536711762088" ID="ID_1891037063" MODIFIED="1536711768983" TEXT="Non linear regretion">
<node CREATED="1536711776216" ID="ID_842359840" MODIFIED="1536711783646" TEXT="Polynomial regression"/>
<node CREATED="1536711787064" ID="ID_878164369" MODIFIED="1536711794270" TEXT="Non linear regression model"/>
<node CREATED="1536711798080" ID="ID_234282394" MODIFIED="1536711805870" TEXT="Transform your data"/>
</node>
</node>
</node>
</node>
<node CREATED="1535309731252" FOLDED="true" HGAP="25" ID="ID_1578045562" MODIFIED="1543785116169" POSITION="left" TEXT="Sequence Mining" VSHIFT="58">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
<node CREATED="1535309980099" ID="ID_1877022811" MODIFIED="1535309986224" TEXT="Predict next event">
<node CREATED="1535309989859" ID="ID_1236999203" MODIFIED="1535310287966" TEXT="Click stream in websites"/>
</node>
<node COLOR="#338800" CREATED="1535309998275" ID="ID_1327925236" MODIFIED="1535310407441" TEXT="Markov model">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
</node>
<node COLOR="#338800" CREATED="1535310026994" ID="ID_1309807829" MODIFIED="1535310407441" TEXT="HMM">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
</node>
</node>
<node COLOR="#990000" CREATED="1535314049905" HGAP="27" ID="ID_1246338472" MODIFIED="1535314159408" POSITION="right" TEXT="Unsupervised Learning" VSHIFT="27">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
<node CREATED="1535309754972" HGAP="17" ID="ID_782429593" MODIFIED="1535314144157" TEXT="Dimension Reduction" VSHIFT="14">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
<node CREATED="1535310319976" ID="ID_2470396" MODIFIED="1535310327694" TEXT="Reduce the size of data">
<node CREATED="1535314518830" ID="ID_1038542011" MODIFIED="1535314530331" TEXT="Reduce redundant feature"/>
</node>
<node CREATED="1535314497974" ID="ID_643684443" MODIFIED="1535314504883" TEXT="Feature selection"/>
<node COLOR="#338800" CREATED="1535310328921" ID="ID_450241615" MODIFIED="1535310407440" TEXT="PCA">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
</node>
</node>
<node CREATED="1535308939049" HGAP="17" ID="ID_1423633852" MODIFIED="1535314152509" STYLE="bubble" TEXT="Clustering" VSHIFT="13">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
<node CREATED="1535309234319" ID="ID_1266368841" MODIFIED="1535309238189" TEXT="Segmentation"/>
<node CREATED="1535309254223" ID="ID_1654335031" MODIFIED="1535309259469" TEXT="Summarization"/>
<node CREATED="1535309504062" ID="ID_1265369071" MODIFIED="1535309516683" TEXT="Group similar cases"/>
<node CREATED="1535309706908" HGAP="25" ID="ID_70385356" MODIFIED="1535314808433" TEXT="Anomaly Detection" VSHIFT="9">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
<node CREATED="1535309920075" ID="ID_691055979" MODIFIED="1535309957696" TEXT="Discover abnormal / unusual cases">
<node CREATED="1535309958787" ID="ID_165860836" MODIFIED="1535309966968" TEXT="Fraud detection"/>
</node>
</node>
<node CREATED="1535314093217" HGAP="23" ID="ID_1594375333" MODIFIED="1535314787322" TEXT="Density Estimation" VSHIFT="10">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
<node CREATED="1535314675213" ID="ID_1802875854" MODIFIED="1535314692154" TEXT="Find some structure in the data"/>
</node>
<node CREATED="1543783284058" ID="ID_1469449637" MODIFIED="1543783300654" TEXT="Why clustering" VSHIFT="8">
<node CREATED="1543783303986" ID="ID_1962136255" MODIFIED="1543783310270" TEXT="Exploratory data analysis"/>
<node CREATED="1543783313370" ID="ID_279529729" MODIFIED="1543783319455" TEXT="Summary generation"/>
<node CREATED="1543783321643" ID="ID_1589068358" MODIFIED="1543783329503" TEXT="Outlier detection">
<node CREATED="1543783330483" ID="ID_804964563" MODIFIED="1543783335135" TEXT="Fraud prevent"/>
<node CREATED="1543783335387" ID="ID_1003421837" MODIFIED="1543783339799" TEXT="Noise removal"/>
</node>
<node CREATED="1543783345371" ID="ID_400061649" MODIFIED="1543783352696" TEXT="Find duplicates"/>
<node CREATED="1543783356156" ID="ID_834925334" MODIFIED="1543783363360" TEXT="Pre-processing step"/>
<node CREATED="1543787031742" ID="ID_838081536" MODIFIED="1543787032426" TEXT="Data compression"/>
</node>
<node CREATED="1543783381253" HGAP="19" ID="ID_1160343302" MODIFIED="1543783460211" TEXT="Algorithms" VSHIFT="17">
<node CREATED="1543783393509" ID="ID_388197350" MODIFIED="1543783411209" TEXT="Partitioned-based clustering">
<node CREATED="1543783542065" ID="ID_1918291263" MODIFIED="1543783553477" TEXT="Produces sphere like clusters"/>
<node CREATED="1543783480367" ID="ID_833120282" MODIFIED="1543783488803" TEXT="Relatively efficient">
<node CREATED="1543783572257" ID="ID_1794558735" MODIFIED="1543783582470" TEXT="Medium/large databases"/>
</node>
<node CREATED="1543783494376" ID="ID_1446920401" MODIFIED="1543783498979" TEXT="K-means">
<node CREATED="1543783880480" ID="ID_521406812" MODIFIED="1543783888652" TEXT="non-overlapping clusters"/>
<node CREATED="1543783966913" ID="ID_896617632" MODIFIED="1543783975613" TEXT="Minimize intra-cluster distances">
<node CREATED="1543784067947" ID="ID_1122027495" MODIFIED="1543784073351" TEXT="Euclidian distance"/>
<node COLOR="#999999" CREATED="1543784629860" ID="ID_1552661570" MODIFIED="1543785746381">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <p>
      Must understand the nature of data to choose a distance measurement
    </p>
  </body>
</html>
</richcontent>
<font NAME="SansSerif" SIZE="10"/>
<icon BUILTIN="messagebox_warning"/>
</node>
</node>
<node CREATED="1543783975865" ID="ID_1010961761" MODIFIED="1543783991350" TEXT="Maximize inter-clusters distances"/>
<node CREATED="1543785459944" ID="ID_1719849490" MODIFIED="1543785470116" TEXT="Iterative algorithm">
<node CREATED="1543785475824" ID="ID_749985155" MODIFIED="1543785488205" TEXT="Find best point for centroid">
<node CREATED="1543785490600" ID="ID_1071769252" MODIFIED="1543785510621" TEXT="Only guarantee local optimum">
<node CREATED="1543785551329" ID="ID_280304033" MODIFIED="1543785567269" TEXT="Run multiple times w/ diff initial conditions"/>
</node>
</node>
</node>
<node CREATED="1543785160524" ID="ID_1229797659" MODIFIED="1543785163448" TEXT="Evaluation">
<node CREATED="1543785164844" ID="ID_1575962736" MODIFIED="1543785167056" TEXT="SSE">
<node COLOR="#999999" CREATED="1543785168076" ID="ID_238514956" MODIFIED="1543785235535" TEXT="Sum of Squared diff between each point and centroid">
<font NAME="SansSerif" SIZE="10"/>
</node>
</node>
<node CREATED="1543785920966" ID="ID_711789569" MODIFIED="1543785928035" TEXT="Check how bad is a cluster">
<node CREATED="1543785944704" ID="ID_84044201" MODIFIED="1543785997184">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <p>
      Calculate the average distance
    </p>
    <p>
      between data points within a cluster
    </p>
  </body>
</html>
</richcontent>
<node CREATED="1543786055720" ID="ID_1608510641" MODIFIED="1543786066068" TEXT="Indicate how dense are clusters"/>
</node>
</node>
<node CREATED="1543786175986" ID="ID_884327040" MODIFIED="1543786187630" TEXT="Right K for clustering">
<node CREATED="1543786190682" ID="ID_1292994054" MODIFIED="1543786260655" TEXT="Run the clustering across diff vals of K">
<icon BUILTIN="full-1"/>
</node>
<node CREATED="1543786267595" ID="ID_999694432" MODIFIED="1543786365553" TEXT="Calculate the accuracy: SSE">
<icon BUILTIN="full-2"/>
</node>
<node CREATED="1543786302196" ID="ID_1852275538" MODIFIED="1543786428754" TEXT="Choose the Elbow point for best K">
<icon BUILTIN="full-3"/>
</node>
</node>
</node>
</node>
<node CREATED="1543783503248" ID="ID_1068554180" MODIFIED="1543783508188" TEXT="K-median"/>
<node CREATED="1543783515496" ID="ID_1462448010" MODIFIED="1543783520132" TEXT="Fuzzy c-means"/>
</node>
<node CREATED="1543783416677" ID="ID_1257718812" MODIFIED="1543783616806" TEXT="Hierarchical clustering" VSHIFT="11">
<node CREATED="1543783604506" ID="ID_771829293" MODIFIED="1543783611885" TEXT="Produces trees of clustering"/>
<node CREATED="1543783669427" ID="ID_1344361377" MODIFIED="1543783675247" TEXT="Small datasets"/>
<node CREATED="1543783624058" ID="ID_1908165718" MODIFIED="1543783629358" TEXT="Agglomeratives"/>
<node CREATED="1543783630195" ID="ID_1880474465" MODIFIED="1543783639502" TEXT="Divisive"/>
</node>
<node CREATED="1543783440606" ID="ID_1731771880" MODIFIED="1543783739649" TEXT="Density-based clustering" VSHIFT="14">
<node CREATED="1543783687524" ID="ID_801688368" MODIFIED="1543783696208" TEXT="Produces arbitrary shape clusters"/>
<node CREATED="1543783717845" ID="ID_169268738" MODIFIED="1543783721137" TEXT="Good for">
<node CREATED="1543783722021" ID="ID_1908495688" MODIFIED="1543783726449" TEXT="Spacial cluster"/>
<node CREATED="1543783726667" ID="ID_28270363" MODIFIED="1543783735041" TEXT="Noise in dataset"/>
</node>
<node CREATED="1543783703868" ID="ID_1276900198" MODIFIED="1543783706976" TEXT="DBSCAN"/>
</node>
</node>
</node>
<node CREATED="1535308944369" HGAP="14" ID="ID_659819367" MODIFIED="1535314580866" TEXT="Association" VSHIFT="41">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
<node CREATED="1535309634277" ID="ID_186276149" MODIFIED="1535309642115" TEXT="Find itens that co-occur">
<node CREATED="1535309659741" ID="ID_805176639" MODIFIED="1535309676322" TEXT="Ex: Grocery items bought together"/>
</node>
<node CREATED="1535314111777" ID="ID_1965048133" MODIFIED="1535314147648" TEXT="Market Basket Analysis">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
</node>
</node>
</node>
<node CREATED="1535309777044" FOLDED="true" ID="ID_1966847875" MODIFIED="1543782589406" POSITION="left" TEXT="Recommendation Systems" VSHIFT="62">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
<node CREATED="1535310446632" ID="ID_734411224" MODIFIED="1535310777178">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <p>
      Associate people's preferences
    </p>
    <p>
      to others who have similar tastes
    </p>
  </body>
</html></richcontent>
</node>
<node CREATED="1535310423672" ID="ID_500909541" MODIFIED="1535310435973" TEXT="Recommend items"/>
</node>
<node CREATED="1535312149253" HGAP="127" ID="ID_705845476" MODIFIED="1535321866553" POSITION="right" TEXT="PIPELINE" VSHIFT="118">
<node CREATED="1535312163229" HGAP="17" ID="ID_1344542010" MODIFIED="1535312335168" TEXT="Data Preprocessing" VSHIFT="-30">
<icon BUILTIN="full-1"/>
<node CREATED="1535312482955" ID="ID_443684770" MODIFIED="1535312489378" TEXT="Feature selection"/>
<node CREATED="1535312489595" ID="ID_1592505691" MODIFIED="1535312500239" TEXT="Feature extraction" VSHIFT="4"/>
</node>
<node CREATED="1535312228285" HGAP="23" ID="ID_468274936" MODIFIED="1535312337488" TEXT="Train / Test split" VSHIFT="-26">
<icon BUILTIN="full-2"/>
</node>
<node CREATED="1535312239476" HGAP="18" ID="ID_571996513" MODIFIED="1535312338984" TEXT="Algorithm Setup" VSHIFT="-7">
<icon BUILTIN="full-3"/>
</node>
<node CREATED="1535312249916" HGAP="15" ID="ID_1062588294" MODIFIED="1535312339824" TEXT="Model fitting" VSHIFT="15">
<icon BUILTIN="full-4"/>
<node CREATED="1535312520179" ID="ID_1169130066" MODIFIED="1535312530631" TEXT="Tuning parameters"/>
</node>
<node CREATED="1535312256596" ID="ID_1551472449" MODIFIED="1535312341104" TEXT="Prediction" VSHIFT="25">
<icon BUILTIN="full-5"/>
</node>
<node CREATED="1535312266212" ID="ID_1462499217" MODIFIED="1535312342088" TEXT="Evaluation" VSHIFT="27">
<icon BUILTIN="full-6"/>
</node>
<node CREATED="1535312270532" HGAP="14" ID="ID_1366129536" MODIFIED="1535312343528" TEXT="Model Export" VSHIFT="39">
<icon BUILTIN="full-7"/>
</node>
</node>
<node COLOR="#006699" CREATED="1535321799648" HGAP="73" ID="ID_1754187519" MODIFIED="1535321853696" POSITION="right" TEXT="IBM SPSS Modeler" VSHIFT="78">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
</node>
<node CREATED="1543162705260" HGAP="13" ID="ID_450850472" MODIFIED="1543162730038" POSITION="right" TEXT="Metrics" VSHIFT="61">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
<node CREATED="1543162742037" ID="ID_821170003" MODIFIED="1543162743128" TEXT="Accuracy">
<node CREATED="1543162990813" ID="ID_682700857" MODIFIED="1543162998241" TEXT="The proportion of the total number of predictions that were correct">
<node CREATED="1543166123049" ID="ID_1830891496" MODIFIED="1543166156934" TEXT="How many predictions are correct into all samples"/>
</node>
<node CREATED="1543165305206" ID="ID_313544774" MODIFIED="1543165591778" TEXT="(TP+TN)/(TP + TN + FP + FN)">
<node CREATED="1543166698238" ID="ID_1954440241" MODIFIED="1543166706777" TEXT="Quantas predi&#xe7;&#xf5;es eu acertei"/>
</node>
<node CREATED="1543702280811" ID="ID_245757457" MODIFIED="1543702288655" TEXT="Confusion Matrix"/>
</node>
<node CREATED="1543162758229" HGAP="19" ID="ID_9594843" MODIFIED="1543702293439" TEXT="Precision" VSHIFT="22">
<node CREATED="1543165804398" ID="ID_129283974" MODIFIED="1543165815890" TEXT="The fraction of actual positives among those examples that are predicted as positive"/>
<node CREATED="1543165848599" HGAP="27" ID="ID_865279875" MODIFIED="1543166574089" TEXT="TP/(TP + FP)" VSHIFT="6">
<node CREATED="1543166532224" ID="ID_1163646920" MODIFIED="1543166653584">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <p>
      Quantos eu <b>classifiquei certo</b>&#160;como positivo dentre <b>todos os que eu classifiquei como positivo</b>
    </p>
  </body>
</html></richcontent>
</node>
</node>
</node>
<node CREATED="1543162768572" HGAP="23" ID="ID_191534666" MODIFIED="1543702295863" TEXT="Recall" VSHIFT="17">
<node CREATED="1543166201332" ID="ID_1533827391" MODIFIED="1543166234530" TEXT="How many positives were measured as positives"/>
<node CREATED="1543166237782" ID="ID_1381513930" MODIFIED="1543166265923" TEXT="TP/(TP + FN)">
<node CREATED="1543166583444" ID="ID_1566497803" MODIFIED="1543166668447">
<richcontent TYPE="NODE"><html>
  <head>
    
  </head>
  <body>
    <p>
      Quantos eu <b>classifiquei certo</b>&#160;como positivo dentre <b>todos os positivos</b>
    </p>
  </body>
</html></richcontent>
</node>
</node>
</node>
</node>
</node>
</map>

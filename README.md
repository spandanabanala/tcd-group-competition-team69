# tcd-group-competition-team69

1.Spandana Banala
2.Rajat Gogna


INCOME PREDICTION
The goal of this assignment is to let you gain experience with machine learning, not only the algorithms but the entire machine learning pipeline. To do so, we provide a synthetic dataset that contains a list of persons and their yearly salaries, whereas for each person a number of features are provided (e.g. age, nationality, and profession). Your goal is to train a machine learning model that predicts a person´s income based on the given features.

We recommend using https://scikit-learn.org for this competition, but you are free to use whatever machine learning library you want as long as it allows exporting its source code. The exported source code must make it easy to see what exactly you did (which algorithm you used, and how you did feature selection etc.). Hence, you must not use AutoML libraries like AutoWeka and Auto-Sklearn as these would not allow you exporting the source code with details on what algorithms and parameters were used.

Also, machine learning libraries with a Graphical User Interface may not be suitable such as KNIME, WEKA and MLJAR. These libraries allow you to create machine learning pipelines in a graphical user interface. It may be difficult to export the source code and see easily what these libraries do "under the hood". That being said, we have not used these libraries ourselves. If you want to use a GUI ML library, and believe that it can export a single file that shows what exactly was done (which algorithm was used, how missing values were handled, …), please ask in the forum if you can use that library, and post an example how the exported file looks like.

Especially for beginners, we would not recommend using deep learning libraries like TensorFlow or Keras. These are often more difficult to use and for the given problem (income prediction) probably not suitable -- typically, you do not need deep learning to predict a person's income, and there is also not so much data in the dataset that deep learning seems promising. Anyway, if you want to try them, go ahead. Also, please note that you are not restricted to using the algorithms that we cover in the lecture. Use whatever algorithms the libraries offer.

If you want, you can use the algorithms that you programmed in the programming assignments, but we would not recommend this. It is very likely that algorithms in e.g. scikit-learn and other functions like feature selection are much better implemented in scikit-learn than when implemented by yourself.

We recommend getting started with the competition as soon as possible. Have a look at the data, download and install scikit-learn, and try out a first simple model (e.g. linear regression). There are plenty of tutorials on how to use scikit-learn to solve regression problems https://www.google.com/search?q=scikit+learn+linear+regression+example (this is not to mean that linear regression will be the best algorithm to participate in the competition, but it is a good starting point to get used to ML libraries).

Marking
You will receive marks mostly based on how well your solution performs compared to your peers (the best performing solutions will receive the highest marks, though I may look at individual solutions and mark solutions up or down if appropriate). Performance is measured as root-mean-square-error (RMSE) for the predicted income. However, be aware that there is a "private leaderboard". When you upload your data to Kaggle you see a public score. This score is based on different data than the private and final score. Hence, you can expect a sligth variance between your public and private scores.

Please note that the deadline for submission on Kaggle is a very hard deadline. After the deadline, you will not be able to submit any more solutions. So, start early with the competition and upload solutions soon, so that in case you experience problems e.g. with registration or uploading solutions, you have enough time to fix them. Only solutions submitted before the deadline will be marked!!!

Keep in mind that you will be assigned to groups for the second competition primarily based on how well you perform in this individual competition. So, there are two motivations to perform well in this competition: the marks for this assignment, and having capable peers for the second competition.

Download the data
Once you registered, and are logged in, you need to "join" the competition (see the blue button at top right). IMPORTANT: To join the competition, you need to use the link that we posted in blackboard. It begins with https://www.kaggle.com/t/515c3a6044e… . Once you clicked that link, at the top of this page, in the menu, click the "Data" link, and download the following files.

tcd ml 2019-20 income prediction training (with labels).csv.zip
tcd ml 2019-20 income prediction test (without labels).csv.zip
tcd ml 2019-20 income prediction submission file.csv.zip

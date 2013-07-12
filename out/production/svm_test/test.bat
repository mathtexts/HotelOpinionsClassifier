java -Xmx1024m -classpath .;lib/jnisvmlight.jar -Djava.library.path=lib LingvoSVM test myfeature.txt reviews_pos_test1.txt reviews_neg_test1.txt model.txt files/stop.txt
pause
java -Xmx1024m -classpath .;lib/jnisvmlight.jar -Djava.library.path=lib LingvoSVM train newfeatures.txt reviews_pos_train.txt reviews_neg_train.txt model.txt files/stop.txt
pause
java -Xmx1024m -classpath .;lib/jnisvmlight.jar -Djava.library.path=lib LingvoSVM test newfeatures.txt reviews_pos_test.txt reviews_neg_test.txt model.txt files/stop.txt
pause
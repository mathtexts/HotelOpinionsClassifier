import java.io.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import com.sun.javafx.collections.transformation.SortedList;
import javafx.util.Pair;
import jnisvmlight.LabeledFeatureVector;
import jnisvmlight.SVMLightModel;
import jnisvmlight.SVMLightInterface;
import jnisvmlight.TrainingParameters;

public class LingvoSVM {

    private static void trainSVM(String featureFile, String trainPos, String trainNeg, String modelFile, String stopFile) throws Exception {
        SVMLightInterface trainer = new SVMLightInterface();
        List<String> stopList = new ArrayList<String>();
        List<String> features = new ArrayList<String>();
        System.out.print("READING STOP-LIST ... ");
        Scanner sc = new Scanner(new File(stopFile));
        while (sc.hasNext()) {
            stopList.add(sc.next().toLowerCase());
        }
        sc.close();
        System.out.print("DONE\n");
        String ss;
        System.out.print("READING FEATURES ... ");
        sc = new Scanner(new File(featureFile));
        while (sc.hasNext()) {
            ss = sc.next();
            if (!stopList.contains(ss.toLowerCase())) features.add(ss.toLowerCase());
        }
        sc.close();
        System.out.print("DONE\n");
        List<LabeledFeatureVector> trainData = new ArrayList<LabeledFeatureVector>();
        List<List<String>> tmp;
        int[] dims;
        double[] values;
        System.out.print("GENERATING POS ... ");
        int i = 0;
        tmp = conv(trainPos, stopList);
        for (List<String> example: tmp) {
            if (example.size() == 0) continue;
            if (i % Math.round(tmp.size()/10) == 0) System.out.print(10 * i / Math.round(tmp.size()/10)+"% ");
            ++i;
            dims = new int[features.size()];
            values = new double[features.size()];
            for (int j = 0; j < features.size(); ++j) {
                int count = 0;
                for (int k = 0; k < example.size(); ++k) {
                    if (example.get(k).equals(features.get(j))) ++count;
                }
                dims[j] = j + 1;
                values[j] = 1.0*count/example.size();
            }
            trainData.add(new LabeledFeatureVector(+1, dims, values));
        }
        System.out.print("DONE\n");
        System.out.print("GENERATING NEG ... ");
        i = 0;
        tmp = conv(trainNeg, stopList);
        for (List<String> example: tmp) {
            if (example.size() == 0) continue;
            if (i % Math.round(tmp.size()/10) == 0) System.out.print(10 * i / Math.round(tmp.size()/10)+"% ");
            ++i;
            dims = new int[features.size()];
            values = new double[features.size()];
            for (int j = 0; j < features.size(); ++j) {
                int count = 0;
                for (int k = 0; k < example.size(); ++k) {
                    if (example.get(k).equals(features.get(j))) ++count;
                }
                dims[j] = j + 1;
                values[j] = 1.0*count/example.size();
            }
            trainData.add(new LabeledFeatureVector(-1, dims, values));
        }
        System.out.print("DONE\n");
        TrainingParameters tp = new TrainingParameters();
        tp.getLearningParameters().verbosity = 0;
        LabeledFeatureVector[] fvec = new LabeledFeatureVector[trainData.size()];
        fvec = trainData.toArray(fvec);
        System.out.print("TRAINING ... ");
        SVMLightModel  model = trainer.trainModel(fvec, new String[] { "-t", "1", "-d", "2" });
        System.out.print("DONE\n");
        System.out.print("WRITING MODEL TO FILE ... ");
        model.writeModelToFile(modelFile);
        System.out.print("DONE\n");
    }

    public static void test(String featureFile, String testPos, String testNeg, String modelFile, String stopFile) throws Exception {
        SVMLightModel  model = SVMLightModel.readSVMLightModelFromURL(new File(modelFile).toURL());
        List<String> stopList = new ArrayList<String>();
        List<String> features = new ArrayList<String>();
        int tp = 0, tn = 0, fp = 0, fn = 0;
        System.out.print("READING STOP-LIST ... ");
        Scanner sc = new Scanner(new File(stopFile));
        while (sc.hasNext()) {
            stopList.add(sc.next().toLowerCase());
        }
        sc.close();
        System.out.print("DONE\n");
        String ss;
        sc = new Scanner(new File(featureFile));
        System.out.print("READING FEATURES ... ");
        while (sc.hasNext()) {
            ss = sc.next();
            if (!stopList.contains(ss.toLowerCase())) features.add(ss.toLowerCase());
        }
        sc.close();
        System.out.print("DONE\n");
        List<List<String>> tmp;
        int[] dims;
        double[] values;
        System.out.print("TESTING POS ... ");
        tmp = conv(testPos, stopList);
        int i = 0;
        for (List<String> example: tmp) {
            if (example.size() == 0) continue;
            if (i % Math.round(tmp.size()/10) == 0) System.out.print(10 * i / Math.round(tmp.size()/10)+"% ");
            ++i;
            dims = new int[features.size()];
            values = new double[features.size()];
            for (int j = 0; j < features.size(); ++j) {
                int count = 0;
                for (int k = 0; k < example.size(); ++k) {
                    if (example.get(k).equals(features.get(j))) ++count;
                }
                dims[j] = j + 1;
                values[j] = 1.0*count/example.size();
            }
            if (model.classify(new LabeledFeatureVector(0, dims, values)) > 0) {
                ++tp;
            } else {
                ++fn;
            }
        }
        System.out.print("DONE\n");
        System.out.print("TESTING NEG ... ");
        tmp = conv(testNeg, stopList);
        i = 0;
        for (List<String> example: tmp) {
            if (example.size() == 0) continue;
            if (i % Math.round(tmp.size()/10) == 0) System.out.print(10 * i / Math.round(tmp.size()/10)+"% ");
            ++i;
            dims = new int[features.size()];
            values = new double[features.size()];
            for (int j = 0; j < features.size(); ++j) {
                int count = 0;
                for (int k = 0; k < example.size(); ++k) {
                    if (example.get(k).equals(features.get(j))) ++count;
                }
                dims[j] = j + 1;
                values[j] = 1.0*count/example.size();
            }
            if (model.classify(new LabeledFeatureVector(0, dims, values)) < 0) {
                ++tn;
            } else {
                ++fp;
            }
        }
        System.out.print("DONE\n");
        double prec = 1.0*tp/(tp+fp);
        double recall = 1.0*tp/(tp+fn);
        double fsc = 2.0*prec*recall/(prec+recall);
        System.out.println("Precision = " + prec + "\nRecall = " + recall + "\nF-Score = " + fsc+ "\nAccuracy = " + 1.0*(tp+tn)/(tp+tn+fp+fn));
        System.out.println("tp: " + tp + "\ntn: " + tn + "\nfp: " + fp + "\nfn: " + fn + "\nsum: " + (tp+tn+fp+fn));
    }

    private static void trainTest(String featureFile, String trainPos, String trainNeg, String testPos, String testNeg, String stopFile) throws Exception {
        SVMLightInterface trainer = new SVMLightInterface();
        List<String> stopList = new ArrayList<String>();
        List<String> features = new ArrayList<String>();
        int tp = 0, tn = 0, fp = 0, fn = 0;
        System.out.print("READING STOP-LIST ... ");
        Scanner sc = new Scanner(new File(stopFile));
        while (sc.hasNext()) {
            stopList.add(sc.next().toLowerCase());
        }
        sc.close();
        System.out.print("DONE\n");
        String ss;
        System.out.print("READING FEATURES ... ");
        sc = new Scanner(new File(featureFile));
        while (sc.hasNext()) {
            ss = sc.next();
            if (!stopList.contains(ss.toLowerCase())) features.add(ss.toLowerCase());
        }
        sc.close();
        System.out.print("DONE\n");
        List<LabeledFeatureVector> trainData = new ArrayList<LabeledFeatureVector>();
        List<List<String>> tmp;
        int[] dims;
        double[] values;
        System.out.print("GENERATING POS ... ");
        int i = 0;
        tmp = conv(trainPos, stopList);
        for (List<String> example: tmp) {
            if (example.size() == 0) continue;
            if (i % Math.round(tmp.size()/10) == 0) System.out.print(10 * i / Math.round(tmp.size()/10)+"% ");
            ++i;
            dims = new int[features.size()];
            values = new double[features.size()];
            for (int j = 0; j < features.size(); ++j) {
                int count = 0;
                for (int k = 0; k < example.size(); ++k) {
                    if (example.get(k).equals(features.get(j))) ++count;
                }
                dims[j] = j + 1;
                values[j] = 1.0*count/example.size();
            }
            trainData.add(new LabeledFeatureVector(+1, dims, values));
        }
        System.out.print("DONE\n");
        System.out.print("GENERATING NEG ... ");
        i = 0;
        tmp = conv(trainNeg, stopList);
        for (List<String> example: tmp) {
            if (example.size() == 0) continue;
            if (i % Math.round(tmp.size()/10) == 0) System.out.print(10 * i / Math.round(tmp.size()/10)+"% ");
            ++i;
            dims = new int[features.size()];
            values = new double[features.size()];
            for (int j = 0; j < features.size(); ++j) {
                int count = 0;
                for (int k = 0; k < example.size(); ++k) {
                    if (example.get(k).equals(features.get(j))) ++count;
                }
                dims[j] = j + 1;
                values[j] = 1.0*count/example.size();
            }
            trainData.add(new LabeledFeatureVector(-1, dims, values));
        }
        System.out.print("DONE\n");
        TrainingParameters tpar = new TrainingParameters();
        tpar.getLearningParameters().verbosity = 0;
        LabeledFeatureVector[] fvec = new LabeledFeatureVector[trainData.size()];
        fvec = trainData.toArray(fvec);
        System.out.print("TRAINING ... ");
        SVMLightModel  model = trainer.trainModel(fvec, new String[] { "-t", "0", "-d", "2" });
        System.out.print("DONE\n");
        System.out.print("TESTING POS ... ");
        tmp = conv(testPos, stopList);
        i = 0;
        for (List<String> example: tmp) {
            if (example.size() == 0) continue;
            if (i % Math.round(tmp.size()/10) == 0) System.out.print(10 * i / Math.round(tmp.size()/10)+"% ");
            ++i;
            dims = new int[features.size()];
            values = new double[features.size()];
            for (int j = 0; j < features.size(); ++j) {
                int count = 0;
                for (int k = 0; k < example.size(); ++k) {
                    if (example.get(k).equals(features.get(j))) ++count;
                }
                dims[j] = j + 1;
                values[j] = 1.0*count/example.size();
            }
            if (model.classify(new LabeledFeatureVector(0, dims, values)) > 0) {
                ++tp;
            } else {
                ++fn;
            }
        }
        System.out.print("DONE\n");
        System.out.print("TESTING NEG ... ");
        tmp = conv(testNeg, stopList);
        i = 0;
        for (List<String> example: tmp) {
            if (example.size() == 0) continue;
            if (i % Math.round(tmp.size()/10) == 0) System.out.print(10 * i / Math.round(tmp.size()/10)+"% ");
            ++i;
            dims = new int[features.size()];
            values = new double[features.size()];
            for (int j = 0; j < features.size(); ++j) {
                int count = 0;
                for (int k = 0; k < example.size(); ++k) {
                    if (example.get(k).equals(features.get(j))) ++count;
                }
                dims[j] = j + 1;
                values[j] = 1.0*count/example.size();
            }
            if (model.classify(new LabeledFeatureVector(0, dims, values)) < 0) {
                ++tn;
            } else {
                ++fp;
            }
        }
        System.out.print("DONE\n");
        double prec = 1.0*tp/(tp+fp);
        double recall = 1.0*tp/(tp+fn);
        double fsc = 2.0*prec*recall/(prec+recall);
        System.out.println("Precision = " + prec + "\nRecall = " + recall + "\nF-Score = " + fsc+ "\nAccuracy = " + 1.0*(tp+tn)/(tp+tn+fp+fn));
        System.out.println("tp: " + tp + "\ntn: " + tn + "\nfp: " + fp + "\nfn: " + fn + "\nsum: " + (tp + tn + fp + fn));
    }

    private static void makeFeatureList(String listFile1, String listFile2, String resultFile) throws Exception {
        int minFreq = 20, minDif = 150;
        HashMap<String, Integer> result = new HashMap<String, Integer>();
        Scanner scanner = new Scanner(new File(listFile1));
        String tmpStr;
        int tmpInt;
        while (scanner.hasNext()) {
            tmpStr = scanner.next().toLowerCase();
            tmpInt = scanner.nextInt();
            result.put(tmpStr, tmpInt);
        }
        scanner.close();
        scanner = new Scanner(new File(listFile2));
        while (scanner.hasNext()) {
            tmpStr = scanner.next().toLowerCase();
            tmpInt = scanner.nextInt();
            if (result.containsKey(tmpStr) && Math.abs(result.get(tmpStr) - tmpInt) < minDif)  {
                result.put(tmpStr, 0);
                continue;
            }
            result.put(tmpStr, tmpInt);
        }
        scanner.close();
        BufferedWriter outFile = new BufferedWriter(new FileWriter(new File(resultFile)));
        for (Map.Entry<String, Integer> pair: result.entrySet()) {
            if (pair.getValue() >= minFreq) outFile.append(pair.getKey() + "\n");
        }
        outFile.close();
    }

    private static List<List<String>> conv(String fileName, List<String> stopList) throws Exception{
        final Process p = Runtime.getRuntime().exec("mystem.exe "+fileName+" -cwl");
        final BufferedReader ir = new BufferedReader(new InputStreamReader(p.getInputStream()));
        List<List<String>> res = new ArrayList<List<String>>();
        String str;
        StringBuilder rev;
        String ss[];
        boolean hasNext = true;
        List<String> tmp;
        //BufferedWriter wr = new BufferedWriter(new FileWriter(new File("ttt.txt")));
        while (hasNext) {
            tmp = new ArrayList<String>();
            rev = new StringBuilder();
            while ((hasNext = (str = ir.readLine()) != null) && !str.equals("+++"))
            {
                rev.append(str+" ");
            }
            int i = 1;
            String[] sss = rev.toString().split("[\\{\\}]");
            while (i < sss.length) {
                ss = sss[i].split("\\|");
                for (int j = 0; j < ss.length; ++j) {
                    if (!stopList.contains(ss[j].toLowerCase().replace("?", ""))) tmp.add(ss[j].toLowerCase().replace("?", ""));

                   // if (hasNext) wr.append(ss[j].toLowerCase().replace("?", "") + "\n");
                }
                i += 2;
            }
            res.add(tmp);
        }
        //wr.close();
        ir.close();
        p.waitFor();
        return res;

        /*Runtime runtime = Runtime.getRuntime();
        String[] sss = new String[6];
        sss[0] = "mystem.exe";
        sss[1] = "tmpin.txt";
        sss[2] = "tmpout.txt";
        sss[3] = "-l";
        sss[4] = "-w";
        sss[5] = "-n";
        Process proc = runtime.exec(sss);
        proc.waitFor();
        Scanner sc = new Scanner(new File("tmpout.txt"));
        List<String> res = new ArrayList<String>();
        String str;
        String ss;
        while (sc.hasNext()) {
            ss = sc.next();
            if (!stopList.contains(ss.toLowerCase())) res.add(ss.toLowerCase());
        }
        sc.close();
        return res;*/
    }

    private static void wordStat(String inFile, String outFile) throws Exception {
        HashMap<String, Integer> res = new HashMap<String, Integer>();
        Scanner scanner = new Scanner(new File(inFile));
        String str;
        while (scanner.hasNext()) {
            str = scanner.next();
            if (res.containsKey(str)) {
                res.put(str, res.get(str) + 1);
            } else {
                res.put(str, 1);
            }
        }
        scanner.close();
        BufferedWriter wr = new BufferedWriter(new FileWriter(new File(outFile)));
        for (Map.Entry<String, Integer> pair: res.entrySet()) {
            if (pair.getValue() >= 3) wr.append(pair.getKey() + " " + pair.getValue() + "\n");
        }
        wr.close();
    }

    private static void doTry(String featureFile, String stopFile, String modelFile, String inFile) throws Exception {
        System.out.print("READING MODEL ... ");
        SVMLightModel  model = SVMLightModel.readSVMLightModelFromURL(new File(modelFile).toURL());
        System.out.print("DONE\n");
        List<String> stopList = new ArrayList<String>();
        List<String> features = new ArrayList<String>();
        System.out.print("READING STOP-LIST ... ");
        Scanner sc = new Scanner(new File(stopFile));
        while (sc.hasNext()) {
            stopList.add(sc.next().toLowerCase());
        }
        sc.close();
        System.out.print("DONE\n");
        String ss;
        sc = new Scanner(new File(featureFile));
        System.out.print("READING FEATURES ... ");
        while (sc.hasNext()) {
            ss = sc.next();
            if (!stopList.contains(ss.toLowerCase())) features.add(ss.toLowerCase());
        }
        sc.close();
        System.out.print("DONE\n");
        List<List<String>> tmp;
        int[] dims;
        double[] values;
        System.out.print("TESTING ... ");
        tmp = conv(inFile, stopList);
        int i = 0;
        double d;
        int countPos = 0, countNeg = 0;
        for (List<String> example: tmp) {
            ++i;
            dims = new int[features.size()];
            values = new double[features.size()];
            for (int j = 0; j < features.size(); ++j) {
                int count = 0;
                for (int k = 0; k < example.size(); ++k) {
                    if (example.get(k).equals(features.get(j))) ++count;
                }
                dims[j] = j + 1;
                values[j] = 1.0*count/example.size();
            }
            if ((d = model.classify(new LabeledFeatureVector(0, dims, values))) > 0) {
                ++countPos;
                System.out.println("#"+i+": Positive");
            } else {
                ++countNeg;
                System.out.println("#"+i+": Negative");
            }
        }
        System.out.print("DONE\n");
        System.out.println(countPos + " : " + countNeg);
    }

    public static void noRepeats(String inFile, String outFile) throws Exception {
        BufferedWriter wr = new BufferedWriter(new FileWriter(new File(outFile)));
        Scanner scanner = new Scanner(new File(inFile));
        boolean hasNext = true;
        StringBuilder sb;
        String str;
        List<String> list = new ArrayList<String>();
        while (hasNext) {
            sb = new StringBuilder();
            while((hasNext = scanner.hasNext()) && !(str = scanner.next()).equals("+++")) {
                sb.append(str+" ");
            }
            if (!list.contains(sb.toString())) list.add(sb.toString());
        }
        for (int i = 0; i < list.size(); ++i) {
            wr.append(list.get(i)+"\n");
            if(i < list.size() - 1) wr.append("+++\n");
        }
        scanner.close();
        wr.close();
    }

    public static void main(String[] args) throws Exception {

        if (args[0].equals("train")) {
            trainSVM(args[1], args[2], args[3], args[4], args[5]);
        } else if (args[0].equals("test")) {
            test(args[1], args[2], args[3], args[4], args[5]);
        } else if (args[0].equals("feature")) {
            makeFeatureList(args[1], args[2], args[3]);
        } else if (args[0].equals("wordstat")) {
            wordStat(args[1], args[2]);
        } else if (args[0].equals("traintest")) {
            trainTest(args[1], args[2], args[3], args[4], args[5], args[6]);
        } else if (args[0].equals("doTry")) {
            doTry(args[1], args[2], args[3], args[4]);
        } else if (args[0].equals("noRepeats")) {
            noRepeats(args[1], args[2]);
        }
    }
}

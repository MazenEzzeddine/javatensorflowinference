
import com.google.common.io.ByteStreams;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.Tensors;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.sql.SQLOutput;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class LabelImage {


    static List<String> labels;
  public static void main(String[] args) throws Exception {
   /* if (args.length < 1) {
      System.err.println("USAGE: Provide a list of image filenames");
      System.exit(1);
    }*/
    final List<String> images = List.of("terrier1u.jpg", "porcupine.jpg", "terrier2.jpg", "whale.jpg",
            "terrier1u.jpg", "porcupine.jpg", "terrier2.jpg", "whale.jpg",
            "terrier1u.jpg", "porcupine.jpg", "terrier2.jpg", "whale.jpg",
            "terrier1u.jpg", "porcupine.jpg", "terrier2.jpg", "whale.jpg",
            "terrier1u.jpg", "porcupine.jpg", "terrier2.jpg", "whale.jpg",
            "terrier1u.jpg", "porcupine.jpg", "terrier2.jpg", "whale.jpg",
            "terrier1u.jpg", "porcupine.jpg", "terrier2.jpg", "whale.jpg",
            "terrier1u.jpg", "porcupine.jpg", "terrier2.jpg", "whale.jpg",
            "terrier1u.jpg", "porcupine.jpg", "terrier2.jpg", "whale.jpg",
            "terrier1u.jpg", "porcupine.jpg", "terrier2.jpg", "whale.jpg",
            "terrier1u.jpg", "porcupine.jpg", "terrier2.jpg", "whale.jpg",
            "terrier1u.jpg", "porcupine.jpg", "terrier2.jpg", "whale.jpg",
            "terrier1u.jpg", "porcupine.jpg", "terrier2.jpg", "whale.jpg",
            "terrier1u.jpg", "porcupine.jpg", "terrier2.jpg", "whale.jpg",
            "terrier1u.jpg", "porcupine.jpg", "terrier2.jpg", "whale.jpg",
            "terrier1u.jpg", "porcupine.jpg", "terrier2.jpg", "whale.jpg",
            "terrier1u.jpg", "porcupine.jpg", "terrier2.jpg", "whale.jpg",
            "terrier1u.jpg", "porcupine.jpg", "terrier2.jpg", "whale.jpg",
            "terrier1u.jpg", "porcupine.jpg", "terrier2.jpg", "whale.jpg",
            "terrier1u.jpg", "porcupine.jpg", "terrier2.jpg", "img.png");

    labels = loadLabels();
   // infer(images);
    inferThreads(images);
  }

    private static void inferThreads(List<String> images) {

        ExecutorService executorService = Executors.newFixedThreadPool(8);

        for ( int i = 0; i < 8; i++) {

            int finalI = i;
            executorService.execute(() -> {
                try {
                    infer(images.subList((finalI *10), ((finalI +1)*10)));
                } catch (IOException e) {
                    e.printStackTrace();
                }

            });

        }


        executorService.shutdown();
    }

    private static byte[] loadGraphDef() throws IOException {
    try (InputStream is = LabelImage.class.getClassLoader().getResourceAsStream("graph.pb")) {
      return ByteStreams.toByteArray(is);
    }
  }

  private static ArrayList<String> loadLabels() throws IOException {
    ArrayList<String> labels = new ArrayList<String>();
    String line;
    final InputStream is = LabelImage.class.getClassLoader().getResourceAsStream("labels.txt");
    try (BufferedReader reader = new BufferedReader(new InputStreamReader(is))) {
      while ((line = reader.readLine()) != null) {
        labels.add(line);
      }
    }
    return labels;
  }

  private static int argmax(float[] probabilities) {
    int best = 0;
    for (int i = 1; i < probabilities.length; ++i) {
      if (probabilities[i] > probabilities[best]) {
        best = i;
      }
    }
    return best;
  }


  private static void infer(List<String> images) throws IOException {
      try (Graph graph = new Graph();
           Session session = new Session(graph)) {
          graph.importGraphDef(loadGraphDef());

          long start = System.currentTimeMillis();

          for (String image: images) {
              float[] probabilities = null;
              byte[] bytes = Files.readAllBytes(Paths.get(image));
              try (Tensor<String> input = Tensors.create(bytes);
                   Tensor<Float> output =
                           session
                                   .runner()
                                   .feed("encoded_image_bytes", input)
                                   .fetch("probabilities")
                                   .run()
                                   .get(0)
                                   .expect(Float.class)) {
                  if (probabilities == null) {
                      probabilities = new float[(int) output.shape()[0]];
                  }
                  output.copyTo(probabilities);
                  int label = argmax(probabilities);
                  System.out.printf(
                          "%-30s --> %-15s (%.2f%% likely)\n",
                          image, labels.get(label), probabilities[label] * 100.0);
              }


          }
          long end  =System.currentTimeMillis();
          System.out.println("loading and calssifying " +  images.size()  +  " images took : " + (end-start));

      }
  }
}

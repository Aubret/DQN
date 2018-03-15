package main.java.fr.univlyon1.networks;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Approximator {
    INDArray getOneResult(INDArray data);
    void learn(INDArray input,INDArray labels,int number);
    Approximator clone();
    Approximator clone(boolean Listener);
    void stop();
}

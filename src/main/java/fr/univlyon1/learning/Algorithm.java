package main.java.fr.univlyon1.learning;

import main.java.fr.univlyon1.environment.Interaction;
import main.java.fr.univlyon1.environment.Observation;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface Algorithm<A> {
    void step(INDArray input, A action, INDArray results);
    void evaluate(INDArray input, Double reward);
}

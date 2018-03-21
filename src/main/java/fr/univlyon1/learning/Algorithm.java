package fr.univlyon1.learning;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Algorithm<A> {
    void step(INDArray input, A action, INDArray results);
    void evaluate(INDArray input, Double reward);
    void epoch();

}

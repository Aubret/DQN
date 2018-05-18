package fr.univlyon1.learning;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Algorithm<A> {
    INDArray behave(INDArray input);
    void step(INDArray input, A action,Double dt);
    void evaluate(INDArray input, Double reward);
    void epoch();
    void learn();
    Double getScore();

}

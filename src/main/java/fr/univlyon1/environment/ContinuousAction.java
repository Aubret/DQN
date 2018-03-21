package fr.univlyon1.environment;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface ContinuousAction {
    void constructAction(INDArray values);
    INDArray DeconstructAction();
}

package fr.univlyon1.environment.space;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface ContinuousAction{
    void constructAction(INDArray values);
    INDArray DeconstructAction();
    void unNormalize();
    ContinuousAction copy();
}

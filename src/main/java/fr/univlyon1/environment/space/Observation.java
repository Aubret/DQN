package fr.univlyon1.environment.space;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Observation {
    INDArray getData();
    void computeData();
}

package fr.univlyon1.networks.lossFunctions;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface SaveScore {
    INDArray getLastScoreArray() ;
    INDArray getValues() ;
}

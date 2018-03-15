package main.java.fr.univlyon1.actorcritic.policy;

import main.java.fr.univlyon1.environment.Observation;
import org.nd4j.linalg.api.ndarray.INDArray;

public interface Policy{
    Integer getAction(INDArray results);
}
